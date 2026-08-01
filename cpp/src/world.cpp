#include "nav/world.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nav {
namespace {
constexpr float kPi = 3.14159265358979323846F;
constexpr float kEpsilon = 1.0e-6F;
} // namespace

World::World(Config config) : config_(config) {
  if (config_.lidar_beams < 4U || config_.lidar_beams > kMaxLidarBeams) {
    throw std::invalid_argument("lidar_beams must be in [4, 64]");
  }
  if (config_.arena_half_extent <= 1.0F || config_.robot_radius <= 0.0F ||
      config_.time_step <= 0.0F || config_.lidar_max_range <= 0.0F) {
    throw std::invalid_argument("arena, robot, time-step, and lidar dimensions must be positive");
  }
  if (config_.action_delay_steps > kMaxActionDelay) {
    throw std::invalid_argument("action_delay_steps must be in [0, 2]");
  }
  observation_.resize(observation_size());
  result_.observation.resize(observation_size());
  reset(0U);
}

float World::clamp(float value, float lower, float upper) noexcept {
  return std::max(lower, std::min(value, upper));
}

float World::wrap_angle(float angle) noexcept { return std::remainder(angle, 2.0F * kPi); }

void World::reset_obstacles() {
  const float jitter = config_.dynamics_randomization ? 0.35F : 0.0F;
  const auto j = [this, jitter]() { return jitter * (2.0F * uniform_(rng_) - 1.0F); };
  obstacles_ = {{{-1.7F + j(), -0.8F + j(), 0.65F},
                 {0.0F + j(), 1.3F + j(), 0.70F},
                 {1.8F + j(), -1.1F + j(), 0.60F},
                 {-0.2F + j(), -2.6F + j(), 0.55F},
                 {2.7F + j(), 2.4F + j(), 0.50F}}};
}

bool World::collides(float x, float y) const noexcept {
  const float boundary = config_.arena_half_extent - config_.robot_radius;
  if (std::abs(x) >= boundary || std::abs(y) >= boundary) {
    return true;
  }
  for (const auto &obstacle : obstacles_) {
    const float dx = x - obstacle.x;
    const float dy = y - obstacle.y;
    const float clearance = config_.robot_radius + obstacle.radius;
    if (dx * dx + dy * dy <= clearance * clearance) {
      return true;
    }
  }
  return false;
}

void World::sample_start_and_goal() {
  const float limit = config_.arena_half_extent - config_.robot_radius - 0.15F;
  std::uniform_real_distribution<float> position(-limit, limit);
  std::uniform_real_distribution<float> angle(-kPi, kPi);

  for (int attempt = 0; attempt < 1000; ++attempt) {
    pose_ = {position(rng_), position(rng_), angle(rng_)};
    if (!collides(pose_.x, pose_.y)) {
      break;
    }
  }
  for (int attempt = 0; attempt < 1000; ++attempt) {
    goal_ = {position(rng_), position(rng_), 0.0F};
    const float dx = goal_.x - pose_.x;
    const float dy = goal_.y - pose_.y;
    if (!collides(goal_.x, goal_.y) && std::hypot(dx, dy) >= 3.0F) {
      return;
    }
  }
  throw std::runtime_error("could not sample a valid start/goal pair");
}

const std::vector<float> &World::reset(std::uint64_t seed) {
  seed_ = seed;
  rng_.seed(seed_);
  // Distributions can cache internal state across draws (e.g. the second
  // Box-Muller value in std::normal_distribution). Clear it so that a reused
  // World replays bit-for-bit identically to a freshly constructed one.
  normal_.reset();
  uniform_.reset();
  velocity_ = {};
  action_queue_.fill({});
  reset_obstacles();
  sample_start_and_goal();
  fill_observation();
  return observation_;
}

float World::distance_to_goal() const noexcept {
  return std::hypot(goal_.x - pose_.x, goal_.y - pose_.y);
}

Action World::prepare_action(Action requested) noexcept {
  if (!std::isfinite(requested.linear)) {
    requested.linear = 0.0F;
  }
  if (!std::isfinite(requested.angular)) {
    requested.angular = 0.0F;
  }
  requested.linear = clamp(requested.linear, -config_.max_linear_speed, config_.max_linear_speed);
  requested.angular =
      clamp(requested.angular, -config_.max_angular_speed, config_.max_angular_speed);

  Action delayed = requested;
  if (config_.action_delay && config_.action_delay_steps > 0U) {
    for (std::size_t i = config_.action_delay_steps; i > 0U; --i) {
      action_queue_[i] = action_queue_[i - 1U];
    }
    action_queue_[0] = requested;
    delayed = action_queue_[config_.action_delay_steps];
  }

  if (config_.actuator_limits) {
    const float dv = config_.max_linear_acceleration * config_.time_step;
    const float dw = config_.max_angular_acceleration * config_.time_step;
    delayed.linear = clamp(delayed.linear, velocity_.linear - dv, velocity_.linear + dv);
    delayed.angular = clamp(delayed.angular, velocity_.angular - dw, velocity_.angular + dw);
  }
  return delayed;
}

const StepResult &World::step(Action action) {
  const float old_distance = distance_to_goal();
  const Action applied = prepare_action(action);
  velocity_ = applied;

  float linear = applied.linear;
  float angular = applied.angular;
  if (config_.dynamics_randomization) {
    linear *= 1.0F + config_.slip_std * normal_(rng_);
    angular *= 1.0F + config_.slip_std * normal_(rng_);
  }

  const float next_heading = wrap_angle(pose_.heading + angular * config_.time_step);
  const float next_x = pose_.x + linear * std::cos(next_heading) * config_.time_step;
  const float next_y = pose_.y + linear * std::sin(next_heading) * config_.time_step;
  const bool collision = collides(next_x, next_y);
  if (!collision) {
    pose_ = {next_x, next_y, next_heading};
  }

  const float new_distance = distance_to_goal();
  const bool success = new_distance <= config_.goal_radius;
  const float effort = 0.5F * (std::abs(applied.linear) / config_.max_linear_speed +
                               std::abs(applied.angular) / config_.max_angular_speed);
  float reward = 2.0F * (old_distance - new_distance) - 0.01F - 0.005F * effort;
  if (success) {
    reward += 10.0F;
  }
  if (collision) {
    reward -= 10.0F;
  }

  fill_observation();
  result_.observation = observation_;
  result_.reward = reward;
  result_.success = success;
  result_.collision = collision;
  result_.distance_to_goal = new_distance;
  result_.applied_action = applied;
  return result_;
}

float World::raycast(float angle) const noexcept {
  const float dx = std::cos(angle);
  const float dy = std::sin(angle);
  float distance = config_.lidar_max_range;
  const float bound = config_.arena_half_extent - config_.robot_radius;

  if (std::abs(dx) > kEpsilon) {
    const float tx1 = (bound - pose_.x) / dx;
    const float tx2 = (-bound - pose_.x) / dx;
    if (tx1 > 0.0F)
      distance = std::min(distance, tx1);
    if (tx2 > 0.0F)
      distance = std::min(distance, tx2);
  }
  if (std::abs(dy) > kEpsilon) {
    const float ty1 = (bound - pose_.y) / dy;
    const float ty2 = (-bound - pose_.y) / dy;
    if (ty1 > 0.0F)
      distance = std::min(distance, ty1);
    if (ty2 > 0.0F)
      distance = std::min(distance, ty2);
  }

  for (const auto &obstacle : obstacles_) {
    const float ox = pose_.x - obstacle.x;
    const float oy = pose_.y - obstacle.y;
    const float b = ox * dx + oy * dy;
    const float c = ox * ox + oy * oy - obstacle.radius * obstacle.radius;
    const float discriminant = b * b - c;
    if (discriminant < 0.0F)
      continue;
    const float root = std::sqrt(discriminant);
    const float near_hit = -b - root;
    const float far_hit = -b + root;
    const float hit = near_hit > 0.0F ? near_hit : far_hit;
    if (hit > 0.0F)
      distance = std::min(distance, hit);
  }
  return clamp(distance, 0.0F, config_.lidar_max_range);
}

void World::fill_observation() {
  const float world_dx = goal_.x - pose_.x;
  const float world_dy = goal_.y - pose_.y;
  const float c = std::cos(pose_.heading);
  const float s = std::sin(pose_.heading);
  const float scale = 2.0F * config_.arena_half_extent;
  observation_[0] = clamp((c * world_dx + s * world_dy) / scale, -1.0F, 1.0F);
  observation_[1] = clamp((-s * world_dx + c * world_dy) / scale, -1.0F, 1.0F);
  observation_[2] = clamp(std::hypot(world_dx, world_dy) / (scale * std::sqrt(2.0F)), 0.0F, 1.0F);
  observation_[3] = wrap_angle(std::atan2(world_dy, world_dx) - pose_.heading) / kPi;
  observation_[4] = clamp(velocity_.linear / config_.max_linear_speed, -1.0F, 1.0F);
  observation_[5] = clamp(velocity_.angular / config_.max_angular_speed, -1.0F, 1.0F);

  for (std::size_t i = 0; i < config_.lidar_beams; ++i) {
    const float fraction = static_cast<float>(i) / static_cast<float>(config_.lidar_beams);
    float reading = raycast(pose_.heading - kPi + 2.0F * kPi * fraction);
    if (config_.sensor_noise) {
      if (uniform_(rng_) < config_.lidar_dropout_probability) {
        reading = config_.lidar_max_range;
      } else {
        reading += config_.lidar_noise_std * normal_(rng_);
        if (config_.lidar_quantization > 0.0F) {
          reading = std::round(reading / config_.lidar_quantization) * config_.lidar_quantization;
        }
      }
    }
    observation_[6U + i] = clamp(reading / config_.lidar_max_range, 0.0F, 1.0F);
  }
}

} // namespace nav
