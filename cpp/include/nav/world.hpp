#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace nav {

constexpr std::size_t kMaxLidarBeams = 64;
constexpr std::size_t kObstacleCount = 5;
constexpr std::size_t kMaxActionDelay = 2;

struct Action {
  float linear{0.0F};
  float angular{0.0F};
};

struct Pose {
  float x{0.0F};
  float y{0.0F};
  float heading{0.0F};
};

struct Circle {
  float x{0.0F};
  float y{0.0F};
  float radius{0.5F};
};

struct Config {
  float arena_half_extent{5.0F};
  float robot_radius{0.22F};
  float time_step{0.1F};
  float max_linear_speed{1.25F};
  float max_angular_speed{2.0F};
  float goal_radius{0.35F};
  float lidar_max_range{5.0F};
  std::size_t lidar_beams{16};
  bool random_obstacles{true};

  bool sensor_noise{false};
  float lidar_noise_std{0.02F};
  float lidar_dropout_probability{0.02F};
  float lidar_quantization{0.01F};

  bool actuator_limits{false};
  float max_linear_acceleration{2.0F};
  float max_angular_acceleration{4.0F};

  bool action_delay{false};
  std::size_t action_delay_steps{1};

  bool dynamics_randomization{false};
  float slip_std{0.025F};
};

struct StepResult {
  std::vector<float> observation;
  float reward{0.0F};
  bool success{false};
  bool collision{false};
  float distance_to_goal{0.0F};
  Action applied_action{};
};

// Distance from (x, y) along `angle` to the nearest obstacle circle or the
// axis-aligned square wall at +/- wall_bound, clamped to [0, max_range].
// Free function (no World state) so the raycast geometry can be unit-tested.
[[nodiscard]] float ray_distance(float x, float y, float angle, const Circle *obstacles,
                                 std::size_t obstacle_count, float wall_bound,
                                 float max_range) noexcept;

class World {
public:
  explicit World(Config config = {});

  const std::vector<float> &reset(std::uint64_t seed);
  const StepResult &step(Action action);

  [[nodiscard]] const Config &config() const noexcept { return config_; }
  [[nodiscard]] const Pose &pose() const noexcept { return pose_; }
  [[nodiscard]] const Pose &goal() const noexcept { return goal_; }
  [[nodiscard]] const std::array<Circle, kObstacleCount> &obstacles() const noexcept {
    return obstacles_;
  }
  [[nodiscard]] std::size_t observation_size() const noexcept { return 6U + config_.lidar_beams; }
  [[nodiscard]] std::uint64_t seed() const noexcept { return seed_; }

private:
  static float clamp(float value, float lower, float upper) noexcept;
  static float wrap_angle(float angle) noexcept;
  [[nodiscard]] float distance_to_goal() const noexcept;
  [[nodiscard]] bool collides(float x, float y) const noexcept;
  [[nodiscard]] float raycast(float angle) const noexcept;
  Action prepare_action(Action requested) noexcept;
  void fill_observation();
  void reset_obstacles();
  void sample_start_and_goal();

  Config config_;
  Pose pose_{};
  Pose goal_{};
  Action velocity_{};
  std::array<Action, kMaxActionDelay + 1U> action_queue_{};
  std::array<Circle, kObstacleCount> obstacles_{};
  std::mt19937_64 rng_{};
  std::normal_distribution<float> normal_{0.0F, 1.0F};
  std::uniform_real_distribution<float> uniform_{0.0F, 1.0F};
  std::uint64_t seed_{0};
  std::vector<float> observation_;
  StepResult result_;
};

} // namespace nav
