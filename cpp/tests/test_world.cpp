#include "nav/world.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {
void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}
} // namespace

int main() {
  nav::Config config;
  nav::World first(config);
  nav::World second(config);

  const auto first_reset = first.reset(5951U);
  const auto second_reset = second.reset(5951U);
  require(first_reset == second_reset, "same seed must produce the same reset observation");
  require(first_reset.size() == 6U + config.lidar_beams, "observation shape must match config");

  const nav::Action action{100.0F, -100.0F};
  const auto first_step = first.step(action);
  const auto second_step = second.step(action);
  require(first_step.observation == second_step.observation, "same trajectory must replay exactly");
  require(std::abs(first_step.applied_action.linear) <= config.max_linear_speed,
          "linear action must be clamped");
  require(std::abs(first_step.applied_action.angular) <= config.max_angular_speed,
          "angular action must be clamped");
  for (float value : first_step.observation) {
    require(std::isfinite(value), "observations must be finite");
    require(value >= -1.0F && value <= 1.0F, "observations must be normalized");
  }

  // A reused world must replay a seed identically to a fresh one. Noise draws
  // leave cached state in std::normal_distribution; reset() must clear it. An
  // odd, dropout-free beam count guarantees an odd number of prior draws, so
  // the cache is occupied regardless of the standard library's Box-Muller
  // ordering.
  nav::Config noisy_config;
  noisy_config.sensor_noise = true;
  noisy_config.lidar_beams = 15U;
  noisy_config.lidar_dropout_probability = 0.0F;
  nav::World fresh_noisy(noisy_config);
  nav::World reused_noisy(noisy_config);
  const auto fresh_noisy_reset = fresh_noisy.reset(7U);
  reused_noisy.reset(1U); // leaves an odd number of cached normal draws
  require(reused_noisy.reset(7U) == fresh_noisy_reset,
          "a reused world must replay a seed identically to a fresh world");

  nav::Config delayed_config;
  delayed_config.action_delay = true;
  delayed_config.action_delay_steps = 1U;
  nav::World delayed(delayed_config);
  delayed.reset(1U);
  const auto delayed_step = delayed.step({1.0F, 0.0F});
  require(delayed_step.applied_action.linear == 0.0F, "one-step delay must initially apply zero");

  // Raycast geometry against known scenes, independent of the randomized layout.
  const float pi = 3.14159265358979F;
  const nav::Circle ahead{2.0F, 0.0F, 0.5F};
  require(std::abs(nav::ray_distance(0.0F, 0.0F, 0.0F, &ahead, 1U, 100.0F, 5.0F) - 1.5F) < 1.0e-4F,
          "beam must hit the obstacle surface at range 1.5");
  require(std::abs(nav::ray_distance(0.0F, 0.0F, pi, &ahead, 1U, 3.0F, 5.0F) - 3.0F) < 1.0e-4F,
          "beam pointing away from the obstacle must reach the wall");
  require(std::abs(nav::ray_distance(0.0F, 0.0F, 0.0F, nullptr, 0U, 2.5F, 5.0F) - 2.5F) < 1.0e-4F,
          "empty scene must return the wall distance");
  require(std::abs(nav::ray_distance(0.0F, 0.0F, 0.0F, nullptr, 0U, 100.0F, 5.0F) - 5.0F) < 1.0e-4F,
          "distant scene must clamp to the sensor max range");

  std::cout << "All nav_core tests passed\n";
  return EXIT_SUCCESS;
}
