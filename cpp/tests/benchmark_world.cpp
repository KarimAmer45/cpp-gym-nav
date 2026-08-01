#include "nav/world.hpp"

#include <chrono>
#include <cstddef>
#include <iostream>

int main() {
  constexpr std::size_t iterations = 1'000'000U;
  nav::World world;
  world.reset(5951U);
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < iterations; ++i) {
    const auto &result = world.step({0.4F, 0.1F});
    if (result.collision || result.success)
      world.reset(static_cast<std::uint64_t>(i));
  }
  const auto elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::cout << "iterations=" << iterations << '\n'
            << "seconds=" << elapsed << '\n'
            << "steps_per_second=" << static_cast<double>(iterations) / elapsed << '\n'
            << "nanoseconds_per_step=" << elapsed * 1.0e9 / static_cast<double>(iterations) << '\n';
}
