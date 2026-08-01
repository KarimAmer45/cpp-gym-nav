#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "nav/world.hpp"

namespace py = pybind11;

namespace {
py::array_t<float> as_array(const std::vector<float> &values) {
  py::array_t<float> output(values.size());
  std::memcpy(output.mutable_data(), values.data(), values.size() * sizeof(float));
  return output;
}

nav::Action
parse_action(const py::array_t<float, py::array::c_style | py::array::forcecast> &input) {
  const auto buffer = input.request();
  if (buffer.ndim != 1 || buffer.size != 2) {
    throw std::invalid_argument("action must be a one-dimensional float array with two values");
  }
  const auto *values = static_cast<const float *>(buffer.ptr);
  return {values[0], values[1]};
}

class BatchWorld {
public:
  BatchWorld(std::size_t count, const nav::Config &config) : config_(config) {
    if (count == 0U) {
      throw std::invalid_argument("count must be positive");
    }
    worlds_.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      worlds_.emplace_back(config_);
    }
  }

  [[nodiscard]] std::size_t size() const noexcept { return worlds_.size(); }
  [[nodiscard]] std::size_t observation_size() const noexcept {
    return worlds_.front().observation_size();
  }
  [[nodiscard]] const nav::Config &config() const noexcept { return config_; }
  nav::World &world(std::size_t index) noexcept { return worlds_[index]; }

private:
  nav::Config config_;
  std::vector<nav::World> worlds_;
};
} // namespace

PYBIND11_MODULE(_core, module) {
  module.doc() = "C++ core for the cpp-gym-nav environment";

  py::class_<nav::Config>(module, "Config")
      .def(py::init<>())
      .def_readwrite("arena_half_extent", &nav::Config::arena_half_extent)
      .def_readwrite("robot_radius", &nav::Config::robot_radius)
      .def_readwrite("time_step", &nav::Config::time_step)
      .def_readwrite("max_linear_speed", &nav::Config::max_linear_speed)
      .def_readwrite("max_angular_speed", &nav::Config::max_angular_speed)
      .def_readwrite("goal_radius", &nav::Config::goal_radius)
      .def_readwrite("lidar_max_range", &nav::Config::lidar_max_range)
      .def_readwrite("lidar_beams", &nav::Config::lidar_beams)
      .def_readwrite("sensor_noise", &nav::Config::sensor_noise)
      .def_readwrite("lidar_noise_std", &nav::Config::lidar_noise_std)
      .def_readwrite("lidar_dropout_probability", &nav::Config::lidar_dropout_probability)
      .def_readwrite("lidar_quantization", &nav::Config::lidar_quantization)
      .def_readwrite("actuator_limits", &nav::Config::actuator_limits)
      .def_readwrite("max_linear_acceleration", &nav::Config::max_linear_acceleration)
      .def_readwrite("max_angular_acceleration", &nav::Config::max_angular_acceleration)
      .def_readwrite("action_delay", &nav::Config::action_delay)
      .def_readwrite("action_delay_steps", &nav::Config::action_delay_steps)
      .def_readwrite("dynamics_randomization", &nav::Config::dynamics_randomization)
      .def_readwrite("slip_std", &nav::Config::slip_std);

  py::class_<nav::World>(module, "World")
      .def(py::init<nav::Config>(), py::arg("config") = nav::Config{})
      .def("reset",
           [](nav::World &world, std::uint64_t seed) {
             std::vector<float> observation;
             {
               py::gil_scoped_release release;
               observation = world.reset(seed);
             }
             return as_array(observation);
           })
      .def("step",
           [](nav::World &world,
              const py::array_t<float, py::array::c_style | py::array::forcecast> &input) {
             const nav::Action action = parse_action(input);
             nav::StepResult result;
             {
               py::gil_scoped_release release;
               result = world.step(action);
             }
             py::dict output;
             output["observation"] = as_array(result.observation);
             output["reward"] = result.reward;
             output["success"] = result.success;
             output["collision"] = result.collision;
             output["distance_to_goal"] = result.distance_to_goal;
             output["applied_action"] =
                 py::make_tuple(result.applied_action.linear, result.applied_action.angular);
             return output;
           })
      .def_property_readonly("observation_size", &nav::World::observation_size)
      .def_property_readonly("seed", &nav::World::seed)
      .def_property_readonly("pose",
                             [](const nav::World &world) {
                               const auto &pose = world.pose();
                               return py::make_tuple(pose.x, pose.y, pose.heading);
                             })
      .def_property_readonly("goal",
                             [](const nav::World &world) {
                               const auto &goal = world.goal();
                               return py::make_tuple(goal.x, goal.y);
                             })
      .def_property_readonly("obstacles", [](const nav::World &world) {
        py::list obstacles;
        for (const auto &obstacle : world.obstacles()) {
          obstacles.append(py::make_tuple(obstacle.x, obstacle.y, obstacle.radius));
        }
        return obstacles;
      });

  py::class_<BatchWorld>(module, "BatchWorld")
      .def(py::init<std::size_t, const nav::Config &>(), py::arg("count"),
           py::arg("config") = nav::Config{})
      .def("reset",
           [](BatchWorld &batch, std::uint64_t seed) {
             py::array_t<float> observations({static_cast<py::ssize_t>(batch.size()),
                                              static_cast<py::ssize_t>(batch.observation_size())});
             auto *output = observations.mutable_data();
             {
               py::gil_scoped_release release;
               for (std::size_t i = 0; i < batch.size(); ++i) {
                 const auto &observation = batch.world(i).reset(seed + i);
                 std::memcpy(output + i * batch.observation_size(), observation.data(),
                             observation.size() * sizeof(float));
               }
             }
             return observations;
           })
      .def("step",
           [](BatchWorld &batch,
              const py::array_t<float, py::array::c_style | py::array::forcecast> &actions) {
             const auto input = actions.request();
             if (input.ndim != 2 || input.shape[0] != static_cast<py::ssize_t>(batch.size()) ||
                 input.shape[1] != 2) {
               throw std::invalid_argument("actions must have shape (count, 2)");
             }
             const auto *action_values = static_cast<const float *>(input.ptr);
             py::array_t<float> observations({static_cast<py::ssize_t>(batch.size()),
                                              static_cast<py::ssize_t>(batch.observation_size())});
             py::array_t<float> rewards(batch.size());
             py::array_t<bool> successes(batch.size());
             py::array_t<bool> collisions(batch.size());
             auto *observation_values = observations.mutable_data();
             auto *reward_values = rewards.mutable_data();
             auto *success_values = successes.mutable_data();
             auto *collision_values = collisions.mutable_data();
             const auto &config = batch.config();
             {
               py::gil_scoped_release release;
               for (std::size_t i = 0; i < batch.size(); ++i) {
                 const nav::Action action{
                     action_values[2U * i] * config.max_linear_speed,
                     action_values[2U * i + 1U] * config.max_angular_speed,
                 };
                 const auto &result = batch.world(i).step(action);
                 std::memcpy(observation_values + i * batch.observation_size(),
                             result.observation.data(), result.observation.size() * sizeof(float));
                 reward_values[i] = result.reward;
                 success_values[i] = result.success;
                 collision_values[i] = result.collision;
               }
             }
             py::dict output;
             output["observations"] = std::move(observations);
             output["rewards"] = std::move(rewards);
             output["successes"] = std::move(successes);
             output["collisions"] = std::move(collisions);
             return output;
           })
      .def("reset_at",
           [](BatchWorld &batch, std::size_t index, std::uint64_t seed) {
             if (index >= batch.size()) {
               throw std::out_of_range("index must be < count");
             }
             py::array_t<float> observation(static_cast<py::ssize_t>(batch.observation_size()));
             auto *output = observation.mutable_data();
             {
               py::gil_scoped_release release;
               const auto &values = batch.world(index).reset(seed);
               std::memcpy(output, values.data(), values.size() * sizeof(float));
             }
             return observation;
           })
      .def_property_readonly("count", &BatchWorld::size)
      .def_property_readonly("observation_size", &BatchWorld::observation_size);
}
