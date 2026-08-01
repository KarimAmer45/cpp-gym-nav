cmake_minimum_required(VERSION 3.24)
project(cpp_gym_nav VERSION 0.1.0 LANGUAGES CXX)

option(CPP_GYM_NAV_BUILD_TESTS "Build C++ tests" ON)
option(CPP_GYM_NAV_BUILD_BENCHMARK "Build the native benchmark" ON)
option(CPP_GYM_NAV_ENABLE_SANITIZERS "Enable address and undefined-behavior sanitizers" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(PYBIND11_FINDPYTHON ON)

add_library(nav_core STATIC
  cpp/src/world.cpp
)
target_include_directories(nav_core PUBLIC cpp/include)
set_target_properties(nav_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CPP_GYM_NAV_ENABLE_SANITIZERS AND NOT MSVC)
  target_compile_options(nav_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(nav_core PUBLIC -fsanitize=address,undefined)
endif()

if(MSVC)
  target_compile_options(nav_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(nav_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_core bindings/bindings.cpp)
target_link_libraries(_core PRIVATE nav_core)
install(TARGETS _core LIBRARY DESTINATION cpp_gym_nav RUNTIME DESTINATION cpp_gym_nav)

include(CTest)
if(BUILD_TESTING AND CPP_GYM_NAV_BUILD_TESTS)
  add_executable(nav_core_tests cpp/tests/test_world.cpp)
  target_link_libraries(nav_core_tests PRIVATE nav_core)
  add_test(NAME nav_core_tests COMMAND nav_core_tests)
endif()

if(CPP_GYM_NAV_BUILD_BENCHMARK)
  add_executable(nav_core_benchmark cpp/tests/benchmark_world.cpp)
  target_link_libraries(nav_core_benchmark PRIVATE nav_core)
endif()
