"""Configure, build, and execute the standalone C++ tests portably."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pybind11


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    build = root / "build" / "native-tests"
    config = "Release"
    cmake = shutil.which("cmake")
    if cmake is None:
        raise SystemExit("cmake is required")
    subprocess.run(
        [
            cmake,
            "-S",
            str(root),
            "-B",
            str(build),
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
            f"-DPython_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=ON",
            "-DCPP_GYM_NAV_BUILD_BENCHMARK=ON",
        ],
        check=True,
    )
    subprocess.run([cmake, "--build", str(build), "--config", config], check=True)
    subprocess.run(
        ["ctest", "--test-dir", str(build), "-C", config, "--output-on-failure"], check=True
    )
    executable = (
        build
        / config
        / ("nav_core_benchmark.exe" if sys.platform == "win32" else "nav_core_benchmark")
    )
    if not executable.exists():
        executable = build / "nav_core_benchmark"
    subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
