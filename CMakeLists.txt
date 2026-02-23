# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Wimblerobotics

cmake_minimum_required(VERSION 3.8)
project(sigyn_oakd_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Dependencies ────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
# OakdDetection.msg lives in sigyn_interfaces — no custom messages here.
find_package(sigyn_interfaces REQUIRED)
find_package(vision_msgs REQUIRED)

# ── Python package (makes 'sigyn_oakd_detection' importable) ─────────────────
ament_python_install_package(${PROJECT_NAME})

# ── Python node ──────────────────────────────────────────────────────────────
install(PROGRAMS
  sigyn_oakd_detection/oakd_detector_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ── Installed resources ──────────────────────────────────────────────────────
install(DIRECTORY
  launch
  config
  models
  DESTINATION share/${PROJECT_NAME}
)

# ── Tests ────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Exclude linters we configure manually below.
  list(APPEND AMENT_LINT_AUTO_EXCLUDE
    ament_cmake_uncrustify
    ament_cmake_copyright
    ament_cmake_flake8
    ament_cmake_pep257
  )
  ament_lint_auto_find_test_dependencies()

  # ── Custom flake8: double-quote style (Google Python Style) ────────────────
  find_package(ament_cmake_flake8 REQUIRED)
  ament_flake8(CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/flake8.ini")

  # ── Custom pep257: Google section-header style ─────────────────────────────
  # Ignore NumPy-style rules (D406,D407,D408,D409,D413) and prefer D212
  # (summary on first line) over D213 (summary on second line).
  find_package(ament_cmake_pep257 REQUIRED)
  ament_pep257("--add-ignore" "D213,D406,D407,D408,D409,D413")

  find_package(ament_cmake_pytest REQUIRED)
  # Pass the package source root so the test can import helpers directly.
  ament_add_pytest_test(test_oakd_detector
    test/test_oakd_detector.py
    TIMEOUT 60
    ENV "OAKD_DETECTION_SRC=${CMAKE_CURRENT_SOURCE_DIR}"
  )
endif()

ament_package()
