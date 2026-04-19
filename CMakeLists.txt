cmake_minimum_required(VERSION 3.27)

project(fastgs_core LANGUAGES CXX)

# ----------------------------- Setup -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(FASTGS_BUILD_PYTHON "Build nanobind Python module" ON)
option(FASTGS_BUILD_TEST "Build C++ test executable" ON)
option(FASTGS_BUILD_METAL "Build metallib when MLX helpers are available" ON)
option(FASTGS_USE_MLX_MODULE_CMAKE_DIR "Resolve MLX root via `python -m mlx --cmake-dir`" OFF)
option(FASTGS_XCODE_DEV_MODE "Enable Xcode Metal source visibility" OFF)

# ----------------------------- Dependencies -----------------------------
find_package(
  Python 3.11
  COMPONENTS Interpreter Development.Module
  REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES)

# Prefer robust pip/conda-installed MLX CMake location by default.
set(MLX_ROOT "${PYTHON_SITE_PACKAGES}/mlx/share/cmake/MLX")
if(FASTGS_USE_MLX_MODULE_CMAKE_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
    RESULT_VARIABLE MLX_CMAKE_DIR_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE MLX_ROOT_FROM_MODULE)

  if(MLX_CMAKE_DIR_RESULT EQUAL 0 AND NOT MLX_ROOT_FROM_MODULE STREQUAL "")
    set(MLX_ROOT "${MLX_ROOT_FROM_MODULE}")
  endif()
endif()

# MLX headers bundled in Python package.
set(MLX_PYTHON_INCLUDE_DIR "${PYTHON_SITE_PACKAGES}/mlx/include")

find_package(MLX CONFIG REQUIRED HINTS "${MLX_ROOT}")

if(FASTGS_BUILD_PYTHON)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)
endif()

# ----------------------------- Sources -----------------------------
file(GLOB FASTGS_CPP_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/fastgs_core/*.cpp"
)

file(GLOB FASTGS_METAL_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/fastgs_core/metal/*.metal"
)

# ----------------------------- Core Library -----------------------------
add_library(fastgs_core_lib ${FASTGS_CPP_SOURCES})

target_include_directories(fastgs_core_lib PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/fastgs_core/include
  ${MLX_PYTHON_INCLUDE_DIR}
)

target_link_libraries(fastgs_core_lib PUBLIC mlx)

if(FASTGS_XCODE_DEV_MODE AND CMAKE_GENERATOR STREQUAL "Xcode")
  target_sources(fastgs_core_lib PRIVATE ${FASTGS_METAL_SOURCES})
  set_source_files_properties(${FASTGS_METAL_SOURCES} PROPERTIES LANGUAGE METAL)
endif()

# ----------------------------- Metal -----------------------------
if(APPLE AND FASTGS_BUILD_METAL)
  if(COMMAND mlx_build_metallib)
    mlx_build_metallib(
      TARGET fastgs_core_metallib
      TITLE fastgs_core
      SOURCES ${FASTGS_METAL_SOURCES}
      INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/fastgs_core
        ${MLX_PYTHON_INCLUDE_DIR}
      OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    add_dependencies(fastgs_core_lib fastgs_core_metallib)

    add_custom_command(TARGET fastgs_core_metallib POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_BINARY_DIR}/fastgs_core.metallib"
        "$<TARGET_FILE_DIR:fastgs_core_lib>/fastgs_core.metallib"
    )
  else()
    message(WARNING "mlx_build_metallib not available. Skip metallib build.")
  endif()
endif()

# ----------------------------- Python Bindings -----------------------------
if(FASTGS_BUILD_PYTHON)
  set(FASTGS_PY_PACKAGE_DIR "${CMAKE_CURRENT_LIST_DIR}/python_package/fastgs_core")

  nanobind_add_module(
    _fastgs_core
    NB_STATIC
    STABLE_ABI
    LTO
    NOMINSIZE
    NB_DOMAIN mlx
    ${CMAKE_CURRENT_LIST_DIR}/fastgs_core/binding/binding.cpp
  )

  target_include_directories(_fastgs_core PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/fastgs_core/include
    ${MLX_PYTHON_INCLUDE_DIR}
  )

  target_link_libraries(_fastgs_core PRIVATE fastgs_core_lib)

  if(BUILD_SHARED_LIBS)
    target_link_options(_fastgs_core PRIVATE -Wl,-rpath,@loader_path)
  endif()

  add_custom_command(TARGET _fastgs_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
      "$<TARGET_FILE:fastgs_core_lib>"
      "$<TARGET_FILE_DIR:_fastgs_core>/$<TARGET_FILE_NAME:fastgs_core_lib>"
  )

  # Local dev convenience: mirror extension artifacts into python package dir.
  add_custom_command(TARGET _fastgs_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
      "${FASTGS_PY_PACKAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy
      "$<TARGET_FILE:_fastgs_core>"
      "${FASTGS_PY_PACKAGE_DIR}/$<TARGET_FILE_NAME:_fastgs_core>"
    COMMAND ${CMAKE_COMMAND} -E copy
      "$<TARGET_FILE:fastgs_core_lib>"
      "${FASTGS_PY_PACKAGE_DIR}/$<TARGET_FILE_NAME:fastgs_core_lib>"
  )

  if(TARGET fastgs_core_metallib)
    add_custom_command(TARGET _fastgs_core POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_BINARY_DIR}/fastgs_core.metallib"
        "$<TARGET_FILE_DIR:_fastgs_core>/fastgs_core.metallib"
    )
    add_custom_command(TARGET _fastgs_core POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_BINARY_DIR}/fastgs_core.metallib"
        "${FASTGS_PY_PACKAGE_DIR}/fastgs_core.metallib"
    )
  endif()
endif()

# ----------------------------- Tests -----------------------------
if(FASTGS_BUILD_TEST)
  add_executable(fastgs_core_dummy_test
    ${CMAKE_CURRENT_LIST_DIR}/fastgs_core/test/test.cpp
  )
  target_include_directories(fastgs_core_dummy_test PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/fastgs_core/include
  )
  target_link_libraries(fastgs_core_dummy_test PRIVATE fastgs_core_lib)
endif()
