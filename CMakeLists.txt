cmake_minimum_required(VERSION 3.27)

project(fastgs_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(FASTGS_BUILD_PYTHON "Build nanobind Python module" ON)
option(FASTGS_BUILD_TEST "Build C++ test executable" ON)
option(FASTGS_BUILD_METAL "Build metallib when MLX helpers are available" ON)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES)

set(MLX_ROOT "${PYTHON_SITE_PACKAGES}/mlx/share/cmake/MLX")
set(MLX_INCLUDE_DIR "${PYTHON_SITE_PACKAGES}/mlx/include")
find_package(MLX CONFIG REQUIRED HINTS "${MLX_ROOT}")

if(FASTGS_BUILD_PYTHON)
  set(nanobind_ROOT "${PYTHON_SITE_PACKAGES}/nanobind/cmake")
  find_package(nanobind CONFIG REQUIRED)
endif()

set(CPP_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/fastgs_core/dummy.cpp
)

set(METAL_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/fastgs_core/metal/dummy.metal
)

add_library(fastgs_core_lib ${CPP_SOURCES})
target_include_directories(fastgs_core_lib PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/fastgs_core/include
  ${MLX_INCLUDE_DIR}
)
target_link_libraries(fastgs_core_lib PUBLIC mlx)

if(APPLE AND FASTGS_BUILD_METAL)
  if(COMMAND mlx_build_metallib)
    mlx_build_metallib(
      TARGET fastgs_core_metallib
      TITLE fastgs_core
      SOURCES ${METAL_SOURCES}
      INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/fastgs_core
        ${MLX_INCLUDE_DIR}
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

if(FASTGS_BUILD_PYTHON)
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
    ${MLX_INCLUDE_DIR}
  )
  target_link_libraries(_fastgs_core PUBLIC fastgs_core_lib mlx)

  if(BUILD_SHARED_LIBS)
    target_link_options(_fastgs_core PUBLIC -Wl,-rpath,@loader_path)
  endif()

  add_custom_command(TARGET _fastgs_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
      "$<TARGET_FILE:fastgs_core_lib>"
      "$<TARGET_FILE_DIR:_fastgs_core>/$<TARGET_FILE_NAME:fastgs_core_lib>"
  )

  if(TARGET fastgs_core_metallib)
    add_custom_command(TARGET _fastgs_core POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_BINARY_DIR}/fastgs_core.metallib"
        "$<TARGET_FILE_DIR:_fastgs_core>/fastgs_core.metallib"
    )
  endif()
endif()

if(FASTGS_BUILD_TEST)
  add_executable(fastgs_core_dummy_test
    ${CMAKE_CURRENT_LIST_DIR}/fastgs_core/test/test.cpp
  )
  target_include_directories(fastgs_core_dummy_test PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/fastgs_core/include
  )
  target_link_libraries(fastgs_core_dummy_test PRIVATE fastgs_core_lib)
endif()
