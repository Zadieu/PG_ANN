cmake_minimum_required(VERSION 3.16)

project(pipeann_gorgeous LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_path(LIBAIO_INCLUDE_DIR libaio.h)
find_library(LIBAIO_LIBRARY aio)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

set(PIPEANN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pipeann")
if(NOT EXISTS "${PIPEANN_SOURCE_DIR}/include/index.h")
  message(FATAL_ERROR "Vendored PipeANN source tree not found at ${PIPEANN_SOURCE_DIR}")
endif()

add_library(pipeann_original_core
  ${PIPEANN_SOURCE_DIR}/src/index.cpp
  ${PIPEANN_SOURCE_DIR}/src/ssd_index.cpp
  ${PIPEANN_SOURCE_DIR}/src/update/direct_insert.cpp
  ${PIPEANN_SOURCE_DIR}/src/update/delete_merge.cpp
  ${PIPEANN_SOURCE_DIR}/src/utils/index_build_utils.cpp
  ${PIPEANN_SOURCE_DIR}/src/utils/pipnn.cpp
  ${PIPEANN_SOURCE_DIR}/src/utils/linux_aligned_file_reader.cpp
  ${PIPEANN_SOURCE_DIR}/src/utils/partition.cpp
  ${PIPEANN_SOURCE_DIR}/src/utils/kmeans_utils.cpp
  ${PIPEANN_SOURCE_DIR}/src/utils/distance.cpp
  ${PIPEANN_SOURCE_DIR}/src/search/coro_search.cpp
  ${PIPEANN_SOURCE_DIR}/src/search/beam_search.cpp
  ${PIPEANN_SOURCE_DIR}/src/search/page_search.cpp
  ${PIPEANN_SOURCE_DIR}/src/search/pipe_search.cpp
  ${PIPEANN_SOURCE_DIR}/src/search/spec_filter_search.cpp
)

target_include_directories(pipeann_original_core PUBLIC ${PIPEANN_SOURCE_DIR}/include)
target_compile_definitions(pipeann_original_core PUBLIC DNDEBUG BG_IO_THREAD)
target_compile_options(pipeann_original_core PRIVATE
  -fopenmp-simd
  -Wno-class-memaccess
  -Wno-unused-variable
  -Wno-unused-parameter
  -Wno-unused-but-set-variable
)
target_link_libraries(pipeann_original_core PUBLIC ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX)

add_library(pipeann_gorgeous_core
  src/integrations/linux_aligned_file_reader.cpp
  src/integrations/pipeann_layout_activate.cpp
  src/integrations/pipeann_parity_bench.cpp
  src/integrations/pipeann_parity_index.cpp
  src/integrations/pipeann_parity_searcher.cpp
  src/tools/vector_io.cpp
  src/tools/tool_cli.cpp
)

target_include_directories(pipeann_gorgeous_core PUBLIC
  include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gorgeous/graph_partition/include
)
target_link_libraries(pipeann_gorgeous_core PUBLIC pipeann_original_core)

if(LIBAIO_INCLUDE_DIR AND LIBAIO_LIBRARY)
  target_include_directories(pipeann_gorgeous_core PRIVATE ${LIBAIO_INCLUDE_DIR})
  target_link_libraries(pipeann_gorgeous_core PRIVATE ${LIBAIO_LIBRARY})
  target_include_directories(pipeann_original_core PRIVATE ${LIBAIO_INCLUDE_DIR})
  target_link_libraries(pipeann_original_core PUBLIC ${LIBAIO_LIBRARY})
endif()

macro(add_pipeann_executable target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE pipeann_gorgeous_core)
endmacro()

add_pipeann_executable(pipeann_gorgeous_bench src/bench_main.cpp)
add_pipeann_executable(pipeann_gorgeous_bench_pipeann src/pipeann_parity_main.cpp)
add_pipeann_executable(pipeann_gorgeous_bench_merge src/bench_merge_main.cpp)
add_pipeann_executable(pipeann_gorgeous_bench_compare src/bench_compare_main.cpp)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(pipeann_gorgeous_core PRIVATE -Wall -Wextra -Wpedantic)
endif()
