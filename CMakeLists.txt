cmake_minimum_required(VERSION 3.16)

project(pipeann_gorgeous_hybrid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

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

target_include_directories(pipeann_original_core PUBLIC
  ${PIPEANN_SOURCE_DIR}/include
)

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
  src/integrations/disk_index_relayout.cpp
  src/integrations/gorgeous_original.cpp
  src/integrations/linux_aligned_file_reader.cpp
  src/integrations/pipeann_builder.cpp
  src/gorgeous_layout.cpp
  src/io/page_reader.cpp
  src/pipeline_search.cpp
  src/quant/approx_distance.cpp
  src/search/candidate_pool.cpp
  src/tools/build_pipeline.cpp
  src/tools/tool_cli.cpp
  src/search/search_session.cpp
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

add_executable(pipeann_gorgeous_hybrid
  src/main.cpp
)

target_link_libraries(pipeann_gorgeous_hybrid PRIVATE pipeann_gorgeous_core)

add_executable(pipeann_gorgeous_build
  src/build_main.cpp
)

target_link_libraries(pipeann_gorgeous_build PRIVATE pipeann_gorgeous_core)

add_executable(pipeann_gorgeous_search
  src/search_main.cpp
)

target_link_libraries(pipeann_gorgeous_search PRIVATE pipeann_gorgeous_core)

add_executable(pipeann_gorgeous_inspect
  src/inspect_main.cpp
)

target_link_libraries(pipeann_gorgeous_inspect PRIVATE pipeann_gorgeous_core)

add_executable(pipeann_gorgeous_bench
  src/bench_main.cpp
)

target_link_libraries(pipeann_gorgeous_bench PRIVATE pipeann_gorgeous_core)

add_executable(pipeann_gorgeous_ground_truth
  src/ground_truth_main.cpp
)

target_link_libraries(pipeann_gorgeous_ground_truth PRIVATE pipeann_gorgeous_core)

add_executable(pipeann_gorgeous_bench_merge
  src/bench_merge_main.cpp
)

target_link_libraries(pipeann_gorgeous_bench_merge PRIVATE pipeann_gorgeous_core)

add_executable(pipeann_gorgeous_bench_compare
  src/bench_compare_main.cpp
)

target_link_libraries(pipeann_gorgeous_bench_compare PRIVATE pipeann_gorgeous_core)

add_executable(pipeann_gorgeous_search_test
  tests/pipeline_search_test.cpp
)

target_link_libraries(pipeann_gorgeous_search_test PRIVATE pipeann_gorgeous_core)

add_executable(pipeann_gorgeous_build_test
  tests/build_pipeline_test.cpp
)

target_link_libraries(pipeann_gorgeous_build_test PRIVATE pipeann_gorgeous_core)

add_executable(pipeann_gorgeous_tool_cli_test
  tests/tool_cli_test.cpp
)

target_link_libraries(pipeann_gorgeous_tool_cli_test PRIVATE pipeann_gorgeous_core)

add_executable(pipeann_gorgeous_integration_test
  tests/integration_modules_test.cpp
)

target_link_libraries(pipeann_gorgeous_integration_test PRIVATE pipeann_gorgeous_core)

add_test(NAME pipeann_gorgeous_search_test COMMAND pipeann_gorgeous_search_test)
add_test(NAME pipeann_gorgeous_build_test COMMAND pipeann_gorgeous_build_test)
add_test(NAME pipeann_gorgeous_tool_cli_test COMMAND pipeann_gorgeous_tool_cli_test)
add_test(NAME pipeann_gorgeous_integration_test COMMAND pipeann_gorgeous_integration_test)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(pipeann_gorgeous_core PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(pipeann_gorgeous_hybrid PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(pipeann_gorgeous_build PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(pipeann_gorgeous_search PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(pipeann_gorgeous_inspect PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(pipeann_gorgeous_bench PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(pipeann_gorgeous_ground_truth PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(pipeann_gorgeous_bench_merge PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(pipeann_gorgeous_bench_compare PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(pipeann_gorgeous_search_test PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(pipeann_gorgeous_build_test PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(pipeann_gorgeous_tool_cli_test PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(pipeann_gorgeous_integration_test PRIVATE -Wall -Wextra -Wpedantic)
endif()
