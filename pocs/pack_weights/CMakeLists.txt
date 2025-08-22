# Pack weights unit tests and benchmarks

# Add include directories for all test targets
include_directories(
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
    ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cpu
    ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cpu/qlutattn
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/include
)

# Test pack weights basic functionality
set(TARGET test_pack_weights_basic)
add_executable(${TARGET} test_pack_weights_basic.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

# Test NEON optimizations
set(TARGET test_pack_weights_neon)
add_executable(${TARGET} test_pack_weights_neon.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

# Performance comparison test
set(TARGET test_pack_weights_perf)
add_executable(${TARGET} test_pack_weights_perf.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

# TODO: Batch processing tests
# Batch processing tests are temporarily disabled pending integration with
# llama.cpp's thread pool. When re-enabled, tests should cover:
# - Memory pool allocation and reuse
# - Multi-threaded batch processing with proper synchronization
# - Performance comparison vs individual processing
# - Edge cases with varying tensor sizes and quantization types