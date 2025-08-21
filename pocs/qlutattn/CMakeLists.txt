set(TARGET bench_quant_acc)
add_executable(${TARGET} bench_quant_acc.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

set(TARGET test_qlutattn_w4g128k)
add_executable(${TARGET} test_qlutattn_w4g128k.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

set(TARGET test_qlutattn_128x128)
add_executable(${TARGET} test_qlutattn_128x128.cpp)
target_link_libraries(
    ${TARGET} PRIVATE
    common llama
    ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cpu/qlutattn # This line is removed
    ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cpu # This line is removed
    ${CMAKE_SOURCE_DIR}/ggml/src # This line is removed
)
target_compile_features(${TARGET} PRIVATE cxx_std_11)

set(TARGET test_channel_quant)
add_executable(${TARGET} test_channel_quant.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

set(TARGET test_flashattn_identical)
add_executable(${TARGET} test_flashattn_identical.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

set(TARGET test-tmac-quant)
add_executable(${TARGET} test_tmac_quant.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

set(TARGET align_attn)
add_executable(${TARGET} align_attn.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

set(TARGET test_tmac_simple)
add_executable(${TARGET} test_tmac_simple.cpp)
target_include_directories(
    ${TARGET} PRIVATE  # This line is removed
    ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cpu/tmac # This line is removed
    ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cpu # This line is removed
    ${CMAKE_SOURCE_DIR}/ggml/src # This line is removed
)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

set(TARGET bench_qlutattn_gemv)
add_executable(${TARGET} bench_qlutattn_gemv.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

set(TARGET bench_qlutattn_vs_q40)
add_executable(${TARGET} bench_qlutattn_vs_q40.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
