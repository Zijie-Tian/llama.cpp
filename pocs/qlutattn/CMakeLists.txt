set(TARGET bench_quant_acc)
add_executable(${TARGET} bench_quant_acc.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

set(TARGET test_qlutattn_w4g128k)
add_executable(${TARGET} test_qlutattn_w4g128k.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)