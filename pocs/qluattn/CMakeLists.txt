set(TARGET bench_quant_acc)
add_executable(${TARGET} bench_quant_acc.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)