set(TARGET llama-transpose)
add_executable(${TARGET} transpose128.c)

# Find OpenMP for this target only
find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} OpenMP::OpenMP_C)
else()
    target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
    message(STATUS "OpenMP not found for ${TARGET}, building without OpenMP support")
endif()

target_compile_features(${TARGET} PRIVATE c_std_11) 