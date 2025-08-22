# dependencies

find_package(Threads REQUIRED)

# third-party

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (EMSCRIPTEN)
else()
    if (NOT GGML_BACKEND_DL)
        add_subdirectory(vdot)
        add_subdirectory(transpose)
        add_subdirectory(qlutattn)
        add_subdirectory(pack_weights)
    endif()
endif()
