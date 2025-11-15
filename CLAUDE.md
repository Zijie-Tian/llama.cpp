- 编译命令
CFLAGS="-march=armv8.7a" CXXFLAGS="-march=armv8.7a" cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -G Ninja -D GGML_CUDA=OFF-B build-arm64
 cmake --build build-arm64 --config Release -j12