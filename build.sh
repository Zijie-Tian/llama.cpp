#! /bin/zsh

# cmake -B build -DGGML_CUDA=ON \
#     -DGGML_CUDA_FORCE_DMMV=ON \
#     -DGGML_CUDA_FORCE_CUBLAS=ON \
#     -DGGML_CUDA_F16=ON \
#     -DGGML_AVX=ON \
#     -DGGML_AVX2=ON \
#     -DGGML_AVX512=ON \
#     -DCMAKE_CUDA_ARCHITECTURES=87

cmake -B build -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87

cmake --build build --config Release --target llama-cli -j 12
