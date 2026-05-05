mkdir -p minitorch/cuda_kernels
nvcc -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC -std=c++17 -Xcompiler="-std=c++17" -ccbin g++-14
nvcc -o minitorch/cuda_kernels/softmax_kernel.so --shared src/softmax_kernel.cu -Xcompiler -fPIC -std=c++17 -Xcompiler="-std=c++17" -ccbin g++-14
nvcc -o minitorch/cuda_kernels/layernorm_kernel.so --shared src/layernorm_kernel.cu -Xcompiler -fPIC -std=c++17 -Xcompiler="-std=c++17" -ccbin g++-14
