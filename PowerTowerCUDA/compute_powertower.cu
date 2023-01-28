#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/complex.h>

#include <opencv2/opencv.hpp>

//Define a CUDA kernel function to compute the power tower fractal
__global__ void compute_powertower_kernel(int rows, int cols, float range_left, float range_right, float range_top, float range_bottom, 
                                          int max_iterations, int threshold, uchar* __restrict__ data)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= rows || x >= cols) { return; }

    float scaleX = cols / (range_right - range_left);
    float scaleY = rows / (range_bottom - range_top);

    float real = x / scaleX + range_left;
    float imag = y / scaleY + range_top;
    thrust::complex<float> c(real, imag);
    thrust::complex<float> z(c);

    for (int i = 0; i < max_iterations; ++i) {
        z = thrust::pow(c, z);
        if (thrust::abs(z) > threshold) {
            data[y * cols + x] = 255;
            return;
        }
    }
    data[y * cols + x] = 0;
}

cv::Mat compute_powertower(int rows, int cols, float range_left, float range_right, float range_top, float range_bottom,
                           int max_iterations, int threshold)
{
    // Allocate memory on the GPU
    uchar* data;
    cudaMalloc((void**)&data, (size_t)rows * cols * sizeof(uchar));

    // Execute the CUDA kernel on the GPU
    dim3 blockSize(32, 32);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
    compute_powertower_kernel <<< gridSize, blockSize >>> (rows, cols, range_left, range_right, range_top, range_bottom,
                                                            max_iterations, threshold, data);

    // Check for any errors during kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Copy the result from the GPU to the host
    cv::Mat powertower_set(rows, cols, CV_8UC1);
    cudaMemcpy(powertower_set.data, data, (size_t)rows * cols * sizeof(uchar), cudaMemcpyDeviceToHost);

    // Check for any errors during memory copy
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Free the GPU memory
    cudaFree(data);

    return powertower_set;
}