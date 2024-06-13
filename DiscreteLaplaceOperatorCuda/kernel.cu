#include <stdio.h>
#include <memory>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

// ********************************
using data_type = float;

constexpr size_t ROWS{ 10000 };
constexpr size_t COLS{ 10000 };

__device__ constexpr int LAPLACE_KERNEL[3][3]
{
	{0, 1, 0},
	{1, -4, 1},
	{0, 1, 0}
};

constexpr int LAPLACE_KERNEL_SIZE{ 3 };
constexpr int HALF_KERNEL_SIZE{ LAPLACE_KERNEL_SIZE / 2 };

template <typename T>
void cuda_deleter(T* deleted_ptr)
{
	printf("%p released!\n", deleted_ptr);
	cudaFree(deleted_ptr);
}

template <typename T>
using device_smart_ptr = std::unique_ptr<T, decltype(&cuda_deleter<T>)>;

template <typename T>
device_smart_ptr<T> smart_device_malloc(size_t size)
{
	T* temp{};
	cudaMalloc(&temp, size * sizeof(T));
	return device_smart_ptr<T>{ temp, cuda_deleter<T> };
}

__global__ void laplace_operator_cuda_kernel(data_type* result, const data_type* source, int rows, int cols)
{
	unsigned int x{ blockIdx.x * blockDim.x + threadIdx.x };
	unsigned int y{ blockIdx.y * blockDim.y + threadIdx.y };
	if (x < HALF_KERNEL_SIZE || x >= rows - HALF_KERNEL_SIZE
		|| y < HALF_KERNEL_SIZE || y >= cols - HALF_KERNEL_SIZE)
		return;

	result[y * cols + x]
		= source[(y - 1) * cols + x - 1] * LAPLACE_KERNEL[0][0]
		+ source[(y - 1) * cols + x - 0] * LAPLACE_KERNEL[0][1]
		+ source[(y - 1) * cols + x + 1] * LAPLACE_KERNEL[0][2]
		+ source[(y - 0) * cols + x - 1] * LAPLACE_KERNEL[1][0]
		+ source[(y - 0) * cols + x - 0] * LAPLACE_KERNEL[1][1]
		+ source[(y - 0) * cols + x + 1] * LAPLACE_KERNEL[1][2]
		+ source[(y + 1) * cols + x - 1] * LAPLACE_KERNEL[2][0]
		+ source[(y + 1) * cols + x - 0] * LAPLACE_KERNEL[2][1]
		+ source[(y + 1) * cols + x + 1] * LAPLACE_KERNEL[2][2];
}

void fill_random_data(data_type(&data)[ROWS][COLS])
{
	constexpr int SEED{ 1350 };
	std::ranlux48_base random{ SEED };
	for (size_t i = 1; i < ROWS - 1; i++)
		for (size_t j = 1; j < COLS - 1; j++)
			data[i][j] = random();
}

cudaError_t laplace_operator_cuda(data_type* result, const data_type* source, unsigned int rows, unsigned int cols)
{
	const size_t elements_count{ (size_t)rows * cols };
	//cudaError_t device_set_result{ cudaSetDevice(0) };
	device_smart_ptr<data_type> dev_result{ smart_device_malloc<data_type>(elements_count) };
	device_smart_ptr<data_type> dev_source{ smart_device_malloc<data_type>(elements_count) };

	std::cout << "Start copy into GPU memory" << std::endl;
	cudaMemcpy(dev_source.get(), source, elements_count * sizeof(data_type), cudaMemcpyHostToDevice);
	std::cout << "Copied into GPU" << std::endl;

	std::cout << "Kernel started" << std::endl;
	const dim3 BLOCKS_COUNT{ cols / LAPLACE_KERNEL_SIZE + 1, rows / LAPLACE_KERNEL_SIZE + 1 };
	const dim3 THREADS_COUNT{ LAPLACE_KERNEL_SIZE, LAPLACE_KERNEL_SIZE };
	laplace_operator_cuda_kernel<<<BLOCKS_COUNT, THREADS_COUNT>>>(dev_result.get(), dev_source.get(), rows, cols);
	std::cout << "Kernel finished" << std::endl;

	cudaDeviceSynchronize();

	std::cout << "Start copy from GPU memory" << std::endl;
	cudaMemcpy(result, dev_result.get(), elements_count * sizeof(data_type), cudaMemcpyDeviceToHost);
	std::cout << "Copied from GPU" << std::endl;

	return cudaError_t(0);
}
// ********************************

int main()
{
	std::unique_ptr<data_type[]> input_ptr{ new data_type[ROWS * COLS] {} };
	std::unique_ptr<data_type[]> output_ptr{ new data_type[ROWS * COLS] {} };
	data_type(&input)[ROWS][COLS]{ reinterpret_cast<data_type(&)[ROWS][COLS]>(*input_ptr.get()) };
	data_type(&output)[ROWS][COLS]{ reinterpret_cast<data_type(&)[ROWS][COLS]>(*output_ptr.get()) };

	std::cout << "Initialize data" << std::endl;
	fill_random_data(input);
	std::cout << "Initialize finished" << std::endl;

	// расчёт и замеры
	std::cout << "Start calculation" << std::endl;
	std::chrono::steady_clock clock{};
	std::chrono::steady_clock::time_point start{ clock.now() };

	cudaError_t cudaStatus = laplace_operator_cuda(output_ptr.get(), input_ptr.get(), ROWS, COLS);

	std::chrono::steady_clock::time_point end{ clock.now() };
	std::cout << "Calculated in: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

	cudaStatus = cudaDeviceReset();
	std::cout << "Success!" << std::endl;
	return 0;
}
