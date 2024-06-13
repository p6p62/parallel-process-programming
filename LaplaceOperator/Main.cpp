#include <iostream>
#include <random>
#include <chrono>
#include <memory>
#include <omp.h>
#include "DiscreteLaplaceOperator.h"

using data_type = float;

constexpr size_t ROWS{ 100000 };
constexpr size_t COLS{ 10000 };

void fill_random_data(data_type(&data)[ROWS][COLS])
{
	constexpr int SEED{ 1350 };
	std::ranlux48_base random{ SEED };
	for (size_t i = 1; i < ROWS - 1; i++)
		for (size_t j = 1; j < COLS - 1; j++)
			data[i][j] = random();
}

int main()
{
	// инициализация буферов
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

	double t1{ omp_get_wtime() };
	DiscreteLaplaceOperator::CalculateSequential(input, output);
	//DiscreteLaplaceOperator::CalculateOpenMP(input, output);
	double t2{ omp_get_wtime() };

	std::chrono::steady_clock::time_point end{ clock.now() };
	std::cout << "Calculated in: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

	std::cout << "wtime: " << t2 - t1 << std::endl;

	// для отмены отброса вызова функции при компиляции в режиме release
	/*long long t{ 0 };
	for (size_t i = 0; i < ROWS; i++)
		for (size_t j = 0; j < COLS; j++)
			t += output[i][j];
	std::cout << "sum: " << t << std::endl;*/
}
