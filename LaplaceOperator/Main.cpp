#include <iostream>
#include <random>
#include <chrono>
#include <memory>
#include "DiscreteLaplaceOperator.h"

using data_type = int;

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
	std::unique_ptr<int[]> input_ptr{ new int[ROWS * COLS] };
	std::unique_ptr<int[]> output_ptr{ new int[ROWS * COLS] };
	int(&input)[ROWS][COLS]{ reinterpret_cast<int(&)[ROWS][COLS]>(*input_ptr.get()) };
	int(&output)[ROWS][COLS]{ reinterpret_cast<int(&)[ROWS][COLS]>(*output_ptr.get()) };

	std::cout << "Initialize data" << std::endl;
	//fill_random_data(input);
	std::cout << "Initialize finished" << std::endl;


	// расчёт и замеры
	std::cout << "Start calculation" << std::endl;
	std::chrono::steady_clock clock{};
	std::chrono::steady_clock::time_point start{ clock.now() };

	DiscreteLaplaceOperator::CalculateSequential(input, output);

	std::chrono::steady_clock::time_point end{ clock.now() };
	std::cout << "Calculated in: "
		<< std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " s" << std::endl;
}
