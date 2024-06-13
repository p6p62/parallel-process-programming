#include <iostream>
#include <random>
#include <chrono>
#include <memory>
#include <omp.h>
#include "DiscreteLaplaceOperator.h"

constexpr auto CHECK_SUM_PRINT = 0;

using data_type = unsigned int;

constexpr size_t ROWS{ 10000 + 2 };
constexpr size_t COLS{ 10000 + 2 };

void fill_random_data(data_type(&data)[ROWS][COLS])
{
	constexpr int SEED{ 1350 };
	std::mt19937 random{ SEED };
	for (size_t i{ 1 }; i < ROWS - 1; i++)
		for (size_t j{ 1 }; j < COLS - 1; j++)
			data[i][j] = random();
}

void sequential_and_openmp()
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

#ifdef MPI_ENABLE
void mpi(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int process_count;
	MPI_Comm_size(MPI_COMM_WORLD, &process_count);
	int rows_per_process_count{ (int)ROWS / process_count };
	int last_process_rows_count{ ROWS % process_count == 0
		? rows_per_process_count
		: (int)ROWS % process_count + 1 };

	if (rank == 0)
	{
		// инициализация буферов
		std::unique_ptr<data_type[]> input_ptr{ new data_type[ROWS * COLS] {} };
		std::unique_ptr<data_type[]> output_ptr{ new data_type[ROWS * COLS] {} };
		data_type(&input)[ROWS][COLS]{ reinterpret_cast<data_type(&)[ROWS][COLS]>(*input_ptr.get()) };
		data_type(&output)[ROWS][COLS]{ reinterpret_cast<data_type(&)[ROWS][COLS]>(*output_ptr.get()) };

		std::cout << "Initialize data" << std::endl;
		fill_random_data(input);
		std::cout << "Initialize finished" << std::endl;

		int offset{ rows_per_process_count * (int)COLS };

		if constexpr (CHECK_SUM_PRINT)
		{
			for (int p{ 1 }; p < process_count - 1; p++)
			{
				float s{};
				for (int i{ p * rows_per_process_count }; i < (p + 1) * rows_per_process_count - 1; i++)
					for (int j{ 1 }; j < (int)COLS - 1; j++)
						s += input[i][j];
				printf_s("p%d, sum: %f\n", p, s);
			}
			float s{};
			for (int i{ process_count * rows_per_process_count }; i < ROWS - 1; i++)
				for (int j{ 1 }; j < (int)COLS - 1; j++)
					s += input[i][j];
			printf_s("p%d, sum: %f\n", process_count - 1, s);
		}

		// замеры
		std::cout << "Start calculation" << std::endl;
		std::chrono::steady_clock clock{};
		std::chrono::steady_clock::time_point start{ clock.now() };

		for (int i{ 1 }; i < process_count - 1; i++)
			MPI_Send(input_ptr.get() + i * offset - COLS, offset + 2 * COLS, MPI_INT, i, 0, MPI_COMM_WORLD);

		MPI_Send(input_ptr.get() + (process_count - 1) * offset - COLS, last_process_rows_count * COLS + COLS, MPI_INT, process_count - 1, 0, MPI_COMM_WORLD);

		for (int i{ 1 }; i < rows_per_process_count; i++)
			for (int j{ 1 }; j < COLS - 1; j++)
				output_ptr[i * COLS + j]
				= input_ptr[((size_t)i - 1) * (int)COLS + j - 1] * LAPLACE_KERNEL[0][0]
				+ input_ptr[((size_t)i - 1) * (int)COLS + j - 0] * LAPLACE_KERNEL[0][1]
				+ input_ptr[((size_t)i - 1) * (int)COLS + j + 1] * LAPLACE_KERNEL[0][2]
				+ input_ptr[((size_t)i - 0) * (int)COLS + j - 1] * LAPLACE_KERNEL[1][0]
				+ input_ptr[((size_t)i - 0) * (int)COLS + j - 0] * LAPLACE_KERNEL[1][1]
				+ input_ptr[((size_t)i - 0) * (int)COLS + j + 1] * LAPLACE_KERNEL[1][2]
				+ input_ptr[((size_t)i + 1) * (int)COLS + j - 1] * LAPLACE_KERNEL[2][0]
				+ input_ptr[((size_t)i + 1) * (int)COLS + j - 0] * LAPLACE_KERNEL[2][1]
				+ input_ptr[((size_t)i + 1) * (int)COLS + j + 1] * LAPLACE_KERNEL[2][2];

		if constexpr (CHECK_SUM_PRINT)
		{
			std::cout << process_count << std::endl;
			std::cout << rows_per_process_count << std::endl;
			std::cout << last_process_rows_count << std::endl;
		}

		for (int i{ 1 }; i < process_count - 1; i++)
			MPI_Recv(&output_ptr[(size_t)i * offset], offset, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		MPI_Recv(&output_ptr[(size_t)(process_count - 1) * offset], offset, MPI_INT, process_count - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		std::chrono::steady_clock::time_point end{ clock.now() };
		std::cout << "Calculated in: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

		std::cout << "Done!" << std::endl;
	}
	else
	{
		if (rank == process_count - 1)
			rows_per_process_count = last_process_rows_count;

		int buffer_size{ rows_per_process_count * (int)COLS + 2 * (int)COLS };

		std::unique_ptr<data_type[]> input_ptr{ new data_type[buffer_size] {} };
		std::unique_ptr<data_type[]> output_ptr{ new data_type[buffer_size] {} };
		MPI_Recv(input_ptr.get(), buffer_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

		if constexpr (CHECK_SUM_PRINT)
		{
			float sum{ 0 };
			for (int i{ 1 }; i < rows_per_process_count; i++)
				for (int j{ 1 }; j < COLS - 1; j++)
					sum += input_ptr[i * COLS + j];
			printf_s("P_ID: %d, sum: %f\n", rank, sum);
		}

		// расчёт
		for (int i{ 1 }; i < rows_per_process_count; i++)
			for (int j{ 1 }; j < COLS - 1; j++)
				output_ptr[i * COLS + j]
				= input_ptr[((size_t)i - 1) * (int)COLS + j - 1] * LAPLACE_KERNEL[0][0]
				+ input_ptr[((size_t)i - 1) * (int)COLS + j - 0] * LAPLACE_KERNEL[0][1]
				+ input_ptr[((size_t)i - 1) * (int)COLS + j + 1] * LAPLACE_KERNEL[0][2]
				+ input_ptr[((size_t)i - 0) * (int)COLS + j - 1] * LAPLACE_KERNEL[1][0]
				+ input_ptr[((size_t)i - 0) * (int)COLS + j - 0] * LAPLACE_KERNEL[1][1]
				+ input_ptr[((size_t)i - 0) * (int)COLS + j + 1] * LAPLACE_KERNEL[1][2]
				+ input_ptr[((size_t)i + 1) * (int)COLS + j - 1] * LAPLACE_KERNEL[2][0]
				+ input_ptr[((size_t)i + 1) * (int)COLS + j - 0] * LAPLACE_KERNEL[2][1]
				+ input_ptr[((size_t)i + 1) * (int)COLS + j + 1] * LAPLACE_KERNEL[2][2];

		MPI_Send(output_ptr.get() + COLS, rows_per_process_count * COLS, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();
}
#endif // MPI_ENABLE

int main(int argc, char* argv[])
{
#ifndef MPI_ENABLE
	sequential_and_openmp();
#endif // !MPI_ENABLE

#ifdef MPI_ENABLE
	mpi(argc, argv);
#endif // MPI_ENABLE
}
