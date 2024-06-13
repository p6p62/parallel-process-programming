#pragma once

//#define MPI_ENABLE

#ifdef MPI_ENABLE
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#endif // MPI_ENABLE


constexpr int LAPLACE_KERNEL[3][3]
{
	{0, 1, 0},
	{1, -4, 1},
	{0, 1, 0}
};

/// <summary>
/// Дискретный двумерный оператор Лапласа. Реализован в
/// предположении, что у входного массива есть граница
/// в один элемент (обработка происходит с отступом в 1
/// элемент от краёв)
/// </summary>
class DiscreteLaplaceOperator
{
public:
	template <typename T_in, typename T_out, size_t rows, size_t cols>
	static void CalculateSequential(
		const T_in(&source_data)[rows][cols],
		T_out(&output_data)[rows][cols]);

	template <typename T_in, typename T_out, size_t rows, size_t cols>
	static void CalculateOpenMP(
		const T_in(&source_data)[rows][cols],
		T_out(&output_data)[rows][cols]);

	//template <typename T_in, typename T_out, size_t rows, size_t cols>
	static void CalculateMPI(
		/*const T_in(&source_data)[rows][cols],
		T_out(&output_data)[rows][cols],*/
		int argc, char* argv[]);
};

template <typename T_in, typename T_out, size_t rows, size_t cols>
inline void DiscreteLaplaceOperator::CalculateSequential(
	const T_in(&source_data)[rows][cols],
	T_out(&output_data)[rows][cols])
{
	for (size_t i = 1; i < rows - 1; i++)
		for (size_t j = 1; j < cols - 1; j++)
			output_data[i][j]
			= source_data[i - 1][j - 1] * LAPLACE_KERNEL[0][0]
			+ source_data[i - 1][j - 0] * LAPLACE_KERNEL[0][1]
			+ source_data[i - 1][j + 1] * LAPLACE_KERNEL[0][2]
			+ source_data[i - 0][j - 1] * LAPLACE_KERNEL[1][0]
			+ source_data[i - 0][j - 0] * LAPLACE_KERNEL[1][1]
			+ source_data[i - 0][j + 1] * LAPLACE_KERNEL[1][2]
			+ source_data[i + 1][j - 1] * LAPLACE_KERNEL[2][0]
			+ source_data[i + 1][j - 0] * LAPLACE_KERNEL[2][1]
			+ source_data[i + 1][j + 1] * LAPLACE_KERNEL[2][2];
}

template<typename T_in, typename T_out, size_t rows, size_t cols>
inline void DiscreteLaplaceOperator::CalculateOpenMP(
	const T_in(&source_data)[rows][cols],
	T_out(&output_data)[rows][cols])
{
#pragma omp parallel for num_threads(2)
	for (int i = 1; i < rows - 1; i++)
		for (size_t j = 1; j < cols - 1; j++)
			output_data[i][j]
			= source_data[i - 1][j - 1] * LAPLACE_KERNEL[0][0]
			+ source_data[i - 1][j - 0] * LAPLACE_KERNEL[0][1]
			+ source_data[i - 1][j + 1] * LAPLACE_KERNEL[0][2]
			+ source_data[i - 0][j - 1] * LAPLACE_KERNEL[1][0]
			+ source_data[i - 0][j - 0] * LAPLACE_KERNEL[1][1]
			+ source_data[i - 0][j + 1] * LAPLACE_KERNEL[1][2]
			+ source_data[i + 1][j - 1] * LAPLACE_KERNEL[2][0]
			+ source_data[i + 1][j - 0] * LAPLACE_KERNEL[2][1]
			+ source_data[i + 1][j + 1] * LAPLACE_KERNEL[2][2];
}

//template <typename T_in, typename T_out, size_t rows, size_t cols>
inline void DiscreteLaplaceOperator::CalculateMPI(
	/*const T_in(&source_data)[rows][cols],
	T_out(&output_data)[rows][cols],*/
	int argc, char* argv[])
{
	/*MPI_Init(&argc, &argv);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0)
	{
		char hello_str[]{ "Hello world!" };
		MPI_Send(hello_str, _countof(hello_str), MPI_CHAR, 1, 0, MPI_COMM_WORLD);
	}
	else if (rank == 1)
	{
		constexpr size_t STR_SIZE{ 15 };
		char hello_str[STR_SIZE];
		MPI_Recv(hello_str, _countof(hello_str), MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		printf("RECIEVED: %s", hello_str);
	}

	MPI_Finalize();*/
}
