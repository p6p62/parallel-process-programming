#pragma once

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
