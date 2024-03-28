#pragma once
#include <vector>

template<typename T>
class GaussMethodSolver
{
private:
	std::vector<std::vector<T>> coefficient_matrix_;
	std::vector<T> result_matrix_;
	std::vector<T> constant_terms_;

private:
	void gaussian_elimination(std::vector<std::vector<T>>& matrix, std::vector<T>& b);
	void back_substitution(const std::vector<std::vector<T>>& matrix, const std::vector<T>& b);

public:
	GaussMethodSolver(std::vector<std::vector<T>>& coefficient_matrix, std::vector<T>& constant_terms)
		: coefficient_matrix_{ coefficient_matrix }, constant_terms_{ constant_terms }
	{}

public:
	std::vector<T> result_matrix() const { return result_matrix_; }
	GaussMethodSolver& solve();
};

// Приведение матрицы к верхнетреугольному виду методом Гаусса
template<typename T>
inline void GaussMethodSolver<T>::gaussian_elimination(std::vector<std::vector<T>>& matrix, std::vector<T>& b)
{
	int n = matrix.size();

	for (int i = 0; i < n; i++) {
		// Приведение текущей строки к единичному ведущему элементу
		T pivot = matrix[i][i];
		for (int j = i; j < n; j++) {
			matrix[i][j] /= pivot;
		}
		b[i] /= pivot;

		// Вычитание текущей строки из всех нижележащих строк
		for (int k = i + 1; k < n; k++) {
			T factor = matrix[k][i];
			for (int j = i; j < n; j++) {
				matrix[k][j] -= factor * matrix[i][j];
			}
			b[k] -= factor * b[i];
		}
	}
}

// Обратный ход метода Гаусса для нахождения решения
template<typename T>
inline void GaussMethodSolver<T>::back_substitution(const std::vector<std::vector<T>>& matrix, const std::vector<T>& b)
{
	int n = matrix.size();
	result_matrix_.resize(n);
	for (int i = n - 1; i >= 0; i--) {
		result_matrix_[i] = b[i];
		for (int j = i + 1; j < n; j++) {
			result_matrix_[i] -= matrix[i][j] * result_matrix_[j];
		}
	}
}

template<typename T>
inline GaussMethodSolver<T>& GaussMethodSolver<T>::solve()
{
	// TODO
	gaussian_elimination(coefficient_matrix_, constant_terms_);
	back_substitution(coefficient_matrix_, constant_terms_);
	return *this;
}
