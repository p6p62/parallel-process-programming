#include <iostream>
#include "GaussMethodSolver.h"

using data_type = double;

int main()
{
	std::vector<std::vector<data_type>> A{ {3, 2, -5}, {2, -1, 3}, {1, 2, -1} };
	std::vector<data_type> B{ -1, 13, 9 };
	std::vector<data_type> X_expected{ 3, 5, 4 };

	GaussMethodSolver<data_type> gauss_solver{ GaussMethodSolver<data_type>{A, B}.solve() };
	std::vector<data_type> result{ gauss_solver.result_matrix() };
}
