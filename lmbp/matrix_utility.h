#pragma once

#include <armadillo>

typedef std::vector<double> stdvec;
typedef std::vector< std::vector<double> > stdvecvec;

using namespace std;
using namespace arma;

namespace util
{
	void print_mat(const mat m, const string& name, bool hide_contents = false, double precision = 16);
	double same_within_precision(const mat& m1, const mat& m2);
	float round_to_decimal(float var, int dec);
	void load_matrix_from_csv(mat& M, string filename);
	void save_matrix_to_csv(const mat& M, string filename);
	stdvecvec mat_to_std_vec(const mat& M);
	void mat_to_std_array(const mat* M, double* arr);
	void mats_to_std_array(vector<mat*> Ms, double* arr);
	void std_array_to_mat(double* arr, mat& M);
	void std_array_to_mats(double* arr, vector<mat*> Ms);
	bool cholesky_eigen(mat& L, mat& R, mat& h_matrix);
	int count_in_matrices(vector<mat*> matrices);
};
