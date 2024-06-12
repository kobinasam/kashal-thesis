
#include <iomanip>

#include <armadillo>
#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "matrix_utility.h"

using namespace std;
using namespace arma;

#define _USE_MATH_DEFINES

typedef std::vector<double> stdvec;
typedef std::vector< std::vector<double> > stdvecvec;
typedef Eigen::MatrixXd eigenmat;

// ==================================================================
void util::print_mat(const mat m, const string& name, bool hide_contents, double precision)
{
    std::cout << "============================================" << endl;
    std::cout << "Name: " << name << ", Rows: " << m.n_rows << ", Cols: " << m.n_cols << endl;
    if (!hide_contents)
    {
        for (uword i = 0; i < m.n_rows; i++)
        {
            for (uword j = 0; j < m.n_cols; j++)
            {
                std::cout << setprecision(precision) << m.at(i, j) << " ";
            }
            std::cout << endl;
        }
    }
    if (m.is_empty())
    {
        std::cout << name << " is empty!";
    }
    std::cout << endl;
}

// ==================================================================
double util::same_within_precision(const mat& m1, const mat& m2)
{
    double precision = 1e-16;
    while (!approx_equal(m1, m2, "both", precision, precision) and precision < 1)
    {
        precision = precision * 10;
    }
    return precision;
}

// ==================================================================
// Floating point math sucks. Use this to guarantee you can round up
// or down on a floating point value
float util::round_to_decimal(float var, int dec)
{
    int multiplier = pow(10, dec);
    int value = (int)(var * multiplier + .5);
    return (float)value / multiplier;
}

// ==================================================================
void util::load_matrix_from_csv(mat& M, string filename)
{
    /* cout << "Loading this file: " << filename << endl; */
    ifstream file_handle;
    file_handle.open(filename);

    if (file_handle.fail())
    {
        throw std::runtime_error("Failed to open " + filename + " due to reason: "); // +strerror(errno));
    }
    M.load(file_handle, arma::csv_ascii);
    file_handle.close();
}

// ==================================================================
void util::save_matrix_to_csv(const mat& M, string filename)
{
    ofstream file_handle;
    file_handle.open(filename);
    if (!file_handle.good())
    {
        throw std::runtime_error("Problem writing to file: " + filename);
    }
    M.save(file_handle, arma::csv_ascii);
    file_handle.close();
}

// ==================================================================
stdvecvec util::mat_to_std_vec(const mat& M)
{
    if (M.n_elem == 0 || M.n_rows == 0 || M.n_cols == 0)
    {
        throw ("Called mat_to_std_vec with incorrect shape");
    }

    stdvecvec V(M.n_rows);
    for (size_t i = 0; i < M.n_rows; ++i) {
        V[i] = arma::conv_to<stdvec>::from(M.row(i));
    };
    return V;
}

// ==================================================================
void util::mat_to_std_array(const mat* M, double* arr)
{
    // NOTE: Requires arr to be allocated correctly same rows / cols as M

    // FIXME: Is there a better way to convert from armadillo objects to arrays?
    // Looks like armadillo only provides methods to convert to vectors.
    // Assuming that conversion is efficient, probably best to convert arma::Mat -> vector -> array?
    stdvecvec V = util::mat_to_std_vec(*M);

    int numrows = V.size();
    int numcols = V[0].size();
    for (int i = 0; i < numrows; i++)
    {
        for (int j = 0; j < numcols; j++)
        {
            arr[i * numcols + j] = V[i][j];
        }
    }
}

// ==================================================================
void util::mats_to_std_array(vector<mat*> Ms, double* arr)
{
    // Combines all matrices in Ms into a single flattened array
    for (int i = 0; i < Ms.size(); i++)
    {
        util::mat_to_std_array(Ms[i], arr);
        // Use pointer arithmetic to increment pointer to fill the right spot in the buffer
        arr += Ms[i]->n_elem;
    }
}

// ==================================================================
void util::std_array_to_mat(double* arr, mat& M)
{
    M = mat(arr, M.n_cols, M.n_rows).t();
}

// ==================================================================
void util::std_array_to_mats(double* arr, vector<mat*> Ms)
{
    // Fill matrices with values from array
    int arr_idx = 0;
    for (int i = 0; i < Ms.size(); i++)
    {
        util::std_array_to_mat(arr, *(Ms[i]));
        arr = arr + Ms[i]->n_elem;
    }
}

// ============================================================================
// FIXME: Armadillo chol() was not working, so I wrote this using Eigen and it gets same results as Matlab
bool util::cholesky_eigen(mat& L, mat& R, mat& h_matrix)
{
    eigenmat A(h_matrix.n_rows, h_matrix.n_cols);

    // FIXME: Probably inefficient, but do this for now
    for (int r = 0; r < h_matrix.n_rows; r++)
    {
        for (int c = 0; c < h_matrix.n_cols; c++)
        {
            A(r, c) = h_matrix.at(r, c);
        }
    }

    Eigen::LLT<eigenmat> lltofA(A);
    eigenmat L_eigen = lltofA.matrixL();
    eigenmat R_eigen = lltofA.matrixU();

    if (lltofA.info() == Eigen::Success)
    {
        // FIXME: Probably inefficient, but do this for now
        L.set_size(h_matrix.n_rows, h_matrix.n_cols);
        for (size_t i = 0, nRows = L_eigen.rows(), nCols = L_eigen.cols(); i < nCols; ++i)
        {
            for (size_t j = 0; j < nRows; ++j)
            {
                L.at(i, j) = L_eigen(i, j);
            }
        }

        R.set_size(h_matrix.n_rows, h_matrix.n_cols);
        for (size_t i = 0, nRows = R_eigen.rows(), nCols = R_eigen.cols(); i < nCols; ++i)
        {
            for (size_t j = 0; j < nRows; ++j)
            {
                R.at(i, j) = R_eigen(i, j);
            }
        }
        return true;
    }
    return false;
}

// ==================================================================
int util::count_in_matrices(vector<mat*> matrices)
{
    int total_count = 0;
    for (int i = 0; i < matrices.size(); i++)
    {
        int count = matrices[i]->n_elem;
        total_count += count;
    }
    return total_count;
}
