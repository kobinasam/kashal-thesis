// built-ins
#include<vector>

// third-party libraries here
#include <mpi.h>
#include <armadillo>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <random>
#include <tuple>

// my modules here
#include "RNN.h"
#include "matrix_utility.h"

// Definitions
#define _USE_MATH_DEFINES

#undef ARMA_OPENMP_THREADS
#define ARMA_OPENMP_THREADS 50 // Max threads used by armadillo for whatever logic they have parallelized. Cool
#define ARMA_NO_DEBUG          // Only define this if we want to disable debugging for production code

// Used to distingish message types in sending/receiving messages
#define MPI_J_TOTAL 0
#define MPI_NEW_JOB 1
#define MPI_JOB_FINISHED 2
#define MPI_LOOP_DONE 3

#define VERBOSE_LOG if (verbose && _rank == 0)

using namespace std;
using namespace arma;

RNN::RNN(
    int rank,
    vector<int> workerids,
    int active_workers,
    int num_samples,
    const mat idq_start_positions,
    const bool add_dropout,
    const double dropout_max_extra_cost,
    const int max_drop,
    const bool use_shortcuts,
    const bool use_idq,
    const double vd,
    const double t_final,
    const double ts,
    const double vdc,
    const colvec idq_ref_centre,
    const double gain1,
    const double gain2,
    const double gain3,
    const double cost_term_power,
    const double starting_mu,
    const double mu_dec,
    const double mu_inc,
    const double mu_max,
    const double mu_min,
    const mat a,
    const mat b,
    const mat starting_w1,
    const mat starting_w2,
    const mat starting_w3
    )
{
    isPrintable = false;
    static bool seeded = false;
    if (!seeded) {
        arma::arma_rng::set_seed_random();
        util::save_matrix_to_csv(starting_w1, "w1_initial_weight.csv");
        util::save_matrix_to_csv(starting_w2, "w2_initial_weight.csv");
        util::save_matrix_to_csv(starting_w3, "w3_initial_weight.csv");
        // cout << "w1" << starting_w1 << endl;
        // cout << "w2" << starting_w2 << endl;
        // cout << "w3" << starting_w3 << endl;
        seeded = true;
    }
    _rank = rank;
    _workerids = workerids;
    _active_workers = active_workers;
    _num_samples = num_samples;
    _use_shortcuts = use_shortcuts;
    _use_idq = use_idq;
    _vd = vd;
    _t_final = t_final;
    _ts = ts;
    _trajectory_length = t_final / ts;
    _vdc = vdc;
    _idq_ref_centre = idq_ref_centre;
    _gain1 = gain1;
    _gain2 = gain2;
    _gain3 = gain3;
    _cost_term_power = cost_term_power;
    _starting_mu = starting_mu;
    _mu_dec = mu_dec;
    _mu_inc = mu_inc;
    _mu_max = mu_max;
    _mu_min = mu_min;
    _dropout_max_extra_cost = dropout_max_extra_cost;
    _idq_start_positions = idq_start_positions;
    _add_dropout = add_dropout;
    _max_drop = max_drop;
    _a = a;
    _b = b;
    _starting_w1 = starting_w1;
    _starting_w2 = starting_w2;
    _starting_w3 = starting_w3;
    _w1 = starting_w1;
    _w2 = starting_w2;
    _w3 = starting_w3;
    _initial_dropout_rate = 1;
    _vdq = colvec({ vd, 0.0 });
    _vmax = vdc * sqrt(3.0 / 2.0) / 2.0;
    _mu = 1.0;
    int neurons_dropped_w1 = 0;
    int neurons_dropped_w2 = 0;
    int neurons_dropped_w3 = 0;
    double global_importance_factor;
    double global_adaptive_rate;
    _mu_mat = mat(1, 1);
    _j_total_mat = mat(1, 1);

    _num_weights = _starting_w1.size() + _starting_w2.size() + _starting_w3.size();

    util::load_matrix_from_csv(_preloaded_idq_refs, "total_idq_refs.csv");

    // The master needs to know which samples are handled by each worker
    int rolling_sum = 0;
    for (int i = 0; i < active_workers - 1; i++)
    {
        rolling_sum += num_elems_to_process();
        _sender_indices[i + 1] = rolling_sum;
    }

    initialize_candidates(_w1, _w1_candidates);
    initialize_candidates(_w2, _w2_candidates);
    initialize_candidates(_w3, _w3_candidates);
}

// ==================================================================
pair<int, int> RNN::end_start_elems_to_process()
{
    int last_idx = (_rank + 1) * _num_samples / _active_workers;
    int first_idx = _rank * _num_samples / _active_workers;
    return { first_idx, last_idx };
}

// ==================================================================
int RNN::num_elems_to_process()
{
    if (_rank >= _num_samples)
    {
        return 0;
    }

    pair<int, int> end_start = end_start_elems_to_process();
    return end_start.second - end_start.first;
}

// ==================================================================
void RNN::send_matrices(vector<mat*> matrices, vector<int>& receivers, int tag)
{
    // We don't want to make multiple calls to send multiple matrices and we don't want to convert multiple matrices
    // multiple times to arrays. So we have this function to send multiple matrices to multiple senders to consolidate work
    // The caller just packages the pointers to the matrices into a vector
    // Then, this function iterates over those matrices to figure out how the total count of the data to send
    // Then we can know how to call one MPI_Send() rather than N sends for N matrices

    int total_count = util::count_in_matrices(matrices);
    double* payload = new double[total_count];
    util::mats_to_std_array(matrices, payload);
    for (int i = 0; i < receivers.size(); i++)
    {
        MPI_Send(payload, total_count, MPI_DOUBLE, receivers[i], tag, MPI_COMM_WORLD);
    }

    delete[] payload;
}

// ==================================================================
void RNN::send_matrices(vector<mat*> matrices, int receiver, int tag)
{
    vector<int> receivers = { receiver };
    send_matrices(matrices, receivers, tag);
}

// ==================================================================
MPI_Status RNN::recv_matrices(vector<mat*> matrices, int sender, int tag)
{
    // Like with send_matrices, the idea is to provide a vector of matrices
    // (the size of each matrix determined by the caller) and then we just fill those matrices
    // with the values from the sender
    int total_count = util::count_in_matrices(matrices);
    double* payload = new double[total_count];

    MPI_Status status;
    MPI_Recv(payload, total_count, MPI_DOUBLE, sender, tag, MPI_COMM_WORLD, &status);

    util::std_array_to_mats(payload, matrices);
    delete[] payload;
    return status;
}


// ======================================================================
// Implements calculateIdq_ref.m
// The point of this function is to generate a random reference, i.e. target value for training
// We are currently preloadding these values generated from Matlab for ease of comparison
colvec RNN::calculate_idq_ref(
    const int& trajectory_number,
    const int& time_step,
    const double& ts)
{
    // NOTE: trajectory_number and time_step come from zero-indexed, rather than one-indexed code
    // So when comparing to matlab, we must add an extra value
    float period = 0.1;
    int change_number = floor(util::round_to_decimal(time_step * ts / period, 2)) + 1;

    // Well the seed computation is inconsistent.
    colvec idq_ref;
    int seed = ((trajectory_number + 1) % 10) * 10000 + change_number;
    srand(seed);
    arma_rng::set_seed(seed);
    idq_ref = colvec({ -300, -300 }) + colvec({ 600, 360 }) % mat(2, 1, arma::fill::randu);
    idq_ref = arma::round(idq_ref * 10) / 10;

    // FIXME: The calculation for the seed is all fucked, so just use the values from matlab
    idq_ref = _preloaded_idq_refs.rows(time_step * 2, time_step * 2 + 1).col(trajectory_number);
    return idq_ref;
}

// ======================================================================
mat RNN::exdiag(const mat& x)
{
    mat y = mat(x.n_rows, x.n_elem, arma::fill::zeros);
    for (uword i = 0; i < x.n_rows; i++)
    {
        y.row(i).cols(x.n_cols * i, x.n_cols * (i + 1) - 1) = x.row(i);
    }
    return y;
}

double RNN::calculate_importance(const colvec& activations) {
    double mean_activation = mean(activations);
    double variance = accu(pow(activations - mean_activation, 2)) / activations.n_elem;
    double std_deviation = sqrt(variance);
    return std_deviation;
}

void RNN::apply_adaptive_dropout(colvec& neuron_outputs, double base_dropout_rate, double training_progress, int layer, bool verbose) {
    global_importance_factor = calculate_importance(neuron_outputs);
    // Ensure there's always a significant chance for dropout, especially early in training
    // global_adaptive_rate = std::max(base_dropout_rate * (1.0 - global_importance_factor) * (0.1 + 0.9 * training_progress), 0.1);
    global_adaptive_rate  = base_dropout_rate * (1.0 - global_importance_factor) * training_progress;
    // global_adaptive_rate = base_dropout_rate * (1.0 - sqrt(global_importance_factor)) * (1.0 - training_progress);
    // global_adaptive_rate = std::max(global_adaptive_rate, base_dropout_rate * 0.5);
    // if (verbose && _rank == 0) {
    //     std::cout << "Layer " << layer
    //               << " Dropout Rate: " << global_adaptive_rate
    //               << ", Importance Factor: " << importance_factor
    //               << std::endl;
    // }

    //     // Ensure the dropout rate is within a reasonable range
    // global_adaptive_rate = std::max(global_adaptive_rate, 0.1); // Minimum dropout rate
    // global_adaptive_rate = std::min(global_adaptive_rate, 0.95); // Maximum dropout rate

    // global_adaptive_rate = base_dropout_rate * (1.0 - global_importance_factor);
    // global_adaptive_rate = std::clamp(global_adaptive_rate, 0.1, 0.95);  // Ensure within bounds

    // std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    // std::default_random_engine generator(std::random_device{}());
    static std::default_random_engine generator;
    std::bernoulli_distribution distribution(1.0 - global_adaptive_rate);
    int count_dropped = 0;

    for (size_t i = 0; i < neuron_outputs.size(); ++i) {
        if (!distribution(generator)) {
            neuron_outputs(i) = 0.0;
            count_dropped++;
        }
    }

    // if (verbose && _rank == 0) {        
    //     std::cout << "Layer " << layer
    //               << ". Neurons dropped in this iteration: " << count_dropped 
    //               << std::endl;
    // }

    switch (layer) {
        case 1:
            neurons_dropped_w1 = count_dropped;
            break;
        case 2:
            neurons_dropped_w2 = count_dropped;
            break;
        case 3:
            neurons_dropped_w3 = count_dropped;
            break;
    }
}


// void RNN::apply_adaptive_dropout(colvec& neuron_outputs, double base_dropout_rate, double training_progress, int layer, bool verbose) {
//     global_importance_factor = calculate_importance(neuron_outputs);
//         // global_importance_factor = arma::mean(arma::abs(neuron_outputs));  // Updated to calculate mean of absolute values
//     //    global_adaptive_rate = base_dropout_rate * (1.0 - global_importance_factor) * training_progress;
//     global_adaptive_rate = base_dropout_rate * (1.0 - sqrt(global_importance_factor)) * (1.0 - training_progress);

//     // Ensure the dropout rate is within a reasonable range
//     global_adaptive_rate = std::max(global_adaptive_rate, 0.1); // Minimum dropout rate
//     global_adaptive_rate = std::min(global_adaptive_rate, 0.95); // Maximum dropout rate

//     static std::default_random_engine generator;
//     std::bernoulli_distribution distribution(1.0 - global_adaptive_rate);


//     int count_dropped = 0;
//     for (size_t i = 0; i < neuron_outputs.size(); ++i) {
//         if (!distribution(generator)) {
//             neuron_outputs(i) = 0.0;
//             count_dropped++;
//         }
//     }

//     switch (layer) {
//         case 1:
//             neurons_dropped_w1 = count_dropped;
//             break;
//         case 2:
//             neurons_dropped_w2 = count_dropped;
//             break;
//         case 3:
//             neurons_dropped_w3 = count_dropped;
//             break;
//     }

//     // if (verbose) {
//     //     std::cout << "Layer " << layer << " adaptive dropout rate: " << global_adaptive_rate
//     //               << ", neurons dropped: " << count_dropped << "/" << neuron_outputs.size()
//     //               << ", importance factor: " << global_importance_factor
//     //               << ", training progress: " << training_progress << std::endl;
//     // }
// }

// double RNN::calculate_importance(const colvec& activations) {
//     double mean_activation = mean(activations);
//     double variance = accu(pow(activations - mean_activation, 2)) / activations.n_elem;
//     double std_deviation = sqrt(variance);

//     return std_deviation;
//     return arma::stddev(activations);
//     // if (activations.n_elem == 0) return 0.0; // Avoid division by zero
//     // double mean_activation = mean(activations);
//     // double variance = var(activations, 1); // Unbiased estimate
//     // return sqrt(variance);
// }


// void RNN::apply_adaptive_dropout(colvec& neuron_outputs, double base_dropout_rate, double training_progress, int layer, bool verbose) {
//     global_importance_factor = calculate_importance(neuron_outputs);
//     global_adaptive_rate = base_dropout_rate * (1.0 - global_importance_factor) * (1.0 - training_progress);
//     global_adaptive_rate = std::max(global_adaptive_rate, 0.05); // Ensure there's a meaningful minimum threshold

//     std::default_random_engine generator(std::random_device{}());
//     std::bernoulli_distribution distribution(1.0 - global_adaptive_rate);
//     // global_adaptive_rate = base_dropout_rate * (1.0 - global_importance_factor) * training_progress;
//     // global_adaptive_rate = std::max(global_adaptive_rate, base_dropout_rate * 0.1);  // Ensuring a minimum dropout rate

//     // std::default_random_engine generator(std::random_device{}());
//     // std::bernoulli_distribution distribution(1.0 - global_adaptive_rate);

//     int count_dropped = 0;
//     for (size_t i = 0; i < neuron_outputs.size(); ++i) {
//         if (!distribution(generator)) {
//             neuron_outputs(i) = 0;
//             count_dropped++;
//         }
//     }

//     // if (isPrintable) {
//     //      std::cout << ", Layer: " << layer
//     //           << ", Dropout Rate: " << adaptive_rate
//     //           << ", Neurons Dropped: " << count_dropped << "/" << neuron_outputs.size()
//     //           << ", Importance: " << importance_factor
//     //           << ", Training Progress: " << training_progress << std::endl;
//     //     isPrintable = false;
//     // }

//     switch (layer) {
//         case 1: neurons_dropped_w1 = count_dropped; break;
//         case 2: neurons_dropped_w2 = count_dropped; break;
//         case 3: neurons_dropped_w3 = count_dropped; break;
//     }
// }

// void RNN::apply_adaptive_dropout(colvec& neuron_outputs, double base_dropout_rate, double training_progress, int layer, bool verbose) {
//     global_importance_factor = calculate_importance(neuron_outputs);
//     global_adaptive_rate = base_dropout_rate * (1.0 - global_importance_factor) * training_progress;

//     static std::default_random_engine generator;
//     std::bernoulli_distribution distribution(1.0 - global_adaptive_rate);

//     int count_dropped = 0;
//     for (size_t i = 0; i < neuron_outputs.size(); ++i) {
//         if (!distribution(generator)) {
//             neuron_outputs(i) = 0.0;
//             count_dropped++;
//         }
//     }

//     switch (layer) {
//         case 1:
//             neurons_dropped_w1 += count_dropped;
//             break;
//         case 2:
//             neurons_dropped_w2 += count_dropped;
//             break;
//         case 3:
//             neurons_dropped_w3 += count_dropped;
//             break;
//     }
// }

// void RNN::apply_adaptive_dropout(colvec& neuron_outputs, double base_dropout_rate, double training_progress, int layer, bool verbose) {
//     global_importance_factor = calculate_importance(neuron_outputs); // Ensure this function is dynamic
//     // Adjust the formula to have a more noticeable effect based on training progress and importance
//     global_adaptive_rate = base_dropout_rate * (1.0 - global_importance_factor) * (1.0 - training_progress);
//     global_adaptive_rate = std::max(global_adaptive_rate, base_dropout_rate * 0.1); // Ensure there's a meaningful minimum threshold

//     std::default_random_engine generator(std::random_device{}());
//     std::bernoulli_distribution distribution(1.0 - global_adaptive_rate);


//     int count_dropped = 0;
//     for (size_t i = 0; i < neuron_outputs.size(); ++i) {
//         if (!distribution(generator)) {  // Neuron is dropped with probability of `adaptive_rate`
//             neuron_outputs(i) = 0.0;
//             count_dropped++;
//         }
//     }

//     switch (layer) {
//         case 1:
//             neurons_dropped_w1 = count_dropped;
//             break;
//         case 2:
//             neurons_dropped_w2 = count_dropped;
//             break;
//         case 3:
//             neurons_dropped_w3 = count_dropped;
//             break;
//     }

//     // if (isPrintable) {
//     //     std::cout << "Layer " << layer << " adaptive dropout rate: " << adaptive_rate
//     //               << ", neurons dropped: " << count_dropped << "/" << neuron_outputs.size() << std::endl;
//     //     std::cout << "Importance Factor: " << importance_factor << std::endl;
//     //     std::cout << "Training Progress: " << training_progress << std::endl;
//     //     isPrintable = false;
//     // }
// }


// ======================================================================
// Implements netaction.m
// The main point of this function is to compute the final output of the neural network, o3
// It also computes dnet_dw, which is the gradient (part of the FATT algorithm) 
    void RNN::net_action(
        const colvec& idq,
        const colvec& idq_ref,
        const mat& hist_err,
        const mat& w1,
        const mat& w2,
        const mat& w3,
        const bool flag,
        mat& o3,
        mat& dnet_dw,
        mat& dnet_didq,
        mat& dnet_dhist_err,
        const bool is_training, // Make sure this indicates training mode
        double training_progress, // This should be calculated per iteration and passed in
        bool verbose
    )
    {
        colvec input0A = (idq / _gain1);
        colvec output0A = input0A.transform([](double val) { return tanh(val); });
        colvec input0B = (idq.rows(0, 1) - idq_ref) / _gain2;
        colvec output0B = input0B.transform([](double val) { return tanh(val); });
        colvec input0C = hist_err / _gain3;
        colvec output0C = input0C.transform([](double val) { return tanh(val); });

        colvec input1 = _use_idq ? join_vert(output0A, output0B, output0C, colvec({ -1 })) : join_vert(output0B, output0C, colvec({ -1 }));

        colvec sum1 = w1 * input1;
        colvec o1 = sum1.transform([](double val) { return tanh(val); });
        // Layer 1 Dropout
        if (is_training) {
            apply_adaptive_dropout(o1, 0.5, training_progress, 1, verbose);
        }

        colvec input2 = _use_shortcuts ? join_vert(o1, input1) : join_vert(o1, colvec({ -1 }));
        colvec sum2 = w2 * input2;
        colvec o2 = sum2.transform([](double val) { return tanh(val); });
        // Layer 2 Dropout
        if (is_training) {
            apply_adaptive_dropout(o2, 0.5, training_progress, 2, verbose);           
        }
 
        colvec input3 = _use_shortcuts ? join_vert(o2, input2) : join_vert(o2, colvec({ -1 }));
        colvec sum3 = w3 * input3;
        colvec o3_vec = sum3.transform([](double val) { return tanh(val); });  // Use o3_vec to avoid shadowing

        if (is_training) {
            apply_adaptive_dropout(o3_vec, 0.5, training_progress, 3, verbose);
        }

        o3 = o3_vec; // Assign the possibly dropped-out output to the original matrix reference


        mat do3_dw3, do3_do2, do3_dw2, do2_dw2, do2_do1, do1_dw1, do3_do1, do3_dw1, do3_do1_d3;
        mat dinput1_0A_0B_didq, do1_dinput1_0A_0B, do3_dinput1_0A_0B_d3, do3_dinput1_0A_0B;
        mat dinput1_0C_dhist_err, do1_dinput1_0C, do3_dinput1_0C, do2_dinput1_0A_0B_d2, do3_dinput1_0A_0B_d2;
        mat do3_dinput1_0C_d3, do2_dinput1_0C_d2, do3_dinput1_0C_d2;

        if (flag)
        {
            // compute Dnet_Dw
            // third layer
            do3_dw3 = (1 - o3 % o3) * input3.t();
            dnet_dw = exdiag(do3_dw3);

            // second layer
            do3_do2 = diagmat(1 - o3 % o3) * w3.cols(0, w2.n_rows - 1);
            do2_dw2 = exdiag((1 - o2 % o2) * input2.t());

            do3_dw2 = do3_do2 * do2_dw2;
            dnet_dw = join_horiz(dnet_dw, do3_dw2);

            // first layer
            do2_do1 = diagmat(1 - o2 % o2) * w2.cols(0, w1.n_rows - 1);

            if (_use_shortcuts)
            {
                do3_do1_d3 = diagmat(1 - o3 % o3) * w3.cols(w2.n_rows, w2.n_rows + w1.n_rows - 1);
                do3_do1 = do3_do1_d3 + do3_do2 * do2_do1;
            }
            else
            {
                do3_do1 = do3_do2 * do2_do1;
            }

            do1_dw1 = exdiag((1 - o1 % o1) * input1.t());
            do3_dw1 = do3_do1 * do1_dw1;

            dnet_dw = join_horiz(dnet_dw, do3_dw1);

            if (_use_idq)
            {
                dinput1_0A_0B_didq = join_vert(
                    diagmat((1 - output0A % output0A) / _gain1),
                    diagmat((1 - output0B % output0B) / _gain2),
                    mat(2, 4, arma::fill::zeros)
                );
            }
            else
            {
                dinput1_0A_0B_didq = diagmat((1 - output0B % output0B) / _gain2);
            }
            do1_dinput1_0A_0B = diagmat(1 - o1 % o1) * w1.cols(0, w1.n_cols - 4);

            // compute Dnet_Didq
            if (_use_shortcuts)
            {
                do3_dinput1_0A_0B_d3 = diagmat(1 - o3 % o3) * w3.cols(w2.n_rows + w1.n_rows, w3.n_cols - 4);
                do2_dinput1_0A_0B_d2 = diagmat(1 - o2 % o2) * w2.cols(w1.n_rows, w2.n_cols - 4);
                do3_dinput1_0A_0B_d2 = do3_do2 * do2_dinput1_0A_0B_d2;
                do3_dinput1_0A_0B = do3_do1 * do1_dinput1_0A_0B + do3_dinput1_0A_0B_d3 + do3_dinput1_0A_0B_d2;
            }
            else
            {
                do3_dinput1_0A_0B = do3_do1 * do1_dinput1_0A_0B;
            }

            dnet_didq = do3_dinput1_0A_0B * dinput1_0A_0B_didq;

            // compute dnet_dhist_err
            dinput1_0C_dhist_err = diagmat((1 - output0C % output0C) / _gain3);
            do1_dinput1_0C = diagmat(1 - o1 % o1) * w1.cols(w1.n_cols - 3, w1.n_cols - 2);

            if (_use_shortcuts)
            {
                do3_dinput1_0C_d3 = diagmat(1 - o3 % o3) * w3.cols(w3.n_cols - 3, w3.n_cols - 2);
                do2_dinput1_0C_d2 = diagmat(1 - o2 % o2) * w2.cols(w2.n_cols - 3, w2.n_cols - 2);
                do3_dinput1_0C_d2 = do3_do2 * do2_dinput1_0C_d2;
                do3_dinput1_0C = do3_do1 * do1_dinput1_0C + do3_dinput1_0C_d3 + do3_dinput1_0C_d2;
            }
            else
            {
                do3_dinput1_0C = do3_do1 * do1_dinput1_0C;
            }
            dnet_dhist_err = do3_dinput1_0C * dinput1_0C_dhist_err;
        }
    }

    // ============================================================================
    // Implementation of unrollTrajectoryFull.m
    // Main point of this algorithm is to compute the total cost of the neural network on the training data (j_total) for a each sample
    // It also computes the current jacobian matrix J_total, which is used in computing the update vector, delta, in the LM algorithm
    void RNN::unroll_trajectory_full(
        const colvec& initial_idq,
        const int& trajectory_number,
        const int& trajectory_length,
        const mat& w3,
        const mat& w2,
        const mat& w1,
        double& j_total,
        rowvec& e_hist_err,
        mat& j_matrix,
        mat& idq_his,
        mat& idq_ref_his,
        bool is_training,
        double training_progress,
        bool verbose
    )
    {

        colvec idq = initial_idq;

        idq_his = mat(6, trajectory_length, arma::fill::zeros);
        idq_ref_his = mat(2, trajectory_length, arma::fill::zeros);
        mat hist_err = mat(2, trajectory_length, arma::fill::zeros);
        e_hist_err = rowvec(trajectory_length, arma::fill::zeros);

        mat didq_dw, dvdq_dw, didq_dw_matrix_sum;
        didq_dw = mat(6, _num_weights, arma::fill::zeros);
        dvdq_dw = mat(2, _num_weights, arma::fill::zeros);
        j_matrix = mat(trajectory_length + 1, _num_weights, arma::fill::zeros);
        didq_dw_matrix_sum = mat(2, _num_weights, arma::fill::zeros);

        mat err_integral, dudq_dw;
        colvec idq_ref, idq_refi;

        // outputs of net_action
        mat o3, udq, ndq, dnet_dw, dnet_didq, dnet_dhist_err;
        for (int i = 0; i < trajectory_length; i++)
        {
            err_integral = _ts * (arma::sum(hist_err, 1) - hist_err.col(i) / 2.0);
            idq_ref = calculate_idq_ref(trajectory_number, i, _ts);
            idq_ref_his.col(i) = idq_ref;
            hist_err.col(i) = (idq.rows(0, 1) - idq_ref); // when the error is too small, the calculation becomes inaccurate
            e_hist_err.col(i) = pow((arma::sum(hist_err.col(i) % hist_err.col(i))), _cost_term_power / 2.0);// the calculation process of pow is slightly from that in Matlab

            net_action(idq, idq_ref, err_integral, w1, w2, w3, false, ndq, dnet_dw, dnet_didq, dnet_dhist_err, is_training, training_progress, verbose);
            udq = ndq * _vmax;

            net_action(idq, idq_ref, err_integral, w1, w2, w3, true, o3, dnet_dw, dnet_didq, dnet_dhist_err, is_training, training_progress, verbose);

            if (_use_idq)
            {
                dudq_dw = _vmax * (dnet_dw + dnet_didq * didq_dw + dnet_dhist_err * _ts * didq_dw_matrix_sum);
            }
            else
            {
                dudq_dw = _vmax * (dnet_dw + dnet_didq * didq_dw.rows(0, 1) + dnet_dhist_err * _ts * (didq_dw_matrix_sum - didq_dw.rows(0, 1) / 2.0));
            }

            didq_dw = _a * didq_dw + _b * join_vert(dvdq_dw, dudq_dw, dvdq_dw);
            didq_dw_matrix_sum = didq_dw_matrix_sum + didq_dw.rows(0, 1);
            idq = _a * idq + _b * join_vert(_vdq, udq, colvec({ 0, 0 }));

            // add saturation to dq currents
            // idq(1:5) = max(min(idq(1:4), 1000*ones(4,1)),-1000*ones(4,1));
            idq.rows(0, 3) = arma::max(arma::min(idq.rows(0, 3), 1000 * mat(4, 1, arma::fill::ones)), -1000 * mat(4, 1, arma::fill::ones));

            idq_his.col(i) = idq;
            idq_refi = calculate_idq_ref(trajectory_number, i + 1, _ts);
            j_matrix.row(i + 1) = (idq.rows(0, 1) - idq_refi).t() * didq_dw.rows(0, 1) * _cost_term_power * pow(arma::sum((idq.rows(0, 1) - idq_refi) % (idq.rows(0, 1) - idq_refi)), _cost_term_power / 2.0 - 1);
        }

        j_total = arma::sum(e_hist_err % e_hist_err);
        j_matrix.shed_row(j_matrix.n_rows - 1);
    }

// ============================================================================
void RNN::find_smallest_weight(const mat& W, const vector<pair<int, int>>& candidates, double& smallest, pair<int, int>& index)
{
    pair<int, int> p = candidates[0];
    double candidate_weight = abs(W(p.first, p.second));
    smallest = abs(W(p.first, p.second));
    for (int i = 0; i < candidates.size(); i++)
    {
        p = candidates[i];
        candidate_weight = abs(W(p.first, p.second));
        if (candidate_weight < smallest)
        {
            smallest = candidate_weight;
            index.first = p.first;
            index.second = p.second;
        }
    }
}

//// ============================================================================
void RNN::remove_smallest_weight(mat& W, pair<int, int> index, vector<pair<int, int>>& dropped, vector<pair<int, int>>& candidates)
{
    _last_weight_dropped = W(index.first, index.second);
    W(index.first, index.second) = 0.0;
    dropped.push_back(index);

    //// FIXME: I'm sure this is a very inefficient data structure, but can't be bothered to prematurely optimize
    for (int i = 0; i < candidates.size(); i++)
    {
        if (candidates[i].first == index.first && candidates[i].second == index.second)
        {
            candidates.erase(candidates.begin() + i);
            break;
        }
    }
}

// ============================================================================
void RNN::remove_smallest_weight(mat& w1, mat& w2, mat& w3)
{
    pair<int, int> w1_index, w2_index, w3_index;
    double w1_smallest, w2_smallest, w3_smallest;

    find_smallest_weight(w1, _w1_candidates, w1_smallest, w1_index);
    find_smallest_weight(w2, _w2_candidates, w2_smallest, w2_index);
    find_smallest_weight(w3, _w3_candidates, w3_smallest, w3_index);

    if (w1_smallest < w2_smallest && w1_smallest < w3_smallest)
    {
        cout << "Removing weight from w1, indices: " << w1_index.first << ", " << w1_index.second << ", value: " << w1(w1_index.first, w1_index.second) << endl;
        remove_smallest_weight(w1, w1_index, _w1_dropped, _w1_candidates);
    }
    else if (w2_smallest < w1_smallest && w2_smallest < w3_smallest)
    {
        cout << "Removing weight from w2, indices: " << w2_index.first << ", " << w2_index.second << ", value: " << w2(w2_index.first, w2_index.second) << endl;
        remove_smallest_weight(w2, w2_index, _w2_dropped, _w2_candidates);
    }
    else if (w3_smallest < w1_smallest && w3_smallest < w2_smallest)
    {
        cout << "Removing weight from w3, indices: " << w3_index.first << ", " << w3_index.second << ", value: " << w3(w3_index.first, w3_index.second) << endl;
        remove_smallest_weight(w3, w3_index, _w3_dropped, _w3_candidates);
    }
}

// ==================================================================
void RNN::apply_dropout(mat& W, vector<pair<int, int>>& dropped)
{
    pair<int, int> to_drop;
    for (int i = 0; i < dropped.size(); i++)
    {
        to_drop = dropped.at(i);
        W(to_drop.first, to_drop.second) = 0.0;
    }
    return;
}

// ============================================================================
void RNN::initialize_candidates(const mat& M, vector<pair<int, int>>& candidates)
{
    pair<int, int> p;
    for (int r = 0; r < M.n_rows; r++)
    {
        for (int c = 0; c < M.n_cols; c++)
        {
            // We never consider diagonal elements
            if (r != c)
            {
                p = { r, c };
                candidates.push_back(p);
            }
        }
    }
}

void RNN::train_best_weights(const int max_iterations, bool verbose)
{
    const int samples_to_process = num_elems_to_process();
    if (samples_to_process == 0) { return; }

    const pair<int, int> range_of_samples = end_start_elems_to_process();
    double mu = _starting_mu;
    int current_drop = _w1_dropped.size() + _w2_dropped.size() + _w3_dropped.size();
    colvec idq, dw, dw_y, rr;
    mat j_matrix, j_matrix2;
    rowvec e_hist_err, e_hist_err2;
    mat jj, ii, h_matrix;
    double j_total, j_total_subsum, j_total_sum;
    double j_total2, j_total_subsum2, j_total_sum2;
    vector<mat*> matrices;
    mat w1_temp = mat(_w1.n_rows, _w1.n_cols);
    mat w2_temp = mat(_w2.n_rows, _w2.n_cols);
    mat w3_temp = mat(_w3.n_rows, _w3.n_cols);
    mat mu_mat = mat(1, 1);
    mat j_total_mat = mat(1, 1);
    bool success;
    MPI_Status status;
    j_total_sum = 0;
    j_total_sum2 = 0;
    double dropout_rate = _initial_dropout_rate;
    unsigned int seed = 12345;
    bool is_training = _add_dropout;
    double training_progress = 0;

    for (int iteration = 1; iteration <= max_iterations; iteration++)
    {
        reset_dropout_counters();
        if (_add_dropout) {
            is_training = true;
            training_progress = static_cast<double>(iteration) / static_cast<double>(max_iterations);
        } else {
            is_training = false;
        }

        mat idq_his, idq_ref_his;
        rowvec hist_err_subtotal = rowvec(_trajectory_length * samples_to_process, arma::fill::zeros);
        mat j_matrix_subtotal = mat(_trajectory_length * samples_to_process, _num_weights, arma::fill::zeros);
        j_total_subsum = 0;

        for (int i = range_of_samples.first, j = 0; i < range_of_samples.second; i++, j++)
        {
            idq = _idq_start_positions.col(i);
            unroll_trajectory_full(idq, i, _trajectory_length, _w3, _w2, _w1, j_total, e_hist_err, j_matrix, idq_his, idq_ref_his, is_training, training_progress, verbose);
            j_total_subsum += j_total;
            hist_err_subtotal.cols(j * (_trajectory_length), (j + 1) * (_trajectory_length) - 1) = e_hist_err;
            j_matrix_subtotal.rows(j * (_trajectory_length), (j + 1) * (_trajectory_length) - 1) = j_matrix;
        }

        if (_rank == 0)
        {
            rowvec hist_err_total = rowvec(_trajectory_length * _num_samples, arma::fill::zeros);
            mat j_matrix_total = mat(_trajectory_length * _num_samples, _num_weights, arma::fill::zeros);
            j_total_sum = j_total_subsum;
            hist_err_total.cols(0, samples_to_process * _trajectory_length - 1) = hist_err_subtotal;
            j_matrix_total.rows(0, samples_to_process * _trajectory_length - 1) = j_matrix_subtotal;

            for (int senderid = 1; senderid < _active_workers; senderid++)
            {
                int sender_samples = num_elems_to_process();
                hist_err_subtotal = rowvec(_trajectory_length * sender_samples, arma::fill::zeros);
                j_matrix_subtotal = mat(_trajectory_length * sender_samples, _num_weights, arma::fill::zeros);
                j_total_mat.zeros();
                status = recv_matrices({ &j_total_mat, &hist_err_subtotal, &j_matrix_subtotal }, senderid);
                int start = _sender_indices[senderid] * _trajectory_length;
                int end = start + (sender_samples * _trajectory_length) - 1;
                hist_err_total.cols(start, end) = hist_err_subtotal;
                j_matrix_total.rows(start, end) = j_matrix_subtotal;
                j_total_sum += j_total_mat.at(0, 0);
            }

            mat L, R, Q; // Corrected here, declaring Q and R
            while (mu < _mu_max)
            {
                jj = j_matrix_total.t() * j_matrix_total;
                ii = -1 * j_matrix_total.t() * arma::vectorise(hist_err_total);
                h_matrix = jj + mu * arma::eye(_num_weights, _num_weights);
                success = util::cholesky_eigen(L, R, h_matrix);
                if (!success)
                {
                    // Attempt QR decomposition if Cholesky fails
                    
                    arma::qr_econ(Q, R, h_matrix);
                    success = arma::solve(dw_y, Q, ii, arma::solve_opts::fast);
                    if (!success)
                    {
                        mu = _starting_mu; // Reset mu on failure
                        cout << "Cholesky decomposition failed, condition number too high, resetting mu." << endl;
                        continue;
                    }
                    mu = mu * _mu_inc;
                    if (mu == _mu_max)
                    {
                        break;
                    }
                    h_matrix = jj + mu * arma::eye(_num_weights, _num_weights);
                    success = util::cholesky_eigen(L, R, h_matrix);
                    arma::solve(dw, R, dw_y, arma::solve_opts::fast);
                }
                else
                {
                    arma::solve(dw_y, L, ii);
                    arma::solve(dw, R, dw_y);
                }

                w3_temp = _w3 + arma::reshape(dw.rows(0, _w3.n_elem - 1), _w3.n_cols, _w3.n_rows).t();
                w2_temp = _w2 + arma::reshape(dw.rows(_w3.n_elem, _w3.n_elem + _w2.n_elem - 1), _w2.n_cols, _w2.n_rows).t();
                w1_temp = _w1 + arma::reshape(dw.rows(_w3.n_elem + _w2.n_elem, dw.n_elem - 1), _w1.n_cols, _w1.n_rows).t();

                send_matrices({ &w1_temp, &w2_temp, &w3_temp }, _workerids, MPI_NEW_JOB);
                j_total_sum2 = 0;
                j_total2 = 0;
                for (int i = range_of_samples.first; i < range_of_samples.second; i++)
                {
                    idq = _idq_start_positions.col(i);
                    unroll_trajectory_full(idq, i, _trajectory_length, w3_temp, w2_temp, w1_temp, j_total2, e_hist_err2, j_matrix2, idq_his, idq_ref_his, is_training, training_progress, verbose);
                    j_total_sum2 += j_total2;
                }

                for (int i = 1; i < _active_workers; i++)
                {
                    j_total_subsum2 = 0;
                    MPI_Recv(&j_total_subsum2, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_J_TOTAL, MPI_COMM_WORLD, &status);
                    j_total_sum2 += j_total_subsum2;
                }

                if (j_total_sum2 < j_total_sum)
                {
                    j_total_sum = j_total_sum2;
                    _w3 = w3_temp;
                    _w2 = w2_temp;
                    _w1 = w1_temp;
                    rr = join_cols(rr, colvec(j_total_sum2 / _trajectory_length / _num_samples));
                    mu = std::max(mu * _mu_dec, _mu_min);
                    if (verbose)
                    {
                        isPrintable = true;
                       auto dropped_neurons = count_dropped_neurons();
                        if(_add_dropout){
                            std::cout << std::setprecision(16)
                                    << "iteration: " << iteration
                                    << ", mu=" << mu
                                    << ", J_total_sum2=" << j_total_sum2 / _trajectory_length / 10
                                    << ", dropout_rate " << global_adaptive_rate << ", "
                                    << "importance_factor " << global_importance_factor
                                    << ", Neurons dropped: "
                                    << "Layer 1: " << std::get<0>(dropped_neurons) << ", "
                                    << "Layer 2: " << std::get<1>(dropped_neurons) << ", "
                                    << "Layer 3: " << std::get<2>(dropped_neurons) 
                                    << std::endl;
                        } else {
                             std::cout << std::setprecision(16)
                                    << "iteration: " << iteration
                                    << ", mu=" << mu
                                    << ", J_total_sum2=" << j_total_sum2 / _trajectory_length / 10
                                    << ", Neurons dropped: "
                                    << "Layer 1: " << std::get<0>(dropped_neurons) << ", "
                                    << "Layer 2: " << std::get<1>(dropped_neurons) << ", "
                                    << "Layer 3: " << std::get<2>(dropped_neurons) 
                                    << std::endl;
                        }
                    }
                    break;
                }
                mu = mu * _mu_inc;
            }
            mu_mat = mat({ mu });
            send_matrices({ &_w1, &_w2, &_w3, &mu_mat }, _workerids, MPI_LOOP_DONE);
        }
        else
        {
            j_total_mat.at(0, 0) = j_total_subsum;
            send_matrices({ &j_total_mat, &hist_err_subtotal, &j_matrix_subtotal }, 0, MPI_JOB_FINISHED);
            while (true)
            {
                status = recv_matrices({ &w1_temp, &w2_temp, &w3_temp, &mu_mat }, 0);
                if (status.MPI_TAG == MPI_NEW_JOB)
                {
                    j_total_subsum2 = 0;
                    for (int i = range_of_samples.first; i < range_of_samples.second; i++)
                    {
                        idq = _idq_start_positions.col(i);
                        unroll_trajectory_full(idq, i, _trajectory_length, w3_temp, w2_temp, w1_temp, j_total2, e_hist_err2, j_matrix2, idq_his, idq_ref_his, is_training, training_progress, verbose);
                        j_total_subsum2 += j_total2;
                    }
                    MPI_Send(&j_total_subsum2, 1, MPI_DOUBLE, 0, MPI_J_TOTAL, MPI_COMM_WORLD);
                }
                else if (status.MPI_TAG == MPI_LOOP_DONE)
                {
                    _w3 = w3_temp;
                    _w2 = w2_temp;
                    _w1 = w1_temp;
                    mu = mu_mat.at(0, 0);
                    break;
                }
            }
        }
        if (mu == _mu_max)
        {
            if (_add_dropout)
            {
                double current_cost = j_total_sum / _trajectory_length / 10;
                cout << "Current cost: " << current_cost << endl;
                mu = _starting_mu; // Reset mu to prevent infinite loop
            }
            break;
        }
    }
}


// void RNN::train_best_weights(const int max_iterations, bool verbose)
// {
//     // each worker computes how many elements it needs to process    
//     const int samples_to_process = num_elems_to_process();
//     // Some workers may have nothing to do
//     if (samples_to_process == 0) { return; }
//     // Each worker computes its reserved indices for depositing values into the aggregation variables
//     const pair<int, int> range_of_samples = end_start_elems_to_process();
//     double mu = _starting_mu;
//     int current_drop = _w1_dropped.size() + _w2_dropped.size() + _w3_dropped.size();
//     colvec idq, dw, dw_y, rr;
//     mat j_matrix, j_matrix2;
//     rowvec e_hist_err, e_hist_err2;
//     mat jj, ii, h_matrix;
//     double j_total, j_total_subsum, j_total_sum;
//     double j_total2, j_total_subsum2, j_total_sum2;
//     vector<mat*> matrices;
//     mat w1_temp = mat(_w1.n_rows, _w1.n_cols);
//     mat w2_temp = mat(_w2.n_rows, _w2.n_cols);
//     mat w3_temp = mat(_w3.n_rows, _w3.n_cols);
//     mat mu_mat = mat(1, 1);
//     mat j_total_mat = mat(1, 1);
//     bool success;
//     MPI_Status status;
//     j_total_sum = 0;
//     j_total_sum2 = 0;
//     double dropout_rate = _initial_dropout_rate;
//     unsigned int seed = 12345; // Choose a seed value
//     bool is_training = _add_dropout;
//     double training_progress = 0;
//     for (int iteration = 1; iteration < max_iterations + 1; iteration++)
//     {
//         reset_dropout_counters();
//         //  double adapt_factor = 1.0 - static_cast<double>(iteration) / max_iterations;
//         // mu = _starting_mu * adapt_factor + _mu_min * (1 - adapt_factor); // Ensure mu doesn't fall below _mu_min
//         if (_add_dropout) {
//             is_training = true;  // Enable training mode
//             training_progress = static_cast<double>(iteration) / static_cast<double>(max_iterations);  // Update progress for dropout calculation
//         } else {
//             is_training = false;  // Disable training mode if not dropping out
//         }
//         // else {
//         //     bool is_training = false;
//         //     double training_progress = 0;
//         // }
//         // double training_progress = static_cast<double>(iteration) / static_cast<double>(max_iterations);      
//         // Use FATT to calculate total cost of each trajectory, the error vector, and the jacobian matrix. 
//         mat idq_his, idq_ref_his;
//         rowvec hist_err_subtotal = rowvec(_trajectory_length * samples_to_process, arma::fill::zeros);
//         mat j_matrix_subtotal = mat(_trajectory_length * samples_to_process, _num_weights, arma::fill::zeros);
//         j_total_subsum = 0;
//         // Each worker (master included) does its own part for this loop
//         for (int i = range_of_samples.first, j = 0; i < range_of_samples.second; i++, j++)
//         {
//             idq = _idq_start_positions.col(i);
//             unroll_trajectory_full(idq, i, _trajectory_length, _w3, _w2, _w1, j_total, e_hist_err, j_matrix, idq_his, idq_ref_his, is_training, training_progress, verbose);
//             j_total_subsum += j_total;
//             hist_err_subtotal.cols(j * (_trajectory_length), (j + 1) * (_trajectory_length)-1) = e_hist_err;
//             j_matrix_subtotal.rows(j * (_trajectory_length), (j + 1) * (_trajectory_length)-1) = j_matrix;
//         }
//         // Master will aggregate results from workers
//         if (_rank == 0)
//         {
//             rowvec hist_err_total = rowvec(_trajectory_length * _num_samples, arma::fill::zeros);
//             mat j_matrix_total = mat(_trajectory_length * _num_samples, _num_weights, arma::fill::zeros);
//             j_total_sum = 0;
//             // The master first adds its piece to the aggregation variables, then waits to get chunks from the workers
//             j_total_sum = j_total_subsum;
//             hist_err_total.cols(0, samples_to_process * _trajectory_length - 1) = hist_err_subtotal;
//             j_matrix_total.rows(0, samples_to_process * _trajectory_length - 1) = j_matrix_subtotal;
//             for (int senderid = 1; senderid < _active_workers; senderid++)
//             {
//                 int sender_samples = num_elems_to_process();
//                 // we receive these subtotals from sender
//                 hist_err_subtotal = rowvec(_trajectory_length * sender_samples, arma::fill::zeros);
//                 j_matrix_subtotal = mat(_trajectory_length * sender_samples, _num_weights, arma::fill::zeros);
//                 j_total_mat.zeros();
//                 status = recv_matrices({ &j_total_mat, &hist_err_subtotal, &j_matrix_subtotal }, senderid);
//                 int start = _sender_indices[senderid] * _trajectory_length;
//                 int end = start + (sender_samples * _trajectory_length) - 1;
//                 // and update our totals
//                 hist_err_total.cols(start, end) = hist_err_subtotal;
//                 j_matrix_total.rows(start, end) = j_matrix_subtotal;
//                 j_total_sum += j_total_mat.at(0, 0);
//             }
//             j_total_sum = j_total_sum;
//             // Now that we've computed j_total, hist_err_total, and j_matrix_total, 
//             // master can do cholensky decomposition to solve for weight updates
//             mat L = mat(_num_weights, _num_weights, arma::fill::zeros);
//             mat R = mat(_num_weights, _num_weights, arma::fill::zeros);
//             while (mu < _mu_max)
//             {
//                 jj = j_matrix_total.t() * j_matrix_total;
//                 ii = -1 * j_matrix_total.t() * arma::vectorise(hist_err_total);
//                 h_matrix = jj + mu * arma::eye(_num_weights, _num_weights);
//                 // cholensky decomposition to solve for weight updates
//                 // FIXME: Armadillo chol just doesn't work? Use Eigen instead
//                 // std::cout << "Attempting Cholesky with mu=" << mu << std::endl;
//                 success = util::cholesky_eigen(L, R, h_matrix);
//                 while (!success)
//                 {
//                     mu = mu * _mu_inc;
//                     if (mu == _mu_max)
//                     {
//                         break;
//                     }
//                     h_matrix = jj + mu * arma::eye(_num_weights, _num_weights);
//                     success = util::cholesky_eigen(L, R, h_matrix);
//                 }
//                 if (mu == _mu_max)
//                 {
//                     cout << "Cholesky decomp failed!!!" << endl;
//                     break;
//                 }
//                 arma::solve(dw_y, L, ii);
//                 arma::solve(dw, R, dw_y);
//                 w3_temp = _w3 + arma::reshape(dw.rows(0, _w3.n_elem), _w3.n_cols, _w3.n_rows).t();
//                 w2_temp = _w2 + arma::reshape(dw.rows(_w3.n_elem, _w3.n_elem + _w2.n_elem), _w2.n_cols, _w2.n_rows).t();
//                 w1_temp = _w1 + arma::reshape(dw.rows(_w3.n_elem + _w2.n_elem, dw.n_elem - 1), _w1.n_cols, _w1.n_rows).t();
//                 // dropout_rate = _initial_dropout_rate * (1.0 - static_cast<double>(iteration) / (max_iterations +  1) );
               
//                 // As soon as we've computed these new weights, we can give them to the workers to do their pieces
//                 send_matrices({ &w1_temp, &w2_temp, &w3_temp }, _workerids, MPI_NEW_JOB);
//                 // Master needs to compute its piece of the sum
//                 j_total_sum2 = 0;
//                 j_total2 = 0;
//                 for (int i = range_of_samples.first; i < range_of_samples.second; i++)
//                 {
//                     idq = _idq_start_positions.col(i);
//                     unroll_trajectory_full(idq, i, _trajectory_length, w3_temp, w2_temp, w1_temp, j_total2, e_hist_err2, j_matrix2, idq_his, idq_ref_his, is_training, training_progress, verbose);
//                     j_total_sum2 += j_total2;
//                 }
//                 // Master needs to get sums from the workers
//                 for (int i = 1; i < _active_workers; i++)
//                 {
//                     j_total_subsum2 = 0;
//                     MPI_Recv(&j_total_subsum2, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_J_TOTAL, MPI_COMM_WORLD, &status);
//                     j_total_sum2 += j_total_subsum2;
//                 }
//                 if (j_total_sum2 < j_total_sum)
//                 {
//                     j_total_sum = j_total_sum2;
//                     _w3 = w3_temp;
//                     _w2 = w2_temp;
//                     _w1 = w1_temp;
//                     rr = join_cols(rr, colvec(j_total_sum2 / _trajectory_length / _num_samples));
//                     mu = std::max(mu * _mu_dec, _mu_min);
//                     if (verbose)
//                     {
//                         isPrintable = true;
//                         auto dropped_neurons = count_dropped_neurons();
//                         if(_add_dropout){
//                             std::cout << std::setprecision(16)
//                                     << "iteration: " << iteration
//                                     << ", mu=" << mu
//                                     << ", J_total_sum2=" << j_total_sum2 / _trajectory_length / 10
//                                     << ", dropout_rate " << global_adaptive_rate << ", "
//                                     << "importance_factor " << global_importance_factor
//                                     << ", Neurons dropped: "
//                                     << "Layer 1: " << std::get<0>(dropped_neurons) << ", "
//                                     << "Layer 2: " << std::get<1>(dropped_neurons) << ", "
//                                     << "Layer 3: " << std::get<2>(dropped_neurons) 
//                                     << std::endl;
//                         } else {
//                              std::cout << std::setprecision(16)
//                                     << "iteration: " << iteration
//                                     << ", mu=" << mu
//                                     << ", J_total_sum2=" << j_total_sum2 / _trajectory_length / 10
//                                     // << ", dropout_rate " << global_adaptive_rate << ", "
//                                     // << "importance_factor " << global_importance_factor
//                                     << ", Neurons dropped: "
//                                     << "Layer 1: " << std::get<0>(dropped_neurons) << ", "
//                                     << "Layer 2: " << std::get<1>(dropped_neurons) << ", "
//                                     << "Layer 3: " << std::get<2>(dropped_neurons) 
//                                     << std::endl;
//                         }

//                         // Check if the dropped neurons count is as expected
//                         // if (std::get<0>(dropped_neurons) > _w1.n_rows || 
//                         //     std::get<1>(dropped_neurons) > _w2.n_rows || 
//                         //     std::get<2>(dropped_neurons) > _w3.n_rows) {
//                         //     throw std::runtime_error("Dropped neurons count exceeds the total neurons in the layer");
//                         // }
//                     }
//                     break;
//                 }
//                 mu = mu * _mu_inc;
//             } // while mu < mu_max loop ends here
//             // Anytime we break out of this loop, we need to let our workers know
//             mu_mat = mat({ mu });
//             send_matrices({ &_w1, &_w2, &_w3, &mu_mat }, _workerids, MPI_LOOP_DONE);
//         }
//         // The workers have done their work, so they just send to the master and wait for the aggregation
//         else
//         {
//             // Then we send the matrices
//             j_total_mat.at(0, 0) = j_total_subsum;
//             send_matrices({ &j_total_mat, &hist_err_subtotal, &j_matrix_subtotal }, 0, MPI_JOB_FINISHED);
//             // Each worker will stay in this loop until master has decided we are done updating weights and mu
//             while (true)
//             {
//                 status = recv_matrices({ &w1_temp, &w2_temp, &w3_temp, &mu_mat }, 0);
//                 if (status.MPI_TAG == MPI_NEW_JOB)
//                 {
//                     j_total_subsum2 = 0;
//                     for (int i = range_of_samples.first; i < range_of_samples.second; i++)
//                     {
//                         idq = _idq_start_positions.col(i);
//                         unroll_trajectory_full(idq, i, _trajectory_length, w3_temp, w2_temp, w1_temp, j_total2, e_hist_err2, j_matrix2, idq_his, idq_ref_his, is_training, training_progress,verbose);
//                         j_total_subsum2 += j_total2;
//                     }
//                     MPI_Send(&j_total_subsum2, 1, MPI_DOUBLE, 0, MPI_J_TOTAL, MPI_COMM_WORLD);
//                 }
//                 else if (status.MPI_TAG == MPI_LOOP_DONE)
//                 {
//                     _w3 = w3_temp;
//                     _w2 = w2_temp;
//                     _w1 = w1_temp;
//                     mu = mu_mat.at(0, 0);
//                     break;
//                 }
//             }
//         }
//         //  cout << "mu: "  << mu << "_mu_max" << _mu_max << endl;
//         if (mu == _mu_max)
//         {
//             // FIXME: This doesn't work with parallelization right now...
//             if (_add_dropout)
//             {
//                 double current_cost = j_total_sum / _trajectory_length / 10;
//                 cout << "Current cost: " << current_cost  << endl;
//                 // if (current_drop < _max_drop)
//                 // {
//                     // cout << "Removing weight when max_drop is " << _max_drop << endl;
//                     // Then we continue training until max iterations, after removing one weight at a time
//                     // remove_smallest_weight(_w1, _w2, _w3);
//                     // current_drop++;
//                     cout << "Current cost: " << current_cost << endl;
//                     auto dropped_neurons = count_dropped_neurons();
//                     cout << "Max mu reached, resetting mu. "
//                          << std::setprecision(16)
//                          << "iteration: " << iteration
//                          << ", mu=" << mu
//                          << ", J_total_sum2=" << j_total_sum2 / _trajectory_length / 10
//                          << ", dropout_rate " << global_adaptive_rate << ", "
//                          << "importance_factor " << global_importance_factor
//                          << ", Neurons dropped: "
//                          << "Layer 1: " << std::get<0>(dropped_neurons) << ", "
//                          << "Layer 2: " << std::get<1>(dropped_neurons) << ", "
//                          << "Layer 3: " << std::get<2>(dropped_neurons) << ", " 
//                         // << "dropout_rate " << global_adaptive_rate << ", "
//                         // << "importance_factor " << global_importance_factor << ", "
//                         << endl;
//                     // cout << "Max mu reached, resetting mu. Dropped neurons: " << count_dropped_neurons() << endl; // Include neuron dropout info
//                     // Reset mu and potentially adjust dropout strategy
//                     // We have to reset mu since necessarily it will have been maxed at this point
//                     mu = _starting_mu;
//                     // break;
//                     // continue;
//                 // }
//             }
//             isPrintable = true;
//             break;
//         }
//         // std::cout << "dropout rate: iteration" << dropout_rate << std::endl;
//     }
//     // std::cout << "Total Dropped Weights: " << _w1_dropped.size() + _w2_dropped.size() + _w3_dropped.size() << "dropout rate: " << dropout_rate << std::endl;
// }

// ============================================================================
void RNN::set_weights(const mat& w1, const mat& w2, const mat& w3)
{
    clear_weights();
    _starting_w1 = w1;
    _starting_w2 = w2;
    _starting_w3 = w3;
    _w1 = w1;
    _w2 = w2;
    _w3 = w3;
    _num_weights = w1.size() + w2.size() + w3.size();
    initialize_candidates(_w1, _w1_candidates);
    initialize_candidates(_w2, _w2_candidates);
    initialize_candidates(_w3, _w3_candidates);
}

// ============================================================================
void RNN::clear_weights()
{
    _w1.clear();
    _w2.clear();
    _w3.clear();
    _starting_w1.clear();
    _starting_w2.clear();
    _starting_w3.clear();
    _num_weights = 0;
    _w1_dropped.clear();
    _w2_dropped.clear();
    _w3_dropped.clear();
    _w1_candidates.clear();
    _w2_candidates.clear();
    _w3_candidates.clear();
}

// ============================================================================
void RNN::get_weights(mat& w1, mat& w2, mat& w3)
{
    w1 = _w1;
    w2 = _w2;
    w3 = _w3;
}

void RNN::reset_dropout_counters() {
    neurons_dropped_w1 = 0;
    neurons_dropped_w2 = 0;
    neurons_dropped_w3 = 0;
}

std::tuple<int, int, int> RNN::count_dropped_neurons() {
    return std::make_tuple(neurons_dropped_w1, neurons_dropped_w2, neurons_dropped_w3);
}
