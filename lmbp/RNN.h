#pragma once

#include <iomanip>
#include <armadillo>

using namespace std;
using namespace arma;

#define _USE_MATH_DEFINES

class RNN
{

private:

	int _rank;
	vector<int> _workerids;
	int _active_workers;
	int _num_samples;
	bool _use_shortcuts;
	bool _use_idq;
	bool _add_dropout;
	double _vd;
	double _t_final;
	double _ts;
	double _vdc;
	colvec _idq_ref_centre;
	double _gain1;
	double _gain2;
	double _gain3;
	double _cost_term_power;
	double _starting_mu;
	double _mu_dec;
	double _mu_inc;
	double _mu_max;
	double _mu_min;
	double _dropout_max_extra_cost;
	int _max_drop;
	mat _idq_start_positions;
	mat _a;
	mat _b;
	mat _w1;
	mat _w2;
	mat _w3;
	mat _starting_w1;
	mat _starting_w2;
	mat _starting_w3;
	double _initial_dropout_rate;
	int neurons_dropped_w1 = 0;
    int neurons_dropped_w2 = 0;
    int neurons_dropped_w3 = 0;
	double global_importance_factor;
    double global_adaptive_rate;
	bool isPrintable;
    colvec _vdq;
    double _vmax;

    double _mu;
	int _trajectory_length;

    mat _mu_mat;
    mat _j_total_mat;
    mat _preloaded_idq_refs;

    int _num_weights;

	map<int, int> _sender_indices;

	vector<pair<int, int>> _w1_dropped, _w2_dropped, _w3_dropped;
	vector<pair<int, int>> _w1_candidates, _w2_candidates, _w3_candidates;

	// Kind of a dumb variable and only used for testing
	double _last_weight_dropped;

protected:

	pair<int, int> end_start_elems_to_process();
	int num_elems_to_process();

	void send_matrices(vector<mat*> matrices, vector<int>& receivers, int tag);
	void send_matrices(vector<mat*> matrices, int receiver, int tag);
	MPI_Status recv_matrices(vector<mat*> matrices, int sender, int tag = MPI_ANY_TAG);

	colvec calculate_idq_ref(const int& trajectory_number, const int& time_step, const double& ts);
	mat exdiag(const mat& x);

	void net_action(
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
	);

	void remove_smallest_weight(mat& w1, mat& w2, mat& w3);
	void remove_smallest_weight(mat& W, pair<int, int> index, vector<pair<int, int>>& dropped, vector<pair<int, int>>& candidates);
	void find_smallest_weight(const mat& W, const vector<pair<int, int>>& candidates, double& smallest, pair<int, int>& index);
	void apply_dropout(mat& W, vector<pair<int, int>>& dropped);
	void initialize_candidates(const mat& M, vector<pair<int, int>>& candidates);
	// void apply_adaptive_dropout(colvec& neuron_outputs, double base_dropout_rate, double training_progress);
	double calculate_importance(const colvec& activations);
	void apply_adaptive_dropout(colvec& neuron_outputs, double base_dropout_rate, double training_progress, int layer, bool verbose);
    void apply_adaptive_neuron_dropout(mat& W, vector<int>& dropped_neurons);
	void reset_dropout_counters();
    void prepare_matrices(arma::rowvec& hist_err_subtotal, arma::mat& j_matrix_subtotal);
    void simulate_computation(int index, double& j_total, arma::rowvec& hist_err_subtotal, arma::mat& j_matrix_subtotal);
    void aggregate_results(arma::mat& j_matrix_total, arma::rowvec& hist_err_total, const arma::rowvec& hist_err_subtotal, const arma::mat& j_matrix_subtotal);
    void update_weights(const arma::colvec& dw);
    void broadcast_weights();
    void send_results_to_master(const arma::rowvec& hist_err_subtotal, const arma::mat& j_matrix_subtotal) ;
    void print_verbose_output(int iteration, double mu, double j_total_sum);
	void send_matrices_to_workers(double mu);
	void receive_new_weights_and_update(double &mu);
	double compute_new_weights_and_costs(double &j_total2);
    std::tuple<int, int, int> count_dropped_neurons();

public:

	// ------------------------------------------------------------------------
	// getters
	double get_ts() { return _ts; }
	double get_trajectory_length() { return _trajectory_length; }
	void get_weights(mat &w1, mat& w2, mat &w3);
	double get_last_weight_dropped() { return _last_weight_dropped; }

	// ------------------------------------------------------------------------
	// setters
	void set_weights(const mat& w1, const mat& w2, const mat& w3);
	void set_dropout(const bool add_dropout) { _add_dropout = add_dropout; }
	void set_max_drop(const int max_drop) { _max_drop = max_drop; }

	void clear_weights();
	void train_best_weights(const int max_iterations, bool verbose);
bool solve_with_fallback(const arma::mat& jj, const arma::vec& ii, arma::vec& dw, double& mu);

	void unroll_trajectory_full(
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
	);
	

	RNN(
		int rank,
		vector<int> workerids, 
		int active_workers, 
		int num_samples,
		const mat idq_start_positions,
		const bool add_dropout = false,
		const double dropout_max_extra_cost = 0.0,
		const int max_drop = 0,
		const bool use_shortcuts = false,
		const bool use_idq = false,
		const double vd = 690,
		const double t_final = 1,
		const double ts = 0.001,
		const double vdc = 1200,
		const colvec idq_ref_centre = colvec({ 0, 0 }),
		const double gain1 = 1000.0,
		const double gain2 = 100.0,
		const double gain3 = 100.0,
		const double cost_term_power = 1.0 / 2.0,
		const double starting_mu = 1.0,
		const double mu_dec = 0.1,
		const double mu_inc = 10,
		const double mu_max = 1e10,
		const double mu_min = 1e-20,
		const mat a = {
			{ 0.922902679404235, 0.365403020170600, 0.001311850123628, 0.000519398207289, -0.006031602076712, -0.002388080200093},
			{ -0.365403020170600, 0.922902679404235, -0.000519398207289, 0.001311850123628, 0.002388080200093, -0.006031602076712},
			{ 0.001311850123628, 0.000519398207289, 0.922902679404235, 0.365403020170600, 0.006031602076712, 0.002388080200093},
			{ -0.000519398207289, 0.001311850123628, -0.365403020170600, 0.922902679404235, -0.002388080200093, 0.006031602076712},
			{ 0.120632041534246, 0.047761604001858, -0.120632041534246, -0.047761604001858, 0.921566702872299, 0.364874069642510},
			{ -0.047761604001858, 0.120632041534245, 0.047761604001858, -0.120632041534245, -0.364874069642510, 0.921566702872299}
		},

		const mat b = {
			{0.488106762997528, 0.093547911260568, -0.485485431756243, -0.091984091707451, -0.000001945097416, 0.000009097619657 },
			{-0.093547911260568, 0.488106762997528, 0.091984091707451, -0.485485431756243, -0.000009097619657, -0.000001945097416 },
			{0.485485431756243, 0.091984091707451, -0.488106762997528, -0.093547911260568, 0.000001945097416, -0.000009097619657 },
			{-0.091984091707451, 0.485485431756243, 0.093547911260568, -0.488106762997528, 0.000009097619657, 0.000001945097416 },
			{0.038901948324797, -0.181952393142100, 0.038901948324797, -0.181952393142100, 0.000002613550852, 0.000001600210032 },
			{0.181952393142100, 0.038901948324797, 0.181952393142100, 0.038901948324797, -0.000001600210032, 0.000002613550852 }
		},

		mat starting_w1 = {
			{ 0.081472368639318, 0.027849821886705, 0.095716694824295, 0.079220732955955, 0.067873515485777 },
			{ 0.090579193707562, 0.054688151920498, 0.048537564872284, 0.095949242639290, 0.075774013057833 },
			{ 0.012698681629351, 0.095750683543430, 0.080028046888880, 0.065574069915659, 0.074313246812492 },
			{ 0.091337585613902, 0.096488853519928, 0.014188633862722, 0.003571167857419, 0.039222701953417 },
			{ 0.063235924622541, 0.015761308167755, 0.042176128262628, 0.084912930586878, 0.065547789017756 },
			{ 0.009754040499941, 0.097059278176062, 0.091573552518907, 0.093399324775755, 0.017118668781156 }
		},

		mat starting_w2 = {
			{ 0.012649986532930, 0.031747977514944, 0.055573794271939, 0.055778896675488, 0.025779225057201, 0.040218398522248, 0.087111112191539 },
			{ 0.013430330431357, 0.031642899914629, 0.018443366775765, 0.031342898993659, 0.039679931863314, 0.062067194719958, 0.035077674488589 },
			{ 0.009859409271100, 0.021756330942282, 0.021203084253232, 0.016620356290215, 0.007399476957694, 0.015436980547927, 0.068553570874754 },
			{ 0.014202724843193, 0.025104184601574, 0.007734680811268, 0.062249725927990, 0.068409606696201, 0.038134520444447, 0.029414863376785 },
			{ 0.016825129849153, 0.089292240528598, 0.091380041077957, 0.098793473495250, 0.040238833269616, 0.016113397184936, 0.053062930385689 },
			{ 0.019624892225696, 0.070322322455629, 0.070671521769693, 0.017043202305688, 0.098283520139395, 0.075811243132742, 0.083242338628518 }
		},

		mat starting_w3 = {
			{ 0.002053577465818, 0.065369988900825, 0.016351236852753, 0.079465788538875, 0.044003559576025, 0.075194639386745, 0.006418708739190 },
			{ 0.092367561262041, 0.093261357204856, 0.092109725589220, 0.057739419670665, 0.025761373671244, 0.022866948210550, 0.076732951077657 }
		}
		// arma::mat starting_w1 = arma::randu(6, 5) * 0.1,
		// arma::mat starting_w2 = arma::randu(6, 7) * 0.1,
		// arma::mat starting_w3 = arma::randu(2, 7) * 0.1
		// 26 dropped weights
		// mat starting_w1 = {
		// 	{2.6017226945184574e-02,9.3321893780273432e-03,7.1694990937832215e-02,3.0767069474848259e-02,1.8714498730574083e-02},
		// 	{1.6535285228736044e-02,9.2032070683753173e-02,8.9400401184256073e-02,6.8846710917021478e-02,8.4033527047935611e-02},
		// 	{1.5367763730381727e-02,9.2164572551194118e-02,1.9948819249277480e-02,9.1166388944195723e-02,2.5972712878042856e-02},
		// 	{5.1951095268894408e-03,4.2322582905167605e-02,4.8901564963139188e-02,4.0437940113391496e-02,4.4088795003771393e-02},
		// 	{6.6690258960113616e-02,3.5597992482335028e-02,3.7697076921540514e-02,4.5381831737452423e-02,2.0339125052263869e-03},
		// 	{2.5703658429188522e-02,7.9576511973432809e-02,6.1355757867034379e-02,9.6366349131855919e-02,2.2803493830770322e-02}
		// },

		// mat starting_w2 = {
		// 	{5.8178762021293989e-02,4.1297344366980175e-02,5.9246097111230238e-02,8.7337739611089427e-02,6.4401008046970784e-02,5.4702648559666368e-03,8.7194976053234827e-02},
		// 	{7.7381159897524573e-02,2.2661985180341373e-02,7.4475870136886863e-02,8.5419753275384924e-02,2.1687588534892895e-02,5.5247256823080489e-04,4.8245495873682993e-02},
		// 	{2.8865435917991279e-02,6.7651789400322335e-02,3.3700265964798103e-02,3.8728838199995466e-02,9.4562582903589903e-02,8.3642161913303533e-03,4.0353652612016663e-02},
		// 	{4.4588897018944954e-03,8.8098546145730552e-02,3.4532031537871377e-02,5.8734332294169182e-02,3.2025508548347202e-02,4.3867113079560850e-02,3.9473245615460360e-02},
		// 	{7.7847080047317926e-02,4.6200839453197692e-03,9.8986214537362199e-02,2.7222541772259270e-02,8.1210575452888345e-02,8.6596576465246294e-02,7.5835148753603768e-02},
		// 	{9.2485618284426252e-02,5.4134522583166304e-02,2.3886482526025297e-02,9.5826948812799434e-02,9.1595757660315621e-02,5.2910088676692793e-02,5.2050986432166393e-02}
		// },

		// mat starting_w3 = {
		// 	{1.3372101461435103e-02,1.2101497556834123e-02,4.3310186902223220e-02,2.5055605364473588e-02,4.3529490265014854e-02,1.9267390154414074e-02,9.4301534107121956e-02},
		// 	{5.3957394103250800e-04,4.4331464111164544e-02,2.4859792032217325e-02,3.7228604716963890e-02,9.5829606858297112e-02,9.2988527319155551e-02,7.5519110581212913e-03}
		// }

		// mat starting_w1 = {
		// 	{6.6827492838616095e-02,3.5467354400481967e-03,4.0425718606161266e-02,7.5824607093471497e-02,7.2550888533995078e-02},
		// 	{1.8752564765536717e-02,6.2011939711612112e-02,2.2284495964929634e-02,3.5119712069028426e-02,4.8361275504762012e-02},
		// 	{3.4016773124608372e-02,2.3977842303994974e-02,1.8839376418456066e-02,1.0240395234746272e-02,6.7551797617700440e-04},
		// 	{2.6436359275702155e-02,1.0490067926669504e-02,6.5183871863862975e-03,8.4641606269633934e-02,1.0589752329950160e-02},
		// 	{9.4606459658734154e-02,7.8566661051463926e-02,2.0972869906008558e-02,8.9322901498579688e-02,6.8959479797571790e-02},
		// 	{9.0630339202100077e-02,3.5542677849221174e-02,9.2790000798263550e-02,2.4426500850940471e-02,6.8403474662688882e-02}
		// },

		// mat starting_w2 = {
		// 	{3.1063028865659106e-02,7.6080807631814115e-02,7.1167332107079914e-02,7.1529789864823265e-02,8.9660159390517274e-02,3.4604415856726127e-02,3.8790064536821024e-02},
		// 	{5.4167247979557576e-02,1.9183218345788233e-02,1.1762852169842023e-02,2.6037475256153925e-04,1.4873360804608562e-02,4.8635395686873049e-02,8.3200002460677561e-02},
		// 	{6.3499926158259154e-02,9.3225490226681890e-02,9.1051020471759669e-02,6.5607342747286898e-02,6.3416776793776067e-02,6.6117377965576124e-02,2.9994189598653923e-02},
		// 	{8.4940824785087854e-02,2.9724634317583827e-02,6.1964179209789950e-02,2.1700419797977266e-02,9.7075884405100996e-02,1.0205032845866956e-02,7.5131592151709593e-04},
		// 	{6.6486582737569674e-02,3.4326045305444540e-03,6.9764122626043812e-02,3.5919244877634274e-02,3.7868418900927736e-02,3.1196039016662760e-02,2.3600219300706125e-02},
		// 	{3.5473539488174839e-02,9.5848768642191454e-02,9.4199175007076835e-02,3.9842103651861888e-02,6.3238604816263075e-02,2.4005238755392080e-02,5.5589051515087265e-02}
		// },

		// mat starting_w3 = {
		// 	{2.4589145546945787e-02,3.2843423455221198e-02,3.6683109892647232e-03,6.0840577332884796e-02,5.9448956488691918e-02,1.6767864360390596e-02,7.0339303455703925e-02},
		// 	{2.7050906552162541e-02,6.4212209460572586e-02,5.0427049971105697e-02,6.7349620523446591e-02,8.6620567857571093e-02,1.6277768653674651e-02,6.7304692972364952e-02}
		// }
	);
};
