#include "embedjoin.hpp"

class Time {

private:

	std::chrono::system_clock::time_point t=std::chrono::system_clock::now();

	std::vector<std::chrono::system_clock::time_point> read_time; // 0,0,0
	std::vector<std::chrono::system_clock::time_point> read_data_time; // 0,0,1
	std::vector<std::chrono::system_clock::time_point> init_lsh_time; // 0,0,2
	std::vector<std::chrono::system_clock::time_point> init_rev_lsh_time; // 0,0,3


	std::vector<std::chrono::system_clock::time_point> embeddata; // 0,1,0
	std::vector<std::chrono::system_clock::time_point> embeddata_allocation_USM; // 0,1,1
	std::vector<std::chrono::system_clock::time_point> embeddata_generate_random_string; // 0,1,2
	std::vector<std::chrono::system_clock::time_point> embeddata_measure; // 0,1,3
	std::vector<std::chrono::system_clock::time_point> embeddata_computation; // 0,1,4




	std::vector<std::chrono::system_clock::time_point> create_buckets; // 0,2,0
	std::vector<std::chrono::system_clock::time_point> create_buckets_measure; // 0,2,1
	std::vector<std::chrono::system_clock::time_point> create_buckets_computation; // 0,2,2

	std::vector<std::chrono::system_clock::time_point> sorting_buckets; // 0,3,0


	std::vector<std::chrono::system_clock::time_point> init_cand; // 0,4,0
	std::vector<std::chrono::system_clock::time_point> init_cand_compute_buckets_delim; // 0,4,1
	std::vector<std::chrono::system_clock::time_point> init_cand_remove_one_elem_buckets; // 0,4,2
	std::vector<std::chrono::system_clock::time_point> init_cand_resize; // 0,4,3
	std::vector<std::chrono::system_clock::time_point> init_cand_scan_cand_vector; // 0,4,4



	std::vector<std::chrono::system_clock::time_point> generate_cand; // 0,5,0
	std::vector<std::chrono::system_clock::time_point> generate_cand_measure; // 0,5,1
	std::vector<std::chrono::system_clock::time_point> generate_cand_computation; // 0,5,2

	std::vector<std::chrono::system_clock::time_point> cand_proc; // 0,6,0
	std::vector<std::chrono::system_clock::time_point> cand_proc_remove_candidates; // 0,6,1
	std::vector<std::chrono::system_clock::time_point> cand_proc_sorting_cand; // 0,6,2
	std::vector<std::chrono::system_clock::time_point> cand_proc_counting_frequencies; // 0,6,3
	std::vector<std::chrono::system_clock::time_point> cand_proc_remove_duplicates; // 0,6,4
	std::vector<std::chrono::system_clock::time_point> cand_proc_sorting_candidates_to_verify; // 0,6,5
	std::vector<std::chrono::system_clock::time_point> cand_proc_removing_low_frequencies_cand; // 0,6,6
	std::vector<std::chrono::system_clock::time_point> cand_proc_removing_duplicates_from_cand_to_verify; // 0,6,7

	std::vector<std::chrono::system_clock::time_point> edit_dist; // 0,7,0

	std::vector<std::chrono::system_clock::time_point> total_join_time; // 1,0,0
	std::vector<std::chrono::system_clock::time_point> total_alg_time; // 2,0,0


	double get_time(std::vector<std::chrono::system_clock::time_point> time){
		long d=std::chrono::duration_cast<std::chrono::milliseconds>(time[1]-time[0]).count();
		double t=(double)d/1000.0;
		return t;
	}

	template<class Func>
	double find_case(int i, int main_phase, int phase, int subphase, Func to_do){


			double res=0;

			if(main_phase==1){
				res=to_do(this->total_join_time);
			}
			else if(main_phase==2){
				res=to_do(this->total_alg_time);
			}
			else if(main_phase==0){

				if(phase==0){

					if(subphase==0){
						res=to_do(this->read_time);
					}
					if(subphase==1){
						res=to_do(this->read_data_time);
					}
					if(subphase==2){
						res=to_do(this->init_lsh_time);
					}
					if(subphase==3){
						res=to_do(this->init_rev_lsh_time);
					}

				}
				if(phase==1){
					if(subphase==0){
						res=to_do(this->embeddata);
					}
					if(subphase==1){
						res=to_do(this->embeddata_allocation_USM);
					}
					if(subphase==2){
						res=to_do(this->embeddata_generate_random_string);
					}
					if(subphase==3){
						res=to_do(this->embeddata_measure);
					}
					if(subphase==4){
						res=to_do(this->embeddata_computation);
					}
				}

				if(phase==2){
					if(subphase==0){
						res=to_do(this->create_buckets);
					}
					if(subphase==1){
						res=to_do(this->create_buckets_measure);
					}
					if(subphase==2){
						res=to_do(this->create_buckets_computation);
					}

				}
				if(phase==3){

					if(subphase==0){
						res=to_do(this->sorting_buckets);
					}

				}
				if(phase==4){
					if(subphase==0){
						res=to_do(this->init_cand);
					}
					if(subphase==1){
						res=to_do(this->init_cand_compute_buckets_delim);
					}
					if(subphase==2){
						res=to_do(this->init_cand_remove_one_elem_buckets);
					}
					if(subphase==3){
						res=to_do(this->init_cand_resize);
					}
					if(subphase==4){
						res=to_do(this->init_cand_scan_cand_vector);
					}

				}
				if(phase==5){
					if(subphase==0){
						res=to_do(this->generate_cand);
					}
					if(subphase==1){
						res=to_do(this->generate_cand_measure);
					}
					if(subphase==2){
						res=to_do(this->generate_cand_computation);
					}
				}
				if(phase==6){
					if(subphase==0){
						res=to_do(this->cand_proc);
					}
					if(subphase==1){
						res=to_do(this->cand_proc_remove_candidates);
					}
					if(subphase==2){
						res=to_do(this->cand_proc_sorting_cand);
					}
					if(subphase==3){
						res=to_do(this->cand_proc_counting_frequencies);
					}
					if(subphase==4){
						res=to_do(this->cand_proc_remove_duplicates);
					}
					if(subphase==5){
						res=to_do(this->cand_proc_sorting_candidates_to_verify);
					}
					if(subphase==6){
						res=to_do(this->cand_proc_removing_low_frequencies_cand);
					}
					if(subphase==7){
						res=to_do(this->cand_proc_removing_duplicates_from_cand_to_verify);
					}
				}
				if(phase==7){
					if(subphase==0){
						res=to_do(this->edit_dist);
					}
				}
			}

			return res;

	}



	double get_time_diff(int main_phase, int phase, int subphase){

		double res = 0.0;
		res=find_case(0, main_phase, phase, subphase, [&](std::vector<std::chrono::system_clock::time_point> &time){
			double t=this->get_time(time);
			return t;
		});

		return res;

	}

	void record_time(int i, int main_phase, int phase, int subphase){


		find_case(i, main_phase, phase, subphase, [&](std::vector<std::chrono::system_clock::time_point> &time){
			auto new_t=std::chrono::system_clock::now();
			time[i]=new_t;
			return 0;
		});


	}



public:
	Time(){

		this->read_time.resize(2); // 0,0,0
		this->read_data_time.resize(2); // 0,0,1
	    this->init_lsh_time.resize(2); // 0,0,2
		this->init_rev_lsh_time.resize(2); // 0,0,3

		this->embeddata.resize(2); // 0,1,0
		this->embeddata_allocation_USM.resize(2); // 0,1,1
		this->embeddata_generate_random_string.resize(2); // 0,1,2
		this->embeddata_measure.resize(2); // 0,1,3
		this->embeddata_computation.resize(2); // 0,1,4




		this->create_buckets.resize(2); // 0,2,0
		this->create_buckets_measure.resize(2); // 0,2,1
		this->create_buckets_computation.resize(2); // 0,2,2

		this->sorting_buckets.resize(2); // 0,3,0


		this->init_cand.resize(2); // 0,4,0
		this->init_cand_compute_buckets_delim.resize(2); // 0,4,1
		this->init_cand_remove_one_elem_buckets.resize(2); // 0,4,2
		this->init_cand_resize.resize(2); // 0,4,3
		this->init_cand_scan_cand_vector.resize(2); // 0,4,4



		this->generate_cand.resize(2); // 0,5,0
		this->generate_cand_measure.resize(2); // 0,5,1
		this->generate_cand_computation.resize(2); // 0,5,2

		this->cand_proc.resize(2); // 0,6,0
		this->cand_proc_remove_candidates.resize(2); // 0,6,1
		this->cand_proc_sorting_cand.resize(2); // 0,6,2
		this->cand_proc_counting_frequencies.resize(2); // 0,6,3
		this->cand_proc_remove_duplicates.resize(2); // 0,6,4
		this->cand_proc_sorting_candidates_to_verify.resize(2); // 0,6,5
		this->cand_proc_removing_low_frequencies_cand.resize(2); // 0,6,6
		this->cand_proc_removing_duplicates_from_cand_to_verify.resize(2); // 0,6,7

		this->edit_dist.resize(2); // 0,7,0

		this->total_join_time.resize(2); // 1,0,0
		this->total_alg_time.resize(2); // 2,0,0
	}

	void print_report(std::string dev, int num_candidates, int num_outputs, std::ostream &out_file=std::cout){

		out_file<<"Step,SubStep,Time(sec),Device"<<std::endl;

		double t=get_time(this->read_time);
		out_file<<"Read Data,\t,"<<t<<std::endl;

		t=get_time(this->read_data_time);
		out_file<<"\t,Read Data and initial sorting,"<<t<<std::endl;

		t=get_time(this->init_lsh_time);
		out_file<<"\t,Init LSH bits,"<<t<<std::endl;

		t=get_time(this->init_rev_lsh_time);
		out_file<<"\t,Init Rev LSH array,"<<t<<std::endl;

		t=get_time(this->embeddata);
		out_file<<"Embedding,\t,"<<t<<","<<dev<<std::endl;

		t=get_time(this->embeddata_allocation_USM);
		out_file<<"\t,USM allocation,"<<t<<std::endl;

		t=get_time(this->embeddata_generate_random_string);
		out_file<<"\t,Random string generation,"<<t<<std::endl;

		t=get_time(this->embeddata_measure);
		out_file<<"\t,Measurement,"<<t<<std::endl;

		t=get_time(this->embeddata_computation);
		out_file<<"\t,Computing,"<<t<<std::endl;

		t=get_time(this->create_buckets);
		out_file<<"Create Buckets,\t,"<<t<<","<<dev<<std::endl;

		t=get_time(this->create_buckets_measure);
		out_file<<"\t,Measurement,"<<t<<std::endl;

		t=get_time(this->create_buckets_computation);
		out_file<<"\t,Computing,"<<t<<std::endl;

		t=get_time(this->sorting_buckets);
		out_file<<"Sort Buckets,\t,"<< t <<std::endl;

		t=get_time(this->init_cand);
		out_file<<"Candidate Initialization,\t,"<<t<<std::endl;

		t=get_time(this->init_cand_compute_buckets_delim);
		out_file<<"\t,Compute buckets delimiter,"<<t<<std::endl;

		t=get_time(this->init_cand_remove_one_elem_buckets);
		out_file<<"\t,Filter one element buckets,"<<t<<std::endl;

		t=get_time(this->init_cand_resize);
		out_file<<"\t,Allocate candidate vector,"<<t<<std::endl;

		t=get_time(this->init_cand_scan_cand_vector);
		out_file<<"\t,Scan cand vector (write i and j),"<<t<<std::endl;

		t=get_time(this->generate_cand);
		out_file<<"Generate Candidate,\t,"<< t <<","<<dev<<std::endl;

		t=get_time(this->generate_cand_measure);
		out_file<<"\t,Measurement,"<<t<<std::endl;

		t=get_time(this->generate_cand_computation);
		out_file<<"\t,Computing,"<<t<<std::endl;

		t=get_time(this->cand_proc);
		out_file<<"Candidates processing,\t,"<<t<<std::endl;

		t=get_time(this->cand_proc_remove_candidates);
		out_file<<"\t,Remove candidates,"<<t<<std::endl;

		t=get_time(this->cand_proc_sorting_cand);
		out_file<<"\t,Sort candidates,"<<t<<std::endl;

		t=get_time(this->cand_proc_counting_frequencies);
		out_file<<"\t,Counting frequencies,"<<t<<std::endl;

		t=get_time(this->cand_proc_remove_duplicates);
		out_file<<"\t,Remove duplicates,"<<t<<std::endl;

		t=get_time(this->cand_proc_sorting_candidates_to_verify);
		out_file<<"\t,Sorting candidates to verify,"<<t<<std::endl;

		t=get_time(this->cand_proc_removing_low_frequencies_cand);
		out_file<<"\t,Remove low frequencies candidates,"<<t<<std::endl;

		t=get_time(this->cand_proc_removing_duplicates_from_cand_to_verify);
		out_file<<"\t,Removing duplicates,"<<t<<std::endl;



		t=get_time(this->edit_dist);
		out_file<<"Edit Distance,\t,"<<t<<std::endl;

		t=get_time(this->total_join_time);
		out_file<<"Total Join time (w/o embedding),\t,"<<t<<std::endl;

		t=get_time(this->total_alg_time);
		out_file<<"Total Alg time,\t,"<<t<<std::endl;

		out_file<<"Number candidates,\t"<<num_candidates<<std::endl;
		out_file<<"Number output,\t"<<num_outputs<<std::endl;
	}

	void start_time(int main_phase, int phase, int subphase){

		record_time(0, main_phase, phase, subphase );

	}


	void end_time(int main_phase, int phase, int subphase){

		record_time(1, main_phase, phase, subphase );

	}

	double get_step_time(int main_phase, int phase, int subphase){

		double t=get_time_diff(main_phase, phase, subphase);
		return t;
	}



};


