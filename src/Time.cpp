#include <map>
#include <chrono>
#include <tuple>
#include <iostream>
#include <fstream>

using namespace std;
using timepoint_t = std::chrono::system_clock::time_point;
using timeinterval_t = std::pair<timepoint_t,timepoint_t>;

namespace init {
    enum { total, read_dataset, init_data, init_lsh, rev_lsh, end };
}
namespace embed {
    enum { total=init::end+1, alloc, rand_str, measure, compute, end };
}
namespace buckets {
    enum { total=embed::end+1, measure, compute, end };
}
namespace sort_buckets {
    enum { total = buckets::end+1, end };
}
namespace cand_init {
    enum { total=sort_buckets::end+1, comp_buck_delim, filter_buck_delim, resize, scan_cand, end };
}
namespace cand {
    enum { total=cand_init::end+1, measure, compute, end };
}
namespace cand_proc {
    enum { total=cand::end+1, rem_cand, sort_cand, count_freq, rem_dup, sort_cand_to_verify, filter_low_freq, make_uniq, end };
}
namespace edit_dist {
    enum{ total=cand_proc::end+1, end };
}
namespace total_join{
    enum{ total=edit_dist::end+1, end };
}
namespace total_alg{
    enum { total=total_join::end+1 };
}

class Time {
public:
	void start_time(int phase_id){
		record_time(0, phase_id);
	}

	void end_time(int phase_id){
		record_time(1, phase_id);
	}

	double get_step_time(int phase_id){
		double t=get_time(timing[phase_id]);
		return t;
	}

	void print_report(std::string dev, uint32_t num_candidates, uint32_t num_outputs, std::ostream &out_file=std::cout){

		out_file<<"Step,SubStep,Time(sec),Device"<<std::endl;
		
		double t=get_time(timing[init::total]);
		out_file<<"Read Data,\t,"<<t<<std::endl;
		
		t=get_time(timing[init::init_data]);
		out_file<<"\t,Init Dataset,"<<t<<std::endl;

		t=get_time(timing[init::init_lsh]);
		out_file<<"\t,Init LSH bits,"<<t<<std::endl;

		t=get_time(timing[init::rev_lsh]);
		out_file<<"\t,Init Rev LSH array,"<<t<<std::endl;

		t=get_time(timing[embed::total]);
		out_file<<"Embedding,\t,"<<t<<","<<dev<<std::endl;

		t=get_time(timing[embed::alloc]);
		out_file<<"\t,USM allocation,"<<t<<std::endl;

		t=get_time(timing[embed::rand_str]);
		out_file<<"\t,Random string generation,"<<t<<std::endl;

		t=get_time(timing[embed::measure]);
		out_file<<"\t,Measurement,"<<t<<std::endl;

		t=get_time(timing[embed::compute]);
		out_file<<"\t,Computing,"<<t<<std::endl;

		t=get_time(timing[buckets::total]);
		out_file<<"Create Buckets,\t,"<<t<<","<<dev<<std::endl;

		t=get_time(timing[buckets::measure]);
		out_file<<"\t,Measurement,"<<t<<std::endl;

		t=get_time(timing[buckets::compute]);
		out_file<<"\t,Computing,"<<t<<std::endl;

		t=get_time(timing[sort_buckets::total]);
		out_file<<"Sort Buckets,\t,"<< t <<std::endl;

		t=get_time(timing[cand_init::total]);
		out_file<<"Candidate Initialization,\t,"<<t<<std::endl;

		t=get_time(timing[cand_init::comp_buck_delim]);
		out_file<<"\t,Compute buckets delimiter,"<<t<<std::endl;

		t=get_time(timing[cand_init::filter_buck_delim]);
		out_file<<"\t,Filter one element buckets,"<<t<<std::endl;

		t=get_time(timing[cand_init::resize]);
		out_file<<"\t,Allocate candidate vector,"<<t<<std::endl;

		t=get_time(timing[cand_init::scan_cand]);
		out_file<<"\t,Scan cand vector (write i and j),"<<t<<std::endl;

		t=get_time(timing[cand::total]);
		out_file<<"Generate Candidate,\t,"<< t <<","<<dev<<std::endl;

		t=get_time(timing[cand::measure]);
		out_file<<"\t,Measurement,"<<t<<std::endl;

		t=get_time(timing[cand::compute]);
		out_file<<"\t,Computing,"<<t<<std::endl;

		t=get_time(timing[cand_proc::total]);
		out_file<<"Candidates processing,\t,"<<t<<std::endl;

		t=get_time(timing[cand_proc::rem_cand]);
		out_file<<"\t,Remove candidates,"<<t<<std::endl;

		t=get_time(timing[cand_proc::sort_cand]);
		out_file<<"\t,Sort candidates,"<<t<<std::endl;

		t=get_time(timing[cand_proc::count_freq]);
		out_file<<"\t,Counting frequencies,"<<t<<std::endl;

		t=get_time(timing[cand_proc::rem_dup]);
		out_file<<"\t,Remove duplicates,"<<t<<std::endl;

		t=get_time(timing[cand_proc::sort_cand_to_verify]);
		out_file<<"\t,Sorting candidates to verify,"<<t<<std::endl;

		t=get_time(timing[cand_proc::filter_low_freq]);
		out_file<<"\t,Remove low frequencies candidates,"<<t<<std::endl;

		t=get_time(timing[cand_proc::make_uniq]);
		out_file<<"\t,Removing duplicates,"<<t<<std::endl;

		t=get_time(timing[edit_dist::total]);
		out_file<<"Edit Distance,\t,"<<t<<std::endl;

		t=get_time(timing[total_join::total]);
		out_file<<"Total Join time (w/o embedding),\t,"<<t<<std::endl;

		t=get_time(timing[total_alg::total]);
		out_file<<"Total Alg time,\t,"<<t<<std::endl;

		out_file<<"Number candidates,\t"<<num_candidates<<std::endl;
		out_file<<"Number output,\t"<<num_outputs<<std::endl;
	}

	void print_summary(uint32_t num_candidates, uint32_t num_outputs){

		std::cout<<"\n\n\nSummary:"<<std::endl<<std::endl;

		double t=get_time(timing[init::total]);
		std::cout<<"Time init input data: "<<t<<std::endl;

		t=get_time(timing[embed::total]);
		std::cout<<"Time PARALLEL embedding data:\t"<<t<<"sec"<<std::endl;

		t=get_time(timing[buckets::total]);
		std::cout<<"Time PARALLEL buckets generation:\t"<< t<<"sec"<<std::endl;

		t=get_time(timing[sort_buckets::total]);
		std::cout<<"Time buckets sorting:\t"<< t <<"sec"<<std::endl;

		t=get_time(timing[cand_init::total]);
		std::cout<<"Time candidate initialization:\t"<< t<<"sec"<<std::endl;

		t=get_time(timing[cand::total]);
		std::cout<<"Time PARALLEL candidates generation:\t"<< t<<"sec"<<std::endl;

		t=get_time(timing[cand_proc::total]);
		std::cout<<"Time candidates processing:\t"<< t<<"sec"<<std::endl;

		t=get_time(timing[cand_proc::sort_cand]);
		std::cout<<"Time candidates sorting (timing[within cand-processing]):\t"<< t<<"sec"<<std::endl;

		t=get_time(timing[edit_dist::total]);
		std::cout<<"Time compute edit distance:\t"<<t <<"sec"<<std::endl;

		t=get_time(timing[total_join::total]);
		std::cout<<"Total time parallel join:\t"<< t<<"sec"<<std::endl;

		t=get_time(timing[total_alg::total]);
		std::cout<<"Total elapsed time :\t"<< t<<"sec"<<std::endl;
		std::cout<<"Number of candidates verified: "<<num_candidates<<std::endl;
		std::cout<<"Number of output pairs: "<<num_outputs<<std::endl;
	}

private:
	map<int,timeinterval_t> timing;
	timepoint_t t=std::chrono::system_clock::now();

	double get_time(timeinterval_t time){
		long d=std::chrono::duration_cast<std::chrono::milliseconds>(time.second-time.first).count();
		double t=(double)d/1000.0;
		return t;
	}

	double get_time_diff(int phase_id){
		double res = 0.0;
		res=get_time(timing[phase_id]);
		return res;
	}

	void record_time(int i, int phase_id){
		if(i==0){
			timing[phase_id].first=std::chrono::system_clock::now();
		}else if(i==1){
			timing[phase_id].second=std::chrono::system_clock::now();
		}
	}
};
