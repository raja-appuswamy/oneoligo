#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <tuple>

using namespace std;
using timepoint_t = std::chrono::system_clock::time_point;
using timeinterval_t = std::pair<timepoint_t, timepoint_t>;

namespace init {
enum { total, init_data, init_lsh, rev_lsh, end };
}
namespace embed {
enum { total = init::end + 1, alloc, rand_str, measure, compute, end };
}
namespace buckets {
enum { total = embed::end + 1, allocation, measure, compute, sort, end };
}
namespace cand_init {
enum {
  total = buckets::end + 1,
  comp_buck_delim,
  filter_buck_delim,
  resize,
  scan_cand,
  end
};
}
namespace cand {
enum { total = cand_init::end + 1, measure, compute, end };
}
namespace cand_proc {
enum {
  total = cand::end + 1,
  rem_cand,
  sort_cand,
  count_freq,
  rem_dup,
  sort_cand_to_verify,
  filter_low_freq,
  make_uniq,
  end
};
}
namespace edit_dist {
enum { total = cand_proc::end + 1, end };
}
namespace lsh {
enum { total = edit_dist::end + 1, end };
}
namespace total_alg {
enum { total = lsh::end + 1, end };
}
namespace cluster {
enum {
  total = total_alg::end + 1,
  init,
  onejoin,
  create_indexes,
  sort,
  dbscan,
  consensus,
  end
};
}

class Time {
public:

  Time(bool is_clust){
    is_cluster=is_clust;
  }

  void start_time(int phase_id) { record_time(0, phase_id); }

  void end_time(int phase_id) { 
    record_time(1, phase_id); 
    if(this->history.count(phase_id)==0){
      this->history[phase_id]=get_time_diff(phase_id);
    }
    else{
      this->history[phase_id]+=get_time_diff(phase_id);
    }
  }

  double get_step_time(int phase_id) {
    double t = get_time(timing[phase_id]);
    return t;
  }


  void print_report(std::string dev, uint32_t num_candidates,
                    uint32_t num_outputs, std::ostream &out_file = std::cout) {

    out_file << "MainStep,Step,SubStep,Time(sec),Device" << std::endl;

    double t=0.0;

    if(this->is_cluster){
      t = mill_to_sec(history[cluster::total]);
      out_file << "Total Cluster time,\t,\t," << t << ","<< std::endl;

      t = mill_to_sec(history[cluster::init]);
      out_file << "\t,Initialization,\t," << t << ","<< std::endl;

      t = mill_to_sec(history[cluster::onejoin]);
      out_file << "\t,OneJoin*,\t," << t << ","<< std::endl;

      t = mill_to_sec(history[cluster::create_indexes]);
      out_file << "\t,Create indexes,\t," << t << ","<< std::endl;

      t = mill_to_sec(history[cluster::sort]);
      out_file << "\t,Sorting,\t," << t << ","<< std::endl;

      t = mill_to_sec(history[cluster::onejoin]);
      out_file << "\t,DBSCAN,\t," << t << ","<< std::endl;

      t = mill_to_sec(history[cluster::consensus]);
      out_file << "\t,Consensus,\t," << t << ","<< std::endl <<std::endl;
    }

    t = mill_to_sec(history[init::total]);
    out_file << "Initialization,\t,\t," << t << ","<< std::endl;

    t = mill_to_sec(history[init::init_data]);
    out_file << "\t,Init Dataset,\t," << t << "," << std::endl;

    t = mill_to_sec(history[init::init_lsh]);
    out_file << "\t,Init LSH bits,\t," << t << ","<< std::endl;

    t = mill_to_sec(history[init::rev_lsh]);
    out_file << "\t,Init Rev LSH array,\t," << t << ","<< std::endl;

    t = mill_to_sec(history[embed::total]);
    out_file << "Embedding,\t,\t," << t << "," << dev << std::endl;

    t = mill_to_sec(history[embed::alloc]);
    out_file << "\t,USM allocation,\t," << t << ","<< std::endl;

    t = mill_to_sec(history[embed::rand_str]);
    out_file << "\t,Random string generation,\t," << t << ","<< std::endl;

    t = mill_to_sec(history[embed::measure]);
    out_file << "\t,Measurement,\t," << t << ","<< std::endl;

    t = mill_to_sec(history[embed::compute]);
    out_file << "\t,Computing,\t," << t << ","<< std::endl;

    t = mill_to_sec(history[lsh::total]);
    out_file << "LSH time,\t,\t," << t << ","<< std::endl;

    t = mill_to_sec(history[buckets::total]);
    out_file << "\t,Create Buckets,\t," << t << "," << dev << std::endl;

    t = mill_to_sec(history[buckets::allocation]);
    out_file << "\t,\t,Buckets Allocation," << t << "," << dev << std::endl;

    t = mill_to_sec(history[buckets::measure]);
    out_file << "\t,\t,Measurement," << t << ","<< std::endl;

    t = mill_to_sec(history[buckets::compute]);
    out_file << "\t,\t,Computing," << t << ","<< std::endl;

    t = mill_to_sec(history[buckets::sort]);
    out_file << "\t,\t,Sort Buckets," << t << ","<< std::endl;

    t = mill_to_sec(history[cand_init::total]);
    out_file << "\t,Candidate Initialization,\t," << t << ","<< std::endl;

    t = mill_to_sec(history[cand_init::comp_buck_delim]);
    out_file << "\t,\t,Compute buckets delimiter," << t << ","<< std::endl;

    t = mill_to_sec(history[cand_init::filter_buck_delim]);
    out_file << "\t,\t,Filter one element buckets," << t << ","<< std::endl;

    t = mill_to_sec(history[cand_init::resize]);
    out_file << "\t,\t,Allocate candidate vector," << t << ","<< std::endl;

    t = mill_to_sec(history[cand_init::scan_cand]);
    out_file << "\t,\t,Scan cand vector (write i and j)," << t << ","<< std::endl;

    t = mill_to_sec(history[cand::total]);
    out_file << "\t,Generate Candidate,\t," << t << "," << dev << std::endl;

    t = mill_to_sec(history[cand::measure]);
    out_file << "\t,\t,Measurement," << t << ","<< std::endl;

    t = mill_to_sec(history[cand::compute]);
    out_file << "\t,\t,Computing," << t << ","<< std::endl;

    t = mill_to_sec(history[cand_proc::total]);
    out_file << "\t,Candidates processing,\t," << t << ","<< std::endl;

    t = mill_to_sec(history[cand_proc::rem_cand]);
    out_file << "\t,\t,Remove candidates," << t << ","<< std::endl;

    t = mill_to_sec(history[cand_proc::sort_cand]);
    out_file << "\t,\t,Sort candidates," << t << ","<< std::endl;

    t = mill_to_sec(history[cand_proc::count_freq]);
    out_file << "\t,\t,Counting frequencies," << t << ","<< std::endl;

    t = mill_to_sec(history[cand_proc::rem_dup]);
    out_file << "\t,\t,Remove duplicates," << t << ","<< std::endl;

    t = mill_to_sec(history[cand_proc::sort_cand_to_verify]);
    out_file << "\t,\t,Sorting candidates to verify," << t << ","<< std::endl;

    t = mill_to_sec(history[cand_proc::filter_low_freq]);
    out_file << "\t,\t,Remove low frequencies candidates," << t << ","<< std::endl;

    t = mill_to_sec(history[cand_proc::make_uniq]);
    out_file << "\t,\t,Removing duplicates," << t << ","<< std::endl;

    t = mill_to_sec(history[edit_dist::total]);
    out_file << "Edit Distance,\t,\t," << t << ","<< std::endl;

    t = mill_to_sec(history[total_alg::total]);
    out_file << "Total OneJoin time,\t,\t," << t << ","<< std::endl;

    out_file << "Number candidates,\t" << num_candidates << ",\t,\t,"<< std::endl;
    out_file << "Number output,\t" << num_outputs << ",\t,\t,"<< std::endl;
  }

  void print_summary(uint32_t num_candidates, uint32_t num_outputs) {

    std::cout << "\n\n\nSummary:" << std::endl << std::endl;

    double t = mill_to_sec(history[init::total]);
    std::cout << "Time init input data: " << t << std::endl;

    t = mill_to_sec(history[embed::total]);
    std::cout << "Time PARALLEL embedding data:\t" << t << "sec" << std::endl;

    t = mill_to_sec(history[buckets::total]);
    std::cout << "Time PARALLEL buckets generation:\t" << t << "sec"
              << std::endl;

    t = mill_to_sec(history[buckets::sort]);
    std::cout << "Time buckets sorting:\t" << t << "sec" << std::endl;

    t = mill_to_sec(history[cand_init::total]);
    std::cout << "Time candidate initialization:\t" << t << "sec" << std::endl;

    t = mill_to_sec(history[cand::total]);
    std::cout << "Time PARALLEL candidates generation:\t" << t << "sec"
              << std::endl;

    t = mill_to_sec(history[cand_proc::sort_cand]);
    std::cout << "Time candidates sorting (within cand-processing):\t" << t
              << "sec" << std::endl;

    t = mill_to_sec(history[edit_dist::total]);
    std::cout << "Time compute edit distance:\t" << t << "sec" << std::endl;

    t = mill_to_sec(history[lsh::total]);
    std::cout << "Total time parallel join:\t" << t << "sec" << std::endl;

    t = mill_to_sec(history[total_alg::total]);
    std::cout << "Total elapsed time :\t" << t << "sec" << std::endl;
    std::cout << "Number of candidates verified: " << num_candidates
              << std::endl;
    std::cout << "Number of output pairs: " << num_outputs << std::endl;
  }

private:
  map<int, timeinterval_t> timing;
  map<int,long> history;

  timepoint_t t = std::chrono::system_clock::now();
  bool is_cluster;

  double get_time(timeinterval_t time) {
    long d = std::chrono::duration_cast<std::chrono::milliseconds>(time.second -
                                                                   time.first)
                 .count();
    double t = (double)d / 1000.0;
    return t;
  }

  double mill_to_sec(long val){
    return (double) val/1000.0;
  }

  long get_time_diff(int phase_id) {
    long diff = std::chrono::duration_cast<std::chrono::milliseconds>(timing[phase_id].second -
                                                                   timing[phase_id].first)
                 .count();
    return diff;
  }

  void record_time(int i, int phase_id) {
    if (i == 0) {
      timing[phase_id].first = std::chrono::system_clock::now();
    } else if (i == 1) {
      timing[phase_id].second = std::chrono::system_clock::now();
    }
  }
};
