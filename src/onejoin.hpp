#ifndef ONEJOIN_H
#define ONEJOIN_H
#include <CL/sycl.hpp>
#include <ctime>
#include <unistd.h>
#include <vector>
#include <list>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <string.h>
#include <algorithm>
#include <tuple>
#include <numeric>
#include <map>
#include <list>
#include <set>
#include <chrono>
#include <limits>
#include <cmath>
#include <cstdint>
#include <thread>
#include <mutex>
#include <atomic>
#include <memory>
//#include <ittnotify.h>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>
#include "tbb/parallel_sort.h"
#include <tbb/concurrent_vector.h>
#include <exception>
#include <optional>
#include "Time.cpp"
#include <boost/program_options.hpp>
#define BOOST_LOG_DYN_LINK 1
#define BOOST_ALL_DYN_LINK 1
#include <boost/log/trivial.hpp>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include "constants.hpp"

using namespace constants;
using namespace std;
using idpair=std::tuple<int, int>;
constexpr size_t max_buffer_size=0xFFFFFFFF;

enum {cpu=0,gpu,both};
namespace alg { enum {join=1,cluster}; };

struct candidate_t {
	uint32_t idx_str1;
	uint32_t len_diff;
	uint32_t idx_str2;
	uint16_t rep12_eq_bit;
	candidate_t(): idx_str1(0), len_diff(0), idx_str2(0), rep12_eq_bit(0) {}
	candidate_t(uint32_t idx_str1, uint32_t len_diff, uint32_t idx_str2, uint8_t rep12_eq_bit):
		idx_str1(idx_str1), len_diff(len_diff), idx_str2(idx_str2), rep12_eq_bit(rep12_eq_bit) {}
	bool operator<(const candidate_t& rhs) const {return ( (idx_str1 < rhs.idx_str1)
			|| (idx_str1 == rhs.idx_str1 && idx_str2 < rhs.idx_str2)
			|| (idx_str1 == rhs.idx_str1 && idx_str2 == rhs.idx_str2 && rep12_eq_bit < rhs.rep12_eq_bit)); }

	bool operator!=(const candidate_t& rhs) const {
		return !(idx_str1 == rhs.idx_str1 && idx_str2 == rhs.idx_str2 && len_diff == rhs.len_diff && rep12_eq_bit == rhs.rep12_eq_bit);
	};
	bool operator==(const candidate_t& rhs) const {
			return (idx_str1 == rhs.idx_str1 && idx_str2 == rhs.idx_str2 && len_diff == rhs.len_diff && rep12_eq_bit == rhs.rep12_eq_bit);
	};
};

struct buckets_t {
	uint32_t idx_rand_str;
	uint32_t idx_hash_func;
	uint32_t hash_id;
	uint32_t idx_str;
	uint32_t idx_rep;
	buckets_t():idx_rand_str(0), idx_hash_func(0), hash_id(0), idx_str(0), idx_rep(0){}
	buckets_t(uint32_t id_rand_string, uint32_t id_hash_func, uint32_t hash_id, uint32_t id_string, uint32_t id_rep ):
		idx_rand_str(id_rand_string), idx_hash_func(id_hash_func),
		hash_id(hash_id), idx_str(id_string), idx_rep(id_string){}
	bool operator<(const buckets_t& rhs) const {return ( (idx_rand_str < rhs.idx_rand_str)
		|| (idx_rand_str == rhs.idx_rand_str && idx_hash_func < rhs.idx_hash_func)
		|| (idx_rand_str == rhs.idx_rand_str && idx_hash_func == rhs.idx_hash_func && hash_id < rhs.hash_id)
		|| (idx_rand_str == rhs.idx_rand_str && idx_hash_func == rhs.idx_hash_func && hash_id == rhs.hash_id && idx_str < rhs.idx_str)
		|| (idx_rand_str == rhs.idx_rand_str && idx_hash_func == rhs.idx_hash_func && hash_id == rhs.hash_id && idx_str == rhs.idx_str && idx_rep < rhs.idx_rep) ); }
};

struct batch_hdr{
	size_t size;
	size_t offset;
	batch_hdr(size_t size, size_t offset): size(size), offset(offset){}
};

struct OutputValues{
	string dev;
	size_t num_candidates;
	size_t num_outputs;
	OutputValues():dev(""), num_candidates(0),num_outputs(0){}
};

#define NUMREPCHARS(len_output) (len_output * NUM_REP)
#define NUMSTRCHARS(len_output) (NUMREPCHARS(len_output) * NUM_STR)
#define ABSPOS(i,j,k,m,len_output) static_cast<unsigned int>(i * NUMSTRCHARS(len_output) + j * NUMREPCHARS(len_output) + k * len_output + m)
#define ABSPOS_P(j,t,d,len) static_cast<unsigned int>(j*NUM_CHAR*len +t*len+d)





void init_logging(bool debug=false);
int edit_distance(const char *x, const int x_len, const  char *y, const int y_len, int k);
void read_dataset(vector<string> &input_data, string filename);
void print_configuration(int batch_size,int n_batches, size_t len_output, size_t num_input_strings, int countfilter, int samplingrange);
std::string getReportFileName(int device, size_t batch_size);
void save_report(int device, size_t batch_size, string dataset_name, OutputValues &output_val, Time &timer);


vector<idpair> onejoin(vector<string> &input_data, size_t batch_size, int device, uint32_t new_samplingrange, uint32_t new_countfilter, Time &timer, OutputValues &output_val, int alg, int num_thr_val=0, string dataset_name="");
void oneCluster(vector<string> &input_data, size_t batch_size, int device, uint32_t new_samplingrange, uint32_t new_countfilter, Time &timer, int nPts, string dataset_name);

#endif
