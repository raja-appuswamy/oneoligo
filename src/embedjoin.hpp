#ifndef EMBEDJOIN_H
#define EMBEDJOIN_H
#include <CL/sycl.hpp>
#include <ctime>
#include <unistd.h>
#include <vector>
#include <list>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <algorithm>
#include <tuple>
#include <numeric>
#include <map>
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
#include <exception>
#include <optional>
#include "Time.cpp"

using namespace std;
using buckets_t = std::tuple<int,int,int,int,int>;
using candidate_t = std::tuple<uint32_t,uint32_t,uint32_t,uint8_t>;
using idpair=std::tuple<int, int>;

#define PRINT_EACH_STEP 0
#define NUMREPCHARS(len_output) (len_output * NUM_REP)
#define NUMSTRCHARS(len_output) (NUMREPCHARS(len_output) * NUM_STR)
#define ABSPOS(i,j,k,m,len_output) static_cast<unsigned int>(i * NUMSTRCHARS(len_output) + j * NUMREPCHARS(len_output) + k * len_output + m)
#define ABSPOS_P(j,t,d,len) static_cast<unsigned int>(j*NUM_CHAR*len +t*len+d)

// Default parameters for GEN DATASET: 150 7 16 12 4 5000 0 50 1

#ifndef NUM_STR
	#define NUM_STR 7 // r: number of CGK-embedding for each input string
#endif

#ifndef NUM_HASH
	#define NUM_HASH 16  //z: number of hash functions for each embedded string
#endif

#ifndef NUM_BITS
	#define NUM_BITS 12// m: number of bits in each hash function
#endif

#ifndef NUM_CHAR
	#define NUM_CHAR 4 //dictsize: alpha beta size of input strings, could be 4 for DNA dataset (ACGT); 26 for UNIREF dataset (A~Z); 37 for TREC dataset (A~Z,0~9,' ')
#endif

#ifndef NUM_STRING
	#define NUM_STRING 300000
#endif

#ifndef LEN_INPUT
	#define LEN_INPUT 5153
#endif

#ifndef ALLOUTPUTRESULT
	#define ALLOUTPUTRESULT 0
#endif

#ifndef SHIFT
	#define SHIFT 50
#endif

#ifndef M
	#define M 1000003 //size of hash table;
#endif

#ifndef K_INPUT
	#define K_INPUT 150 // edit distance threshold
#endif

#ifndef NUM_REP
	#define NUM_REP 3 // edit distance threshold
#endif

#ifndef PRINT_EMB
	#define PRINT_EMB 0
#endif

#ifndef PRINT_BUCK
	#define PRINT_BUCK 0
#endif

#ifndef PRINT_CAND
	#define PRINT_CAND 0
#endif

int edit_distance(const char *x, const int x_len, const  char *y, const int y_len, int k);
void read_dataset(vector<string> &input_data, string filename);
void print_oristrings( char *oristrings, vector<int> len );
void print_embedded( char **output, int len_output, int batch_size, std::string filename );
void print_buckets( vector<buckets_t> &buckets, std::string filename);
void print_candidate_pairs( vector<candidate_t> &candidates, std::string filename );
void print_configuration(int batch_size, int n_batches, size_t len_output, int countfilter, int samplingrange);
void onejoin(vector<string> &input_data, size_t batch_size, size_t n_batches, int device, uint32_t new_samplingrange, uint32_t new_countfilter, Time &timer, string dataset_name="");

#endif
