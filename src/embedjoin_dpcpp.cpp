#include <CL/sycl.hpp>
#include <dpstd/execution>
#include <dpstd/algorithm>
#include <dpstd/iterator>
#include "tbb/parallel_sort.h"
#include <exception>

#include <experimental/algorithm>

#include "embedjoin.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <chrono>
#include "mkl.h"
#include <cmath>
#include <cstdint>
#include <tbb/parallel_sort.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <memory>


using namespace std;
using namespace cl::sycl;
using namespace dpstd::execution;



#define PRINT_EACH_STEP 0

#define NUMREPCHARS(len_output) (len_output * NUM_REP)

#define NUMSTRCHARS(len_output) (NUMREPCHARS(len_output) * NUM_STR)

#define ABSPOS(i,j,k,m,len_output) static_cast<unsigned int>(i * NUMSTRCHARS(len_output) + j * NUMREPCHARS(len_output) + k * len_output + m)


#define ABSPOS_P(j,t,d,len) static_cast<unsigned int>(j*NUM_CHAR*len +t*len+d)




// Parameters: 150 7 16 12 4 5000 0 50 1

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

//embedrange: the length of truncation, recommended to be the average length of strings (you could try smaller values to further save the embedding time)

//#define T 1 //T:

//Dataset 80K
//#define LEN_INPUT        5110
//#define NUM_STRING        80001

//Dataset 50K



int samplingrange=5000; //the maximum digit to embed, the range to sample

int countfilter=1;// Number of required matches (>T) for a pair of substrings to be considered as candidate




std::string filename="reducedGen320ks.txt";

// OUTPUT FILENAME
static std::string file_embed_strings="embedded.txt";
static std::string file_buckets="buckets.txt";
static std::string file_candidate_pairs="candidate_pairs.txt";
static std::string file_output="parallel_join_output.txt";


#if GPU
static std::string filename_report="parallel_join_report_GPU_"+filename;
#else
static std::string filename_report="parallel_join_report_CPU_"+filename;
#endif

typedef std::tuple<int, int> idpair;
typedef std::tuple<int, int, int> idthree;



class arrayWrapper{

public:
	size_t size;
	size_t offset; //In number of element

	arrayWrapper(size_t size, size_t offset){
		this->size=size;
		this->offset=offset;
	}



};



vector<int> indices;
//vector<vector<vector<string>>> embdata;// embedded strings

vector<idpair> outputs;

//TODO: Remove or manage better
vector<string> tmp_oridata;

std::mutex output;


void setuplsh(int (*hash_lsh)[NUM_BITS], vector<int> &a, vector<int> &lshnumber)
{
	//vector<vector<int>> inihash_lsh(NUM_HASH, vector<int>(NUM_BITS));
	//hash_lsh = inihash_lsh;

	for (int i = 0; i < NUM_HASH; i++)
	for (int j = 0; j < NUM_BITS; j++)
		hash_lsh[i][j] = rand() % (samplingrange);

	for (int i = 0; i < NUM_BITS; i++)
		a.push_back(rand() % (M - 1));


	//lshnumber.reserve(NUM_BITS*NUM_HASH);
	for (int i = 0; i < NUM_HASH; i++){
		for(int j=0; j < NUM_BITS; j++){
		//lshnumber.insert(lshnumber.end(), hash_lsh[i].begin(), hash_lsh[i].end());
			lshnumber.emplace_back(hash_lsh[i][j]);
		}
	}
	//std::cout<<"Size lshnumber: "<<lshnumber.size()<<std::endl;

	sort(lshnumber.begin(), lshnumber.end());
	lshnumber.erase(unique(lshnumber.begin(), lshnumber.end()), lshnumber.end());
	samplingrange = lshnumber[lshnumber.size() - 1];

	for (int i = 0; i < NUM_HASH; i++)
		for (int j = 0; j < NUM_BITS; j++)
			hash_lsh[i][j] = lower_bound(lshnumber.begin(), lshnumber.end(), hash_lsh[i][j]) - lshnumber.begin();

}


void print_configuration(int batch_size,int n_batches, int len_output){
	std::cout<<"\nParameter selected:"<<std::endl;
	std::cout<<"\tNum of strings:\t\t\t\t\t"<<NUM_STRING<<std::endl;
	std::cout<<"\tMax len of strings:\t\t\t\t"<<LEN_INPUT<<std::endl;
	std::cout<<"\tLen output:\t\t\t\t\t"<<len_output<<std::endl;
	std::cout<<"\tSamplingrange:\t\t\t\t\t"<<samplingrange<<std::endl;
	std::cout<<"\tNumber of Hash Function:\t\t\t"<<NUM_HASH<<std::endl;
	std::cout<<"\tNumber of Bits per hash function:\t\t"<<NUM_BITS<<std::endl;
	std::cout<<"\tNumber of Random Strings per input string:\t"<<NUM_STR<<std::endl;
	std::cout<<"\tNumber of Replication per input string:\t\t"<<NUM_REP<<std::endl;
	std::cout<<"\tK distance:\t\t\t\t\t"<<K_INPUT<<std::endl;
	std::cout<<"\tCount filter:\t\t\t\t\t"<<countfilter<<std::endl;
	std::cout<<"\tBatch size:\t\t\t\t\t"<<batch_size<<std::endl;
	std::cout<<"\tNumber of batches:\t\t\t\t"<<n_batches<<std::endl;
	std::cout<<std::endl;
}

void print_info(std::string message){

	std::mutex output;
	std::lock_guard<std::mutex> lock(output);
	cout<<message<<std::endl;

}



void readdata(vector<int> &len_oristrings, char (*oristrings)[LEN_INPUT], vector<string> &oridata_modified )
{
	ifstream  data(filename);
	string cell;
	int number_string = 0;
	auto start=std::chrono::system_clock::now();

	while (getline(data, cell))
	{
		//TODO: Decide what to do
		if(number_string==NUM_STRING){
			break;
		}
		number_string++;
		tmp_oridata.push_back(cell);
	}
	//compare c;

	auto end=std::chrono::system_clock::now();

	std::cout<<"\nReading in read function: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<std::endl;


//	print_info("\nReading in read function: "+to_string((float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000));


	for (int i = 0; i < tmp_oridata.size(); i++)
		indices.push_back(i);

	start=std::chrono::system_clock::now();

	tbb::parallel_sort(indices.begin(), indices.end(), [&](int i1, int i2) { return tmp_oridata[i1].size() <  tmp_oridata[i2].size();});
	tbb::parallel_sort(tmp_oridata.begin(), tmp_oridata.end(), [&](auto s1,auto s2){return s1.size()<s2.size();});

	end=std::chrono::system_clock::now();

	std::cout<<"\nSorting in read function: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<std::endl;



	oridata_modified = tmp_oridata;		// append distinguisher at the end
	for (int j = 0; j < oridata_modified.size(); j++){
		for(int k = 0;k < 8; k++) oridata_modified[j].push_back(j>>(8*k));
	}

	 start=std::chrono::system_clock::now();


	for(int i=0; i<NUM_STRING; i++){
		memset(oristrings[i],0,LEN_INPUT);
		strncpy(oristrings[i],tmp_oridata[i].c_str(),std::min(static_cast<int>(tmp_oridata[i].size()),LEN_INPUT));
		len_oristrings.emplace_back(tmp_oridata[i].size());
	}

	 end=std::chrono::system_clock::now();

	std::cout<<"\nMemory op in read function: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<std::endl;
	std::cout<<std::endl;
}



void initialization( vector<int> &len_oristrings, char (*oristrings)[LEN_INPUT], vector<string> &oridata_modified, int (*hash_lsh)[NUM_BITS], vector<int> &a, vector<int> &lshnumber )
{



	//ORIGINAL VERSION
	readdata(len_oristrings, oristrings, oridata_modified);

//	fprintf(stderr, "Read:%s \n", filename.c_str());
//	fprintf(stderr, "Threshold:%d \n", K_INPUT);
//	fprintf(stderr, "Shift, Num of Rep: %d %d \n", SHIFT, NUM_REP);
//	if(countfilter)
//		fprintf(stderr, "Count filter: %d, %d\n", SHIFT, NUM_REP);
//	fprintf(stderr, "r,z,m: %d %d %d \n", NUM_STR, NUM_HASH, NUM_BITS);


	setuplsh(hash_lsh, a, lshnumber);

}

void inititalize_dictionary(int* dictionary){

	if(NUM_CHAR==4){
		dictionary['A']=0;
		dictionary['C']=1;
		dictionary['G']=2;
		dictionary['T']=3;

	}else if (NUM_CHAR == 26 || NUM_CHAR == 25){
		int j=0;
		for(int i=(int)'A'; i<=(int)'Z'; i++){
			dictionary[i]=j;
			j++;
		}

	}

}


void parallel_embedding_while_loop(queue &device_queue, int* len_oristrings, char *oristrings, char* embdata, uint32_t batch_size, vector<int> &lshnumber, int *p, int len_p, uint32_t len_output, int* dictionary){


	{
		std::lock_guard<std::mutex> lk(output);
		cout << "Task: Embedding Data";
		cout << "\tDevice: " << device_queue.get_device().get_info<info::device::name>() << std::endl;
	}
	unsigned int size_p=static_cast<unsigned int>(NUM_STR*NUM_CHAR*(samplingrange+1));

	buffer<int,1> buffer_p(p,range<1>{size_p});

	buffer<char, 2> buffer_oristrings(reinterpret_cast<char*>(oristrings),range<2>{batch_size,LEN_INPUT});

	buffer<int, 1> buffer_lshnumber(lshnumber.data(),range<1>{lshnumber.size()});

	unsigned int size_emb=static_cast<unsigned int>(batch_size*NUM_STR*NUM_REP*len_output);

	buffer<char, 1> buffer_output(embdata, range<1>{size_emb});

	buffer<int,1> buffer_dict(dictionary,range<1>{256});

	buffer<int,1> buffer_len_oristrings(len_oristrings,range<1>(batch_size));

	uint32_t samprange=samplingrange;
	buffer<uint32_t,1> buffer_samplingrange(&samprange,range<1>(1));

    buffer<uint32_t, 1> buffer_len_output(&len_output,range<1>{1});



	device_queue.submit([&](handler &cgh){


		  auto acc_oristrings = buffer_oristrings.get_access<access::mode::read>(cgh);
		  auto acc_lshnumber = buffer_lshnumber.get_access<access::mode::read,access::target::constant_buffer>(cgh);
		  auto acc_output = buffer_output.get_access<access::mode::write>(cgh);
		  auto acc_dict = buffer_dict.get_access<access::mode::read>(cgh);
		  auto acc_samplingrange = buffer_samplingrange.get_access<access::mode::read>(cgh);
		  auto acc_len_oristrings = buffer_len_oristrings.get_access<access::mode::read>(cgh);
		  auto acc_p=buffer_p.get_access<access::mode::read>(cgh);

          auto acc_len_output=buffer_len_output.get_access<access::mode::read>(cgh);


		  //Executing kernel
		  cgh.parallel_for<class EmbedString>(range<3>{batch_size,NUM_STR,NUM_REP}, [=](id<3> index){


			  int id = index[0];
			  int l=index[1];
			  int k=index[2];

			  int partdigit = 0;
			  int size = acc_len_oristrings[id];

			  int r=0;

			  int len_out=acc_lshnumber.get_range()[0];

			  int i = SHIFT*k;

			  int len=acc_samplingrange[0]+1;



			  for (int j = 0; i < size && j <= acc_samplingrange[0]; i++){

				  char s = acc_oristrings[id][i];

				  r=acc_dict[s];

				  j += ( acc_p[ABSPOS_P(l,r,j,len)] + 1 );

				  while (partdigit < acc_lshnumber.get_range()[0] && j > acc_lshnumber[partdigit]){


					  acc_output[ABSPOS(id,l,k,partdigit, acc_len_output[0])]=s;
					  partdigit++;

				 }
			  }

		  });

		});

	// ensure any asynchronous exceptions caught are handled before proceeding
	device_queue.wait_and_throw();
}




void parallel_embeddingUSM(queue &device_queue, int* len_oristrings, char *oristrings, char** embdata, unsigned int batch_size, vector<int> &lshnumber, int *p, int len_p, uint32_t len_output, int* dictionary){


    std::cout<<std::endl;
	std::cout << "Device: " << device_queue.get_device().get_info<info::device::name>() << std::endl;

	unsigned int size_p=NUM_STR*NUM_CHAR*(samplingrange+1);

	buffer<int,1> buffer_p(reinterpret_cast<int*>(p),range<1>{size_p});

	buffer<char, 2> buffer_oristrings(reinterpret_cast<char*>(oristrings),range<2>{NUM_STRING,LEN_INPUT});

	buffer<int, 1> buffer_lshnumber(lshnumber.data(),range<1>{lshnumber.size()});


	buffer<int,1> buffer_dict(dictionary,range<1>{256});
	buffer<int,1> buffer_len_oristrings(len_oristrings,range<1>(NUM_STRING));

	uint32_t samprange=samplingrange;
	buffer<uint32_t,1> buffer_samplingrange(&samprange,range<1>(1));
    
    buffer<unsigned int, 1> buffer_batch_size(&batch_size,range<1>{1});
    
    buffer<uint32_t, 1> buffer_len_output(&len_output,range<1>{1});



	cout << "\tTask: Embedding Data"<< std::endl;


	device_queue.submit([&](handler &cgh){


		  auto acc_oristrings = buffer_oristrings.get_access<access::mode::read>(cgh);
		  auto acc_lshnumber = buffer_lshnumber.get_access<access::mode::read>(cgh);

		  auto acc_dict = buffer_dict.get_access<access::mode::read>(cgh);
		  auto acc_samplingrange = buffer_samplingrange.get_access<access::mode::read>(cgh);
		  auto acc_len_oristrings = buffer_len_oristrings.get_access<access::mode::read>(cgh);
		  auto acc_p=buffer_p.get_access<access::mode::read>(cgh);
		
          auto acc_batch_size=buffer_batch_size.get_access<access::mode::read>(cgh);

          auto acc_len_output=buffer_len_output.get_access<access::mode::read>(cgh);

        
		  //Executing kernel
		  cgh.parallel_for<class EmbedString>(range<3>{NUM_STRING,NUM_STR,NUM_REP}, [=](id<3> index){


			  int id = index[0];
			  int l=index[1];
			  int k=index[2];

			  int partdigit = 0;
			  int size = acc_len_oristrings[id];

			  int r=0;

			  int len_out=acc_lshnumber.get_range()[0];

			  int i = SHIFT*k;

			  int len=acc_samplingrange[0]+1;

			  for (int j = 0; i < size && j <= acc_samplingrange[0]; j++){

				  char s = acc_oristrings[id][i];

				  r=acc_dict[s];

				  i += (acc_p[ABSPOS_P(l,r,j,len)] ? 0:1);

				  embdata[id/acc_batch_size[0]][ABSPOS(id%acc_batch_size[0],l,k,partdigit,acc_len_output[0])]=s;
				  partdigit++;

			  }

		  });

		});

	// ensure any asynchronous exceptions caught are handled before proceeding
	device_queue.wait_and_throw();
}



void parallel_embedding_batched(queue &device_queue, int* len_oristrings, char *oristrings, char* embdata, unsigned int batch_size, vector<int> &lshnumber, int *p, int len_p, uint32_t len_output, int* dictionary){


	{
		std::lock_guard<std::mutex> lk(output);
		cout << "\tTask: Embedding Data\t";
		std::cout << "Device: " << device_queue.get_device().get_info<info::device::name>() << std::endl;
		std::cout<<"\n\tLen output: "<<len_output<<std::endl;
	}

	unsigned int size_p=NUM_STR*NUM_CHAR*(samplingrange+1);

	buffer<int,1> buffer_p(reinterpret_cast<int*>(p),range<1>{size_p}, {property::buffer::use_host_ptr()});

	buffer<char, 2> buffer_oristrings(reinterpret_cast<char*>(oristrings),range<2>{batch_size,LEN_INPUT}, {property::buffer::use_host_ptr()});

	buffer<int, 1> buffer_lshnumber(lshnumber.data(),range<1>{lshnumber.size()});

	buffer<char, 1> buffer_embdata(embdata, range<1>{static_cast<unsigned int>(batch_size*NUM_STR*NUM_REP*len_output)}, {property::buffer::use_host_ptr()});

	buffer<int,1> buffer_dict(dictionary,range<1>{256});

	buffer<int,1> buffer_len_oristrings(len_oristrings,range<1>(batch_size), {property::buffer::use_host_ptr()});

	buffer<uint32_t,1> buffer_len_output(&len_output,range<1>{1});

	uint32_t samprange=samplingrange;
	buffer<uint32_t,1> buffer_samplingrange(&samprange,range<1>(1));

	device_queue.submit([&](handler &cgh){

		auto acc_oristrings = buffer_oristrings.get_access<access::mode::read>(cgh);
		auto acc_lshnumber = buffer_lshnumber.get_access<access::mode::read>(cgh);
		auto acc_embdata = buffer_embdata.get_access<access::mode::write>(cgh);
		auto acc_dict = buffer_dict.get_access<access::mode::read>(cgh);
		auto acc_samplingrange = buffer_samplingrange.get_access<access::mode::read>(cgh);
		auto acc_len_oristrings = buffer_len_oristrings.get_access<access::mode::read>(cgh);
		auto acc_p=buffer_p.get_access<access::mode::read>(cgh);

		auto acc_len_output=buffer_len_output.get_access<access::mode::read>(cgh);


			  //Executing kernel
		cgh.parallel_for<class EmbedString>(range<3>{batch_size,NUM_STR,NUM_REP}, [=](id<3> index){


			int id = index[0];
			int l=index[1];
			int k=index[2];

			int partdigit = 0;
			int size = acc_len_oristrings[id];

			int r=0;

			int len_out=acc_lshnumber.get_range()[0];

			int i = SHIFT*k;

			int len=acc_samplingrange[0]+1;

			for (int j = 0; i < size && j <= acc_samplingrange[0]; j++){

				char s = acc_oristrings[id][i];

				r=acc_dict[s];

				i += (acc_p[ABSPOS_P(l,r,j,len)] ? 0:1);

				acc_embdata[ABSPOS(id,l,k,partdigit, acc_len_output[0])]=s;
				partdigit++;

			}
		});

		});

	// ensure any asynchronous exceptions caught are handled before proceeding
	device_queue.wait_and_throw();




}



void create_bucket_2dev(queue &device_queue, char **embdata, tuple<int,int,int,int,int> *buckets, unsigned int batch_size, unsigned int split_size, unsigned int split_offset, int* hash_lsh, vector<int> &a, vector<int> &lshnumber, uint32_t len_output){

	int dictionary[256]={0};
	inititalize_dictionary(dictionary);

    std::cout<<std::endl;

	cout << "\tTask: Buckets Generation\t";
	std::cout << "Device: " << device_queue.get_device().get_info<info::device::name>() << std::endl;

    {

//	buffer<char, 1> buffer_embeddata(embdata, range<1>{static_cast<unsigned int>(batch_size*NUM_STR*NUM_REP*LEN_OUTPUT)});

	buffer<tuple<int,int,int,int,int>> buffer_buckets(buckets, range<1>{split_size*NUM_STR*NUM_HASH*NUM_REP}, {property::buffer::use_host_ptr()});
	buffer<int, 1> buffer_a(a.data(),range<1>{a.size()});
	buffer<int,2> buffer_hash_lsh(hash_lsh,range<2>{NUM_HASH,NUM_BITS});
	buffer<int,1> buffer_dict(dictionary,range<1>{256});
	buffer<int, 1> buffer_lshnumber(lshnumber.data(),range<1>{lshnumber.size()});

	buffer<unsigned int, 1> buffer_batch_size(&batch_size,range<1>{1});

    buffer<uint32_t, 1> buffer_len_output(&len_output,range<1>{1});
    buffer<uint32_t, 1> buffer_split_offset(&split_offset,range<1>{1});


    print_info("\n\tSplit size: "+to_string(split_size)+"\n\tOffset: "+to_string(split_offset));



	device_queue.submit([&](handler &cgh){

		//Executing kernel

		auto acc_buckets = buffer_buckets.get_access<access::mode::write>(cgh);

		auto acc_dict = buffer_dict.get_access<access::mode::read>(cgh);

		auto acc_hash_lsh = buffer_hash_lsh.get_access<access::mode::read>(cgh);

		auto acc_a = buffer_a.get_access<access::mode::read>(cgh);

		auto acc_lshnumber = buffer_lshnumber.get_access<access::mode::read>(cgh);

		auto acc_batch_size=buffer_batch_size.get_access<access::mode::read>(cgh);

		auto acc_len_output=buffer_len_output.get_access<access::mode::read>(cgh);

        auto acc_split_offset=buffer_split_offset.get_access<access::mode::read>(cgh);


		cgh.parallel_for<class CreateBuckets>(range<3>{split_size,NUM_STR,NUM_REP}, [=](item<3> index){

			int i=index[0]+acc_split_offset[0];
			int t=index[1];
			int q=index[2];


			int output_position;


			size_t linear_id=index.get_linear_id();


			int id=0;
			char dict_index=0;
			int id_mod=0;
			int digit=-1;


			for(int k=0; k<NUM_HASH; k++){

				id = 0;
				id_mod=0;

				for (int j = 0; j < NUM_BITS; j++){

					// if ( acc_hash_lsh[0][j]<acc_embeddata[index].size() ) {
					digit=acc_lshnumber[acc_hash_lsh[k][j]];
					dict_index=embdata[(int)(i/acc_batch_size[0])][ABSPOS(i%acc_batch_size[0],t,q,digit,acc_len_output[0])];
					id += (acc_dict[dict_index]) * acc_a[j];
					// }

				}

				id_mod=id % M;

				output_position=k+linear_id*NUM_HASH;

				get<0>(acc_buckets[output_position])=t;
				get<1>(acc_buckets[output_position])=k;
				get<2>(acc_buckets[output_position])=id_mod;
				get<3>(acc_buckets[output_position])=i;
				get<4>(acc_buckets[output_position])=q;
			}

		});

	});

	// ensure any asynchronous exceptions caught are handled before proceeding
	device_queue.wait_and_throw();
	}

}


void create_buckets_2dev_wrapper(vector<queue> &queues, char **embdata, vector<tuple<int,int,int,int,int>> &buckets, uint32_t n_batches, uint32_t batch_size, int* hash_lsh, vector<int> &a, vector<int> &lshnumber, uint32_t len_output){


	std::cout<< "\nSelected: Create buckets - batched"<<std::endl;



	int num_dev=queues.size();

	int split_size=n_batches/num_dev*batch_size;

	vector<std::thread> threads;

//	std::atomic<int> n(0);

	int n=0;

	for(auto &q:queues){

		threads.emplace_back(thread([&, n](){



			//int local_n=n.fetch_add(1);
			int offset=n*split_size;

			create_bucket_2dev(q, embdata, buckets.data()+offset*NUM_REP*NUM_HASH*NUM_STR, batch_size, split_size, offset,(int*)hash_lsh, a, lshnumber, len_output);

		}));

		n++;
	}



	for(auto &t:threads){
		if(t.joinable()){

			t.join();
		}
	}

}


// To use when call embedding WITH while loop.
void create_bucket_without_lshnumber_offset(queue &device_queue, char **embdata, tuple<int,int,int,int,int> *buckets, unsigned int batch_size, unsigned int split_size, unsigned int split_offset, int* hash_lsh, vector<int> &a, vector<int> &lshnumber, uint32_t len_output){

	int dictionary[256]={0};
	inititalize_dictionary(dictionary);

	{
		std::lock_guard<std::mutex> lock(output);
		cout << "\n\tTask: Buckets Generation\t";
		std::cout << "Device: " << device_queue.get_device().get_info<info::device::name>() << std::endl;
	}

    {

		buffer<tuple<int,int,int,int,int>> buffer_buckets(buckets,range<1>{split_size*NUM_STR*NUM_HASH*NUM_REP}, {property::buffer::use_host_ptr()}); // Wrong dimension
		buffer<int, 1> buffer_a(a.data(),range<1>{a.size()});
		buffer<int,2> buffer_hash_lsh(hash_lsh,range<2>{NUM_HASH,NUM_BITS});
		buffer<int,1> buffer_dict(dictionary,range<1>{256});


		buffer<unsigned int, 1> buffer_batch_size(&batch_size,range<1>{1});

		buffer<uint32_t, 1> buffer_len_output(&len_output,range<1>{1});

	    buffer<uint32_t, 1> buffer_split_offset(&split_offset,range<1>{1});


		device_queue.submit([&](handler &cgh){


		//Executing kernel

//		auto acc_embeddata = buffer_embeddata.get_access<access::mode::read>(cgh);

			auto acc_buckets = buffer_buckets.get_access<access::mode::write>(cgh);
			auto acc_dict = buffer_dict.get_access<access::mode::read>(cgh);
			auto acc_hash_lsh = buffer_hash_lsh.get_access<access::mode::read>(cgh);

			auto acc_a = buffer_a.get_access<access::mode::read>(cgh);

			auto acc_batch_size=buffer_batch_size.get_access<access::mode::read>(cgh);

			auto acc_len_output=buffer_len_output.get_access<access::mode::read>(cgh);

	        auto acc_split_offset=buffer_split_offset.get_access<access::mode::read>(cgh);


			cgh.parallel_for<class CreateBuckets>(range<3>{split_size,NUM_STR,NUM_REP}, [=](item<3> index){

				int i=index[0]+acc_split_offset[0];
				int t=index[1];
				int q=index[2];

				int output_position;

				size_t linear_id=index.get_linear_id();

				int id=0;
				char dict_index=0;
				int id_mod=0;
				int digit=-1;

				for(int k=0; k<NUM_HASH; k++){

					id = 0;
					id_mod=0;

					for (int j = 0; j < NUM_BITS; j++){

						//if ( acc_hash_lsh[0][j]<acc_embeddata[index].size() ) {
						digit=acc_hash_lsh[k][j];
						dict_index=embdata[(int)i/acc_batch_size[0]][ABSPOS(i%acc_batch_size[0],t,q,digit,acc_len_output[0])];
						id += (acc_dict[dict_index]) * acc_a[j];
						//}
					}
					id_mod=id % M;

					output_position=k+linear_id*NUM_HASH;

					get<0>(acc_buckets[output_position])=t;
					get<1>(acc_buckets[output_position])=k;
					get<2>(acc_buckets[output_position])=id_mod;
					get<3>(acc_buckets[output_position])=i;
					get<4>(acc_buckets[output_position])=q;
				}

			});

		});

	// ensure any asynchronous exceptions caught are handled before proceeding
		device_queue.wait_and_throw();
	}

}


void create_buckets_without_lshnumber_offset_2dev_wrapper(vector<queue> &queues, char **embdata, vector<tuple<int,int,int,int,int>> &buckets, uint32_t n_batches, uint32_t batch_size, int* hash_lsh, vector<int> &a, vector<int> &lshnumber, uint32_t len_output){

	std::cout<< "Selected: Create buckets - without lshnumber offset"<<std::endl;



	int num_dev=queues.size();

	int split_size=n_batches/num_dev*batch_size;

	vector<std::thread> threads;



	int n=0;

	for(auto &q:queues){

		threads.emplace_back(thread([&, n](){


			int offset=n*split_size;

			create_bucket_without_lshnumber_offset(q, embdata, buckets.data()+offset*NUM_REP*NUM_HASH*NUM_STR, batch_size, split_size, offset,(int*)hash_lsh, a, lshnumber, len_output);

		}));

		n++;
	}



	for(auto &t:threads){
		if(t.joinable()){

			t.join();
		}
	}

}


void generate_candidates_2dev(queue &device_queue, vector<int> &len, char* oristrings, char **embdata, tuple<int,int,int,int,int> *buckets, unsigned int buckets_size, unsigned int batch_size, tuple<int,int> *bucket_delimiter, unsigned int bucket_delimiter_size, std::tuple<int,int,int,int,int,int> *candidate, unsigned int candidate_size, int *candidates_start, unsigned int candidates_start_size, int *local_hash_lsh, vector<int> &lshnumber, uint32_t len_output){


    {//Scope for lock_guard
    	std::lock_guard<std::mutex> lk(output);
    	std::cout<<std::endl;
    	cout << "\tTask: Candidate Pairs Generation\t";
    	cout << "Device: " << device_queue.get_device().get_info<info::device::name>() << std::endl;

    	cout << "\n\t\tCandidates size: "<<candidate_size << std::endl;
		cout << "\t\tBuckets size: "<<buckets_size << std::endl;
		cout << "\t\tCandidates start size: "<<candidates_start_size << std::endl<<std::endl;

		cout<<"\t\tBuckets[0]: "<<get<0>(buckets[0])<<std::endl;
		cout<<"\t\tBuckets["<<buckets_size-1<<"]: "<<get<0>(buckets[buckets_size-1])<<std::endl<<std::endl;

		cout<<"\t\tCandidates[0]: "<<get<0>(candidate[0])<<std::endl;
		cout<<"\t\tCandidates["<<candidate_size-1<<"]: "<<get<0>(candidate[candidate_size-1])<<std::endl;

		cout<<"\t\tBuckets_delimiter[0]: "<<get<0>(bucket_delimiter[0])<<std::endl;
		cout<<"\t\tBuckets_delimiter["<<bucket_delimiter_size-1<<"]: "<<get<0>(bucket_delimiter[bucket_delimiter_size-1])<<std::endl;


		cout<<"\t\tCandidate_start[0]: "<< candidates_start[0] << std::endl;
		cout<<"\t\tCandidate_start["<<candidates_start_size-1<<"]: "<< candidates_start[candidates_start_size-1] <<std::endl;

    }// End scope of lock_guard, lock released automatically

	{

		buffer<int, 1> buffer_lshnumber(lshnumber.data(),range<1>{lshnumber.size()}, {property::buffer::use_host_ptr()});

		buffer<char,2> buffer_oristrings(oristrings,range<2>{NUM_STRING,LEN_INPUT}, {property::buffer::use_host_ptr()});

		buffer<int, 1> buffer_candidate_start(candidates_start,range<1>{candidates_start_size}, {property::buffer::use_host_ptr()});

		buffer<tuple<int,int,int,int,int>> buffer_buckets(buckets,range<1>{buckets_size}, {property::buffer::use_host_ptr()});

		buffer<tuple<int,int>> buffer_delimiter(bucket_delimiter,range<1>{bucket_delimiter_size}, {property::buffer::use_host_ptr()});




		buffer<int, 2> buffer_hash_lsh(reinterpret_cast<int*>(local_hash_lsh),range<2>{NUM_HASH,NUM_BITS}, {property::buffer::use_host_ptr()});

		buffer<tuple<int,int,int,int,int,int>> buffer_candidates(candidate,range<1>{candidate_size}, {property::buffer::use_host_ptr()});

		buffer<int,1> buffer_len(len.data(),range<1>{len.size()}, {property::buffer::use_host_ptr()});

        buffer<unsigned int, 1> buffer_batch_size(&batch_size,range<1>{1});

        buffer<uint32_t, 1> buffer_len_output(&len_output,range<1>{1});
        
//      buffer<uint32_t, 1> buffer_offset(&offset,range<1>{1});

		device_queue.submit([&](handler &cgh){

			auto acc_delimiter = buffer_delimiter.get_access<access::mode::read>(cgh);
			auto acc_buckets = buffer_buckets.get_access<access::mode::read>(cgh);
			auto acc_oridata = buffer_oristrings.get_access<access::mode::read>(cgh);

//			auto acc_embdata = buffer_embeddata.get_access<access::mode::read>(cgh);

			auto acc_hash_lsh = buffer_hash_lsh.get_access<access::mode::read>(cgh);
			auto acc_candidate_start = buffer_candidate_start.get_access<access::mode::read>(cgh);
			auto acc_candidate = buffer_candidates.get_access<access::mode::write>(cgh);

			auto acc_len = buffer_len.get_access<access::mode::read>(cgh);

			auto acc_lshnumber = buffer_lshnumber.get_access<access::mode::read>(cgh);

            auto acc_batch_size=buffer_batch_size.get_access<access::mode::read>(cgh);

//            auto acc_offset=buffer_offset.get_access<access::mode::read>(cgh);

            auto acc_len_output=buffer_len_output.get_access<access::mode::read>(cgh);


            
			cgh.parallel_for<class GenerateCandidates>(range<1>{bucket_delimiter_size}, [=](item<1> index){

				int b=index[0];
				int begin=get<0>(acc_delimiter[b]);
				int size=get<1>(acc_delimiter[b]);
				int end_bucket=begin+size;
				bool found=true;

				int index_output=acc_candidate_start[b];


				for( int i=begin; i<end_bucket-1; i++ ) {

					int t1=get<0>(acc_buckets[i]);
					int k1=get<1>(acc_buckets[i]);
					int i1=get<3>(acc_buckets[i]);
					int q1=get<4>(acc_buckets[i]);

					for (int j = i + 1; j < end_bucket; j++) {

						int t2=get<0>(acc_buckets[j]);
						int k2=get<1>(acc_buckets[j]);
						int i2=get<3>(acc_buckets[j]);
						int q2=get<4>(acc_buckets[j]);


						get<0>(acc_candidate[index_output])=i1;
						get<1>(acc_candidate[index_output])=q1;
						get<2>(acc_candidate[index_output])=i2;
						get<3>(acc_candidate[index_output])=q2;
						get<4>(acc_candidate[index_output])=abs_diff(acc_len[i1], acc_len[i2]);

						int sum=0;
						uint8_t c1;
						uint8_t c2;

						for (int j = 0; j < NUM_BITS; j++){

							//TODO Handle better
							c1=embdata[i1/acc_batch_size[0]][ ABSPOS(i1%acc_batch_size[0],t1,q1,acc_lshnumber[acc_hash_lsh[k1][j]],acc_len_output[0]) ];
							c2=embdata[i2/acc_batch_size[0]][ ABSPOS(i2%acc_batch_size[0],t1,q2,acc_lshnumber[acc_hash_lsh[k1][j]],acc_len_output[0]) ];
							if(c1!=0 && c2!=0){
								sum+=abs_diff(c1,c2);
							}

						}

						get<5>(acc_candidate[index_output])=sum;
						index_output++;
//
					}
				}

			});

		});

		device_queue.wait_and_throw();

	}
}


void generate_candidates_2dev_wrapper(vector<queue>& queues, vector<int> &len, char* oristrings, char **embdata, vector<tuple<int,int,int,int,int>> &buckets, unsigned int batch_size, vector<tuple<int,int>> &bucket_delimiter, vector<std::tuple<int,int,int,int,int,int>>& candidate, vector<int> &candidates_start, int * local_hash_lsh, vector<int> &lshnumber, uint32_t len_output, vector<arrayWrapper> &partitionsBucketsDelimiter, vector<arrayWrapper> &partitionsCandStart, vector<arrayWrapper> &partitionsBuckets, vector<arrayWrapper> &partitionsCandidates){

	std::cout<<"\nSelected: Generate candidate"<<std::endl;
	vector<std::thread> threads;
	int p=0;

	for(auto &q : queues){

		threads.push_back(thread([&,p](){

			{
				std::lock_guard<std::mutex> lk(output);

				cout<<"\n\tPartition "<<p<<": \n\n";
				cout<<"\tBuckets: from 0 to "<<buckets.size()<<"\n";
				cout<<"\tBuckets part: from "<< partitionsBuckets[p].offset <<" to "<<partitionsBuckets[p].offset+partitionsBuckets[p].size<<"\n\n";

				cout<<"\tcandidates_start[partitionCandStart["<<p<<"].offset]: "<< candidates_start[partitionsCandStart[p].offset]<<" it should be 0"<<std::endl;

				cout<<"\tbuckestDelimiter[partitionsBucketsDelimiter["<<p<<"].size]: "<< get<0>(bucket_delimiter[partitionsBucketsDelimiter[p].size-1]) << std::endl;
				cout<<"\tbuckestDelimiter[partitionsBucketsDelimiter["<<p<<"].size+1]: "<< get<0>(bucket_delimiter[partitionsBucketsDelimiter[p].size+1]) << std::endl;


				cout<<"\tCandStart: "<<candidates_start.data()<<std::endl;
			}

			generate_candidates_2dev(q, len, oristrings, embdata, buckets.data()+partitionsBuckets[p].offset, partitionsBuckets[p].size, batch_size, bucket_delimiter.data()+partitionsBucketsDelimiter[p].offset, partitionsBucketsDelimiter[p].size, candidate.data()+partitionsCandidates[p].offset, partitionsCandidates[p].size, candidates_start.data()+partitionsCandStart[p].offset, partitionsCandStart[p].size, local_hash_lsh, lshnumber, len_output);

		}));

		p++;
	}


	for(auto &t:threads){
			if(t.joinable()){
				t.join();
			}
		}

	return;
}


void generate_candidates_without_lshnumber_offset(queue &device_queue, vector<int> &len, char* oristrings, char **embdata, tuple<int,int,int,int,int> *buckets, unsigned int buckets_size, unsigned int batch_size, tuple<int,int> *bucket_delimiter, unsigned int bucket_delimiter_size, std::tuple<int,int,int,int,int,int> *candidate, unsigned int candidate_size, int *candidates_start, unsigned int candidates_start_size, int *local_hash_lsh, vector<int> &lshnumber, uint32_t len_output){

	{
		std::lock_guard<std::mutex> lock(output);
		cout << "\n\tTask: Candidate Pairs Generation\t";
		std::cout << "Device: " << device_queue.get_device().get_info<info::device::name>() << std::endl;
	}
//	std::cout << "Max allocation size: " << device_queue.get_device().get_info<info::device::max_mem_alloc_size>() << std::endl;

	{

//		buffer<char,1> buffer_embeddata(reinterpret_cast<char*>(embdata), range<1>{static_cast<unsigned int>(batch_size*NUM_STR*NUM_REP*LEN_OUTPUT)});

		buffer<char,2> buffer_oristrings(oristrings,range<2>{NUM_STRING,LEN_INPUT});

		buffer<int, 1> buffer_candidate_start(candidates_start, range<1>{candidates_start_size});

		buffer<tuple<int,int,int,int,int>> buffer_buckets(buckets,range<1>{buckets_size});

		buffer<tuple<int,int>> buffer_delimiter(bucket_delimiter,range<1>{bucket_delimiter_size});

		cout << bucket_delimiter_size<< " "<< candidate_size<< std::endl;

		buffer<int, 2> buffer_hash_lsh(reinterpret_cast<int*>(local_hash_lsh),range<2>{NUM_HASH,NUM_BITS});

		buffer<tuple<int,int,int,int,int,int>> buffer_candidates(candidate,range<1>{candidate_size});

		buffer<int,1> buffer_len(len.data(),range<1>{len.size()});

        buffer<unsigned int, 1> buffer_batch_size(&batch_size,range<1>{1});

        buffer<uint32_t, 1> buffer_len_output(&len_output,range<1>{1});

		device_queue.submit([&](handler &cgh){

			auto acc_delimiter = buffer_delimiter.get_access<access::mode::read>(cgh);
			auto acc_buckets = buffer_buckets.get_access<access::mode::read>(cgh);
			auto acc_oridata = buffer_oristrings.get_access<access::mode::read>(cgh);

//			auto acc_embdata = buffer_embeddata.get_access<access::mode::read>(cgh);

			auto acc_hash_lsh = buffer_hash_lsh.get_access<access::mode::read>(cgh);
			auto acc_candidate_start = buffer_candidate_start.get_access<access::mode::read>(cgh);
			auto acc_candidate = buffer_candidates.get_access<access::mode::write>(cgh);

			auto acc_len = buffer_len.get_access<access::mode::read>(cgh);

            auto acc_batch_size=buffer_batch_size.get_access<access::mode::read>(cgh);

            auto acc_len_output=buffer_len_output.get_access<access::mode::read>(cgh);

			cgh.parallel_for<class GenerateCandidates>(range<1>{bucket_delimiter_size}, [=](item<1> index){

				int b=index[0];
				int begin=get<0>(acc_delimiter[b]);
				int size=get<1>(acc_delimiter[b]);
				int end_bucket=begin+size;
				bool found=true;

				int index_output=acc_candidate_start[b];


				for( int i=begin; i<end_bucket-1; i++ ) {

					int t1=get<0>(acc_buckets[i]);
					int k1=get<1>(acc_buckets[i]);
					int i1=get<3>(acc_buckets[i]);
					int q1=get<4>(acc_buckets[i]);

					for (int j = i + 1; j < end_bucket; j++) {

						int t2=get<0>(acc_buckets[j]);
						int k2=get<1>(acc_buckets[j]);
						int i2=get<3>(acc_buckets[j]);
						int q2=get<4>(acc_buckets[j]);


						get<0>(acc_candidate[index_output])=i1;
						get<1>(acc_candidate[index_output])=q1;
						get<2>(acc_candidate[index_output])=i2;
						get<3>(acc_candidate[index_output])=q2;
						get<4>(acc_candidate[index_output])=abs_diff(acc_len[i1], acc_len[i2]);

						int sum=0;
						uint8_t c1;
						uint8_t c2;

						for (int j = 0; j < NUM_BITS; j++){

							//TODO Handle better

							c1=embdata[i1/acc_batch_size[0]][ ABSPOS(i1%acc_batch_size[0],t1,q1,acc_hash_lsh[k1][j],acc_len_output[0]) ];
							c2=embdata[i2/acc_batch_size[0]][ ABSPOS(i2%acc_batch_size[0],t1,q2,acc_hash_lsh[k1][j],acc_len_output[0]) ];
							if(c1!=0 && c2!=0){
								sum+=abs_diff(c1,c2);
							}
						}

						get<5>(acc_candidate[index_output])=sum;
						index_output++;
//
					}
				}

			});

		});

		device_queue.wait_and_throw();

	}
}


void generate_candidates_without_lshnumber_offset_2dev_wrapper(vector<queue>& queues, vector<int> &len, char* oristrings, char **embdata, vector<tuple<int,int,int,int,int>> &buckets, unsigned int batch_size, vector<tuple<int,int>> &bucket_delimiter, vector<std::tuple<int,int,int,int,int,int>>& candidate, vector<int> &candidates_start, int * local_hash_lsh, vector<int> &lshnumber, uint32_t len_output, vector<arrayWrapper> &partitionsBucketsDelimiter, vector<arrayWrapper> &partitionsCandStart, vector<arrayWrapper> &partitionsBuckets, vector<arrayWrapper> &partitionsCandidates){

	cout << "Selected: Generate candidates - without lshnumber offset"<< std::endl;

	vector<std::thread> threads;
	int p=0;

	for(auto &q : queues){

		threads.push_back(thread([&,p](){

			{
				std::lock_guard<std::mutex> lk(output);

				cout<<"\n\tPartition "<<p<<": \n\n";
				cout<<"\tBuckets: from 0 to "<<buckets.size()<<"\n";
				cout<<"\tBuckets part: from "<< partitionsBuckets[p].offset <<" to "<<partitionsBuckets[p].offset+partitionsBuckets[p].size<<"\n\n";

				cout<<"\tcandidates_start[partitionCandStart["<<p<<"].offset]: "<< candidates_start[partitionsCandStart[p].offset]<<" it should be 0"<<std::endl;

				cout<<"\tbuckestDelimiter[partitionsBucketsDelimiter["<<p<<"].size]: "<< get<0>(bucket_delimiter[partitionsBucketsDelimiter[p].size-1]) << std::endl;
				cout<<"\tbuckestDelimiter[partitionsBucketsDelimiter["<<p<<"].size+1]: "<< get<0>(bucket_delimiter[partitionsBucketsDelimiter[p].size+1]) << std::endl;


				cout<<"\tCandStart: "<<candidates_start.data()<<std::endl;
			}

			generate_candidates_without_lshnumber_offset(q, len, oristrings, embdata, buckets.data()+partitionsBuckets[p].offset, partitionsBuckets[p].size, batch_size, bucket_delimiter.data()+partitionsBucketsDelimiter[p].offset, partitionsBucketsDelimiter[p].size, candidate.data()+partitionsCandidates[p].offset, partitionsCandidates[p].size, candidates_start.data()+partitionsCandStart[p].offset, partitionsCandStart[p].size, local_hash_lsh, lshnumber, len_output);

		}));

		p++;

	}


	for(auto &t:threads){
			if(t.joinable()){
				t.join();
			}
		}

	return;
}

void generate_random_string(int* p, int len_p){

	for (int j = 0; j < NUM_STR; j++) {

		for (int t = 0; t < NUM_CHAR; t++){

			for (int d = 0; d < samplingrange + 1; d++){

				p[ABSPOS_P(j,t,d,len_p)]=1-rand() % 2;

			}

			for (int d = 0; d < samplingrange + 1; d++){

				if(p[ABSPOS_P(j,t,d,len_p)]==1){

					if(d>0 && p[ABSPOS_P(j,t,d-1,len_p)]==1){
						p[ABSPOS_P(j,t,d,len_p)] = p[ABSPOS_P(j,t,d-1,len_p)] - 1;
					}
					else{
						int next = d+1;
						while(next < samplingrange + 1 && p[ABSPOS_P(j,t,next,len_p)]==1){
							p[ABSPOS_P(j,t,d,len_p)]++;
							next++;
						}
					}
				}
			}
		}
	}

}

void print_output(std::string file_name=file_output)
{
	std::cout<<"Start saving results"<<std::endl;
    ofstream outFile;

    outFile.open(file_name, ios::out | ios::trunc);

    if (!outFile.is_open()) {
        std::cerr<<"Not possible to open file"<<std::endl;
        exit(-1);
    }
	sort(outputs.begin(), outputs.end());
	outputs.erase(unique(outputs.begin(), outputs.end()), outputs.end());


	if(ALLOUTPUTRESULT) {
        for (int i = 0; i < outputs.size(); i++) {

            outFile << indices[get<0>(outputs[i])] << " " << indices[get<1>(outputs[i])] << std::endl;
            outFile << tmp_oridata[get<0>(outputs[i])] << std::endl;
            outFile << tmp_oridata[get<1>(outputs[i])] << std::endl;
        }
    }

	std::cerr<< "Num of outputs: "<< int(outputs.size())<<std::endl;
	outputs.clear();
}



//TODO: Just to try
void print_oristrings( char *oristrings, vector<int> len )
{

	char **tmp;
	tmp=&oristrings;

	string filename_output="oristrings.txt";

	ofstream outFile;

	outFile.open(filename_output, ios::out | ios::trunc);

	if (outFile.is_open()) {
		for(int i=0; i<NUM_STRING; i++){

			for(int t=0; t<len[i]; t++){

				outFile<<tmp[i][t];

			}
			outFile<<std::endl;

			}
	}
}



void print_embedded( char **output, int len_output, int batch_size, std::string filename=file_embed_strings ){

	ofstream outFile;

	outFile.open(filename, ios::out | ios::trunc);

	if (outFile.is_open()) {
		for(int i=0; i<NUM_STRING; i++){
				for(int j=0; j<NUM_STR; j++ ){
					for(int k=0; k<NUM_REP; k++){
						for(int t=0; t<len_output; t++){

							if(output[i/batch_size][ABSPOS(i%batch_size,j,k,t,len_output)]==0){
								break;
							}
							outFile<<output[i/batch_size][ABSPOS(i%batch_size,j,k,t,len_output)];

							//outFile<<output[i][j][k][t];
						}
						outFile<<std::endl;

					}
				}
			}
	}
}

void print_buckets( vector<tuple<int,int,int,int,int>> &buckets, std::string filename=file_buckets){

	ofstream outFile;

	outFile.open(filename, ios::out | ios::trunc);

	if (outFile.is_open()) {
		for(int i=0; i<buckets.size(); i++){

			outFile<<get<0>(buckets[i])<<", "<<get<1>(buckets[i])<<", "<<get<2>(buckets[i])<<", "<<get<3>(buckets[i])<<", "<<get<4>(buckets[i])<<std::endl;

		}
	}
}

void print_candidate_pairs( vector<tuple<int,int,int,int,int,int>> &candidates, std::string filename=file_candidate_pairs ){

	ofstream outFile;

	outFile.open(filename, ios::out | ios::trunc);

	if (outFile.is_open()) {
		for(int i=0; i<candidates.size(); i++){

			outFile<</*get<4>(candidates[i])<<", "<<get<5>(candidates[i])<<", "<<*/get<0>(candidates[i])<<", "<<get<1>(candidates[i])<<", "<<get<2>(candidates[i])<<", "<<get<3>(candidates[i])<<std::endl;

		}
	}
}




void initialize_candidate_pairs(vector<queue>& queues, std::vector<tuple<int,int>> &buckets_delimiter, vector<tuple<int,int,int,int,int>> &buckets, vector<std::tuple<int,int,int,int,int,int>> &candidates, vector<int> &candidates_start, vector<arrayWrapper> &partitionsBucketsDelimiter, vector<arrayWrapper> &partitionsCandStart, vector<arrayWrapper> &partitionsBuckets, vector<arrayWrapper> &partitionsCandidates ){



	int j=0;
	long size=0;

	buckets_delimiter.emplace_back(make_tuple(0,0));

	auto start=std::chrono::system_clock::now();

		for(int i=0; i<buckets.size()-1; i++){ // Pay attention to size of "bucket"

			get<1>(buckets_delimiter[j])++;

			if( (get<0>(buckets[i])!=get<0>(buckets[i+1]))
							|| (get<0>(buckets[i])==get<0>(buckets[i+1]) && get<1>(buckets[i])!=get<1>(buckets[i+1]) )
							|| (get<0>(buckets[i])==get<0>(buckets[i+1]) && get<1>(buckets[i])==get<1>(buckets[i+1]) && get<2>(buckets[i])!=get<2>(buckets[i+1])) ){
				j++;
				buckets_delimiter.emplace_back(make_tuple(i+1,0));
			}
		}

	auto end=std::chrono::system_clock::now();



	std::cout<<"\nTime cand-init: count element: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;


	start=std::chrono::system_clock::now();

		auto new_end=std::remove_if(dpstd::execution::par_unseq, buckets_delimiter.begin(),buckets_delimiter.end(),[](std::tuple<int,int> &e){return std::get<1>(e)<2;});

		buckets_delimiter.erase( new_end, buckets_delimiter.end());

	end=std::chrono::system_clock::now();

	std::cout<<"Time cand-init: remove element: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;

	int num_splits=queues.size();

	start=std::chrono::system_clock::now();

		size=0;
		int size_partition=static_cast<int>((buckets_delimiter.size()/num_splits));


		int rest=buckets_delimiter.size()%num_splits;

		int size_buckets_part=0;
		int size_cand_start_part=0;

		int total_size=0;

		int b=0;

		cout<<"\n\tNumber of partitions: "<<num_splits<<std::endl;

		for(int split=0; split<num_splits; split++){

			size_partition+=(split==(num_splits-1))?rest:0;


			cout<<"\n\tSize_partition at iter "<<split<<": "<<size_partition<<std::endl<<std::endl;
			cout<<"\tNorm delim at iter "<<split<<": "<<size_buckets_part<<std::endl<<std::endl;
			cout<<"\tNorm cand start at iter "<<split<<": "<<size_cand_start_part<<std::endl<<std::endl;


			for(int delim=0; delim<size_partition; delim++){


				candidates_start.emplace_back(size);

				int n=get<1>(buckets_delimiter[b]);
				size+=((n*(n-1))/2);

//				if(delim<10 || delim>size_partition-10 ){
//					std::cout<<get<0>(buckets_delimiter[b]) << " - "<<size_buckets_part<<" = ";
//				}
				get<0>(buckets_delimiter[b])-=size_buckets_part;

//				if(delim<10 || delim>size_partition-10 ){
//					std::cout<<get<0>(buckets_delimiter[b])<<"\t\t"<<candidates_start[b]<<std::endl;
//
//				}

//				if(split>0 && b==0){
//					size_cand_start_part=size;//candidates_start[size_partition];
//				}

				b++;
			}

//			cout<<"----------------------------------------------------------"<<std::endl;

//			cout<<"buckets_delimiter[size_partition] at iter "<<split<<": "<<((split==num_splits-1)?(buckets.size()-partitionsBuckets[split-1].size):get<0>(buckets_delimiter[b]))<<std::endl;
			cout<<"\tb at iter "<<split<<": "<<b<<std::endl;

			cout<<"----------------------------------------------------------"<<std::endl;


			if(num_splits==1){
				size_buckets_part=buckets.size();
			}else{
				size_buckets_part=(split==num_splits-1)?(buckets.size()-partitionsBuckets[split-1].size):get<0>(buckets_delimiter[b]);

			}

			size_cand_start_part=size;//candidates_start[size_partition];

			total_size+=size;
			size=0;

			size_t offset=split==0?0:partitionsBucketsDelimiter[split-1].size*split;

			partitionsBucketsDelimiter.emplace_back(arrayWrapper(size_partition, offset) );



			offset=split==0?0:partitionsBuckets[split-1].size+partitionsBuckets[split-1].offset;

			partitionsBuckets.emplace_back(arrayWrapper(size_buckets_part, offset));


			offset=split==0?0:partitionsCandStart[split-1].size*split;

			partitionsCandStart.emplace_back(arrayWrapper(size_partition, offset) );


			offset=split==0?0:partitionsCandidates[split-1].size+partitionsCandidates[split-1].offset;

			partitionsCandidates.emplace_back(arrayWrapper(size_cand_start_part,offset ));


		}

	end=std::chrono::system_clock::now();

	std::cout<<"\nTime cand-init: count max pairs: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;




	start=std::chrono::system_clock::now();

		candidates.resize(total_size, tuple<int,int,int,int,int,int>(-1,-1,-1,-1,-1,-1));

		std::cout<<"\n\tAddress candidates: "<<candidates.data()<<std::endl;
		std::cout<<"\tAddress candidates start: "<<candidates_start.data()<<std::endl;
		std::cout<<"\tAddress candidates start partition: "<<candidates_start.data()<<std::endl;

//		partitionsCandidates[0].setPtr(candidates);
//		partitionsCandidates[1].setPtr(candidates);
//		partitionsCandStart[0].setPtr(candidates_start);
//		partitionsCandStart[1].setPtr(candidates_start);

	end=std::chrono::system_clock::now();

	std::cout<<"\nTime cand-init: resize: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;

}


void initialize_candidate_pairs_onDevice(vector<queue>& queues, std::vector<tuple<int,int>> &buckets_delimiter, vector<tuple<int,int,int,int,int>> &buckets, vector<std::tuple<int,int,int,int,int,int>> &candidates, vector<int> &candidates_start, vector<arrayWrapper> &partitionsBucketsDelimiter, vector<arrayWrapper> &partitionsCandStart, vector<arrayWrapper> &partitionsBuckets, vector<arrayWrapper> &partitionsCandidates ){


	{

		auto start=std::chrono::system_clock::now();

		std::vector<tuple<int,int>> delimiter(buckets.size());

		get<0>(delimiter[0]) = 0;

		queue q(default_selector{},property::queue::in_order());



		{

			cl::sycl::buffer<tuple<int,int>> buckets_delimiter_buf{ delimiter.data(), delimiter.size(), {property::buffer::use_host_ptr()}};
			cl::sycl::buffer<tuple<int,int,int,int,int>> array_buf{ buckets.data(), buckets.size(), {property::buffer::use_host_ptr()}};

			q.submit([&](sycl::handler& cgh) {


			auto pv_acc = buckets_delimiter_buf.get_access<cl::sycl::access::mode::write>(cgh);

			auto array_acc = array_buf.get_access<cl::sycl::access::mode::read>(cgh);

			cgh.parallel_for<class partition_kernel>(cl::sycl::range<1>{buckets.size() - 2},
				[=](cl::sycl::id<1> idx) {

					if ( (get<0>(array_acc[idx[0]])!=get<0>(array_acc[idx[0] + 1]))
							|| (get<0>(array_acc[idx[0]])==get<0>(array_acc[idx[0] + 1]) && get<1>(array_acc[idx[0]])!=get<1>(array_acc[idx[0] + 1]) )
							|| (get<0>(array_acc[idx[0]])==get<0>(array_acc[idx[0] + 1]) && get<1>(array_acc[idx[0]])==get<1>(array_acc[idx[0] + 1]) && get<2>(array_acc[idx[0]])!=get<2>(array_acc[idx[0] + 1])) ) {

						get<0>(pv_acc[idx[0] + 1]) = idx[0] + 1;
					}

				});
			}).wait();
		} // For synch

		auto new_end=std::remove_if(dpstd::execution::par_unseq, delimiter.begin()+1,delimiter.end(),[](std::tuple<int,int> &e){return std::get<0>(e)==0;});

			delimiter.erase( new_end, delimiter.end());

		auto end=std::chrono::system_clock::now();
		std::cout<<"Time cand-init: parallel count element: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;


		size_t size=0;
		 start=std::chrono::system_clock::now();

		{

			cl::sycl::buffer<tuple<int,int>> buckets_delimiter_buf{ delimiter.data(), delimiter.size(), {property::buffer::use_host_ptr()}};

			q.submit([&](sycl::handler& cgh) {


			auto pv_acc = buckets_delimiter_buf.get_access<cl::sycl::access::mode::write>(cgh);


			cgh.parallel_for<class partition_kernel>(cl::sycl::range<1>{buckets.size()-1},
				[=](cl::sycl::id<1> idx) {

					get<1>(pv_acc[idx[0]]) = get<0>(pv_acc[idx[0]+1]) - get<0>(pv_acc[idx[0]]);

				});

			}).wait();
		} // For synch

		 new_end=std::remove_if(dpstd::execution::par_unseq, delimiter.begin(),delimiter.end(),[](std::tuple<int,int> &e){return std::get<1>(e)<2;});

			delimiter.erase( new_end, delimiter.end());

		end=std::chrono::system_clock::now();
		std::cout<<"\nTime cand-init: parallel count size partition: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;


		std::for_each(delimiter.begin(), delimiter.begin()+10,[](tuple<int,int>d){
			std::cout<<get<0>(d)<<" "<<get<1>(d)<<std::endl;
		});


	}

		buckets_delimiter.clear();


	int j=0;
	long size=0;

	buckets_delimiter.emplace_back(make_tuple(0,0));

	auto start=std::chrono::system_clock::now();

		for(int i=0; i<buckets.size()-1; i++){ // Pay attention to size of "bucket"

			get<1>(buckets_delimiter[j])++;

			if( (get<0>(buckets[i])!=get<0>(buckets[i+1]))
							|| (get<0>(buckets[i])==get<0>(buckets[i+1]) && get<1>(buckets[i])!=get<1>(buckets[i+1]) )
							|| (get<0>(buckets[i])==get<0>(buckets[i+1]) && get<1>(buckets[i])==get<1>(buckets[i+1]) && get<2>(buckets[i])!=get<2>(buckets[i+1])) ){
				j++;
				buckets_delimiter.emplace_back(make_tuple(i+1,0));
			}
		}

	auto end=std::chrono::system_clock::now();

	std::cout<<std::endl;

	std::cout<<"Time cand-init: count element: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;


	std::for_each(buckets_delimiter.begin(), buckets_delimiter.begin()+10,[](tuple<int,int>d){
				std::cout<<get<0>(d)<<" "<<get<1>(d)<<std::endl;
	});



	start=std::chrono::system_clock::now();

		auto new_end=std::remove_if(dpstd::execution::par_unseq, buckets_delimiter.begin(),buckets_delimiter.end(),[](std::tuple<int,int> &e){return std::get<1>(e)<2;});

		buckets_delimiter.erase( new_end, buckets_delimiter.end());

	end=std::chrono::system_clock::now();

	std::cout<<"Time cand-init: remove element: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;

	int num_splits=queues.size();

	start=std::chrono::system_clock::now();

		size=0;
		int size_partition=static_cast<int>((buckets_delimiter.size()/num_splits));


		int rest=buckets_delimiter.size()%num_splits;

		int size_buckets_part=0;
		int size_cand_start_part=0;

		int total_size=0;

		int b=0;

		for(int split=0; split<num_splits; split++){

			size_partition+=(split==(num_splits-1))?rest:0;


			cout<<"Size_partition at iter "<<split<<": "<<size_partition<<std::endl;
			cout<<"Norm delim at iter "<<split<<": "<<size_buckets_part<<std::endl;
			cout<<"Norm cand start at iter "<<split<<": "<<size_cand_start_part<<std::endl;


			for(int delim=0; delim<size_partition; delim++){


				candidates_start.emplace_back(size);

				int n=get<1>(buckets_delimiter[b]);
				size+=((n*(n-1))/2);

				if(delim<10 || delim>size_partition-10 ){
					std::cout<<get<0>(buckets_delimiter[b]) << " - "<<size_buckets_part<<" = ";
				}
				get<0>(buckets_delimiter[b])-=size_buckets_part;

				if(delim<10 || delim>size_partition-10 ){
					std::cout<<get<0>(buckets_delimiter[b])<<"\t\t"<<candidates_start[b]<<std::endl;

				}

//				if(split>0 && b==0){
//					size_cand_start_part=size;//candidates_start[size_partition];
//				}

				b++;
			}

			cout<<"----------------------------------------------------------"<<std::endl;

			cout<<"buckets_delimiter[size_partition] at iter "<<split<<": "<<((split==num_splits-1)?(buckets.size()-partitionsBuckets[split-1].size):get<0>(buckets_delimiter[b]))<<std::endl;
			cout<<"b at iter "<<split<<": "<<b<<std::endl;

			size_buckets_part=(split==num_splits-1)?(buckets.size()-partitionsBuckets[split-1].size):get<0>(buckets_delimiter[b]);

			size_cand_start_part=size;//candidates_start[size_partition];

			total_size+=size;
			size=0;

			size_t offset=split==0?0:partitionsBucketsDelimiter[split-1].size*split;

			partitionsBucketsDelimiter.emplace_back(arrayWrapper(size_partition, offset) );



			offset=split==0?0:partitionsBuckets[split-1].size+partitionsBuckets[split-1].offset;

			partitionsBuckets.emplace_back(arrayWrapper(size_buckets_part, offset));


			offset=split==0?0:partitionsCandStart[split-1].size*split;

			partitionsCandStart.emplace_back(arrayWrapper(size_partition, offset) );


			offset=split==0?0:partitionsCandidates[split-1].size+partitionsCandidates[split-1].offset;

			partitionsCandidates.emplace_back(arrayWrapper(size_cand_start_part,offset ));


		}

	end=std::chrono::system_clock::now();

	std::cout<<"Time cand-init: count max pairs: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;

	std::cout<<"\n\n\n\n"<<std::endl;

	std::cout<<"Address: "<<candidates.data()<<std::endl;

//	cout<<"buckets[0]: size = "<< partitions[0].second << "\toffset = "<<0<<std::endl;
//	cout<<"buckets[1]: size = "<< partitions[1].second << "\toffset = "<<partitions[0].second<<std::endl;


	std::cout<<"\n\n\n\n"<<std::endl;

	start=std::chrono::system_clock::now();

		vector<tuple<int,int,int,int,int,int>> tmp(total_size);

	end=std::chrono::system_clock::now();

	std::cout<<"Allocation at declaration: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;

	tmp.size();

	start=std::chrono::system_clock::now();

		candidates.resize(total_size, tuple<int,int,int,int,int,int>(-1,-1,-1,-1,-1,-1));

		std::cout<<"Address candidates: "<<candidates.data()<<std::endl;
		std::cout<<"Address candidates start: "<<candidates_start.data()<<std::endl;
		std::cout<<"Address candidates start partition: "<<candidates_start.data()<<std::endl;

//		partitionsCandidates[0].setPtr(candidates.data());
//		partitionsCandidates[1].setPtr(candidates.data());
//		partitionsCandStart[0].setPtr(candidates_start.data());
//		partitionsCandStart[1].setPtr(candidates_start.data());

	end=std::chrono::system_clock::now();

	std::cout<<"Time cand-init: resize: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;

}



void parallel_embedding_batched_wrapper(queue &device_queue, vector<int> &len_oristrings, char (*oristrings)[LEN_INPUT], char** set_embdata_dev, unsigned int batch_size,uint32_t n_batches, vector<int> &lshnumber, uint32_t &len_output){

	std::cout<< "Selected: Parallel embedding - batched version"<<std::endl;


	// DICTIONARY

	int dictionary[256]={0};
	inititalize_dictionary(dictionary);


	for(int n=0; n<n_batches; n++){
		set_embdata_dev[n]=(char*)malloc_shared(batch_size*NUM_STR*NUM_REP*len_output*sizeof(char), device_queue.get_device(), device_queue.get_context());
	}

	uint32_t len_p=samplingrange+1;

	int (*p)=new int[NUM_STR*NUM_CHAR*len_p];

	generate_random_string(p, len_p);

	for(int n=0; n<n_batches; n++){
		parallel_embedding_batched(device_queue, len_oristrings.data()+n*batch_size, (char*) oristrings[n*batch_size], (char*)set_embdata_dev[n], batch_size, lshnumber, p, len_p, len_output, dictionary);
	}

	delete[] p;
}




void parallel_embedding_batched_2dev_wrapper(vector<queue> &queues, vector<int> &len_oristrings, char (*oristrings)[LEN_INPUT], char** &set_embdata_dev, unsigned int batch_size, uint32_t n_batches, vector<int> &lshnumber, uint32_t &len_output){

	std::cout<< "Selected: Parallel embedding - batched version"<<std::endl;



	std::atomic<int> task(0);

	// DICTIONARY

	int dictionary[256]={0};
	inititalize_dictionary(dictionary);

	cout<<"\n\tLen output"<<len_output<<std::endl;


	for(int n=0; n<n_batches; n++){
			set_embdata_dev[n]=malloc_shared<char>(batch_size*NUM_STR*NUM_REP*len_output, queues.back());
	}

	uint32_t len_p=samplingrange+1;

	int (*p)=new int[NUM_STR*NUM_CHAR*len_p];

	generate_random_string(p, len_p);

	vector<std::thread> threads;

	for(auto &q : queues){

		threads.push_back(thread([&](){


			while(true){

				int n=task.fetch_add(1);

				if(n<n_batches){
					// set_embdata_gpu[n]=(char*)malloc_shared(batch_size*NUM_STR*NUM_REP*len_output*sizeof(char), cpu_queue.get_device(), cpu_queue.get_context());
	//				set_embdata_dev[n]=malloc_shared<char>(batch_size*NUM_STR*NUM_REP*len_output, gpu_queue);

					parallel_embedding_batched(q, len_oristrings.data()+n*batch_size, (char*) oristrings[n*batch_size], (char*)set_embdata_dev[n], batch_size, lshnumber, p, len_p, len_output, dictionary);
				}
				else{
					break;
				}

			}
		}));


	}

	for(auto &t:threads){
		if(t.joinable()){
			t.join();
		}
	}


	delete[] p;
}


void parallel_embedding_while_loop_2dev_wrapper(vector<queue> &queues, vector<int> &len_oristrings, char (*oristrings)[LEN_INPUT], char** &set_embdata_dev, unsigned int batch_size, uint32_t n_batches, vector<int> &lshnumber, uint32_t &len_output){

	std::cout<< "Selected: Parallel embedding - while loop version"<<std::endl;



	std::atomic<int> task(0);

	// DICTIONARY

	int dictionary[256]={0};
	inititalize_dictionary(dictionary);

	len_output=lshnumber.size();

	cout<<"\n\tlen output: "<<len_output<<std::endl;

	for(int n=0; n<n_batches; n++){
			set_embdata_dev[n]=malloc_shared<char>(batch_size*NUM_STR*NUM_REP*len_output, queues.back());
	}

	uint32_t len_p=samplingrange+1;

	int (*p)=new int[NUM_STR*NUM_CHAR*len_p];

	generate_random_string(p, len_p);

	vector<std::thread> threads;

	for(auto &q : queues){

		threads.push_back(thread([&](){


			while(true){

				int n=task.fetch_add(1);

				if(n<n_batches){
					// set_embdata_gpu[n]=(char*)malloc_shared(batch_size*NUM_STR*NUM_REP*len_output*sizeof(char), cpu_queue.get_device(), cpu_queue.get_context());
	//				set_embdata_dev[n]=malloc_shared<char>(batch_size*NUM_STR*NUM_REP*len_output, gpu_queue);

					parallel_embedding_while_loop(q, len_oristrings.data()+n*batch_size, (char*) oristrings[n*batch_size], (char*)set_embdata_dev[n], batch_size, lshnumber, p, len_p, len_output, dictionary);
				}
				else{
					break;
				}

			}
		}));


	}

	for(auto &t:threads){
		if(t.joinable()){
			t.join();
		}
	}


	delete[] p;
}




void parallel_embedding_USM_wrapper(queue &device_queue, vector<int> &len_oristrings, char (*oristrings)[LEN_INPUT], char** set_embdata_dev, unsigned int batch_size, uint32_t n_batches, vector<int> &lshnumber, uint32_t &len_output){

	std::cout<< "Selected: Parallel embedding - USM version"<<std::endl;

	int dictionary[256]={0};
	inititalize_dictionary(dictionary);

	for(int n=0; n<n_batches; n++){
		set_embdata_dev[n]=(char*)malloc_shared(batch_size*NUM_STR*NUM_REP*len_output*sizeof(char), device_queue.get_device(), device_queue.get_context());
	}

	uint32_t len_p=samplingrange+1;

	int (*p)=new int[NUM_STR*NUM_CHAR*len_p];

	generate_random_string(p, len_p);

	parallel_embeddingUSM(device_queue, len_oristrings.data(), (char*) oristrings, (char**)set_embdata_dev, batch_size, lshnumber, p, len_p, len_output, dictionary);

	delete[] p;
}



int main(int argc, char **argv) {


	int device=0;
	int alg_number[3]={0,0,0};

	unsigned int batch=30000;
	unsigned int n_batches=10;

	if (argc==10){

		filename = argv[1];
		device=atoi(argv[2]);
		alg_number[0]=atoi(argv[3]); // Embed function
		alg_number[1]=atoi(argv[4]); // Create buckets
		alg_number[2]=atoi(argv[5]); // Generate candidate

		if (argc>6){
			samplingrange=atoi(argv[6]);
		}

		if(argc>7){
			countfilter=atoi(argv[7]);
		}

		if(argc>8){
			batch=atoi(argv[8]);
		}
		if(argc>9){
			n_batches=atoi(argv[9]);
		}

//		k_input = atoi(argv[2]);
//		num_str = atoi(argv[3]);
//		num_hash = atoi(argv[4]);
//		num_bit = atoi(argv[5]);
//		num_char = atoi(argv[6]);
//		samplingrange = atoi(argv[7]);
//		outputresults = atoi(argv[8]);
//		if (argc >9)
//			shift= atoi(argv[9]);
//		if (argc >10)
//			countfilter= atoi(argv[10]);
//		if(shift !=0)
//			num_rep = ceil(double (k_input) / double (shift));
		//srand(time(NULL));

	}
	else{
		fprintf(stderr, "usage: ./embedjoin inputdata 0/1(cpu/gpu) [0-2]step1 [0-1]step2 [0-1]step3\n");
		exit(-1);
	}


	//OUTPUT STRINGS

    uint32_t len_output=samplingrange;

	print_configuration(batch, n_batches, len_output);

	auto asyncHandler = [&](cl::sycl::exception_list eL) {
		for (auto& e : eL) {
			try {
				std::rethrow_exception(e);
			}catch (cl::sycl::exception& e) {
				std::cout << e.what() << std::endl;
				std::cout << "fail" << std::endl;
				// std::terminate() will exit the process, return non-zero, and output a
				// message to the user about the exception
				std::terminate();
			}
		}
	};

	//DECLARING VARIABLES

	long time_init=0;
	long time_embedding_data=0;
	long time_create_buckets=0;
	long time_sorting_buckets=0;
	long time_candidate_initialization=0;
	long time_generate_candidates=0;
	long time_candidate_processing=0;
	long time_sorting_candidates=0;
	long time_edit_distance=0;
	long total_time_join=0;
	long total_time=0;


	std::chrono::time_point<std::chrono::system_clock> start, end, start_tot, end_tot;
	std::chrono::time_point<std::chrono::system_clock> start_alg, end_alg;

	//HASH

	vector<int> a; // the random vector for second level hash table
	int (*hash_lsh)[NUM_BITS] = new int[NUM_HASH][NUM_BITS];

	vector<int> lshnumber;

	//INPUT STRINGS

	char (*oristrings)[LEN_INPUT];
	oristrings = new char[NUM_STRING][LEN_INPUT];
	vector<string> oridata_modified;
	vector<int> len_oristrings;





    
	vector<tuple<int,int,int,int,int>> buckets(NUM_STRING*NUM_STR*NUM_HASH*NUM_REP);

	std::vector<std::tuple<int,int,int,int,int,int>> candidates;

	std::vector<tuple<int,int>> buckets_delimiter;
	std::vector<int> candidates_start;


	vector<queue> queues;

	if(device==0 || device==2){

		queues.push_back(queue(cpu_selector{}, asyncHandler, property::queue::in_order()));

	}

	if(device==1 || device==2){

		try{

			queue tmp_queue(gpu_selector{}, asyncHandler, property::queue::in_order());

			queues.push_back(std::move(tmp_queue));

		}catch(std::exception& e){
			std::cout<<"Attention: no GPU device detected. The program will run on CPU."<<std::endl;
			device=1; // Force device to CPU
		}

	}

	cout<<"\nNumber of devices: "<<queues.size()<<std::endl<<std::endl;

	char **set_embdata_dev=(char**)malloc_shared<char*>(n_batches, queues.back());


	 /**
	  *
	  * FOR BENCHMARK
	  *
	  * **/


	 std::vector<void(*)(vector<queue>&, vector<int> &, char (*)[LEN_INPUT], char** &, unsigned int , uint32_t , vector<int> &, uint32_t &)> f_emb;
	 std::vector<void(*)(vector<queue>&, char **, vector<tuple<int,int,int,int,int>> &, uint32_t , uint32_t , int* , vector<int> &, vector<int> &, uint32_t)> f_bucket;
	 std::vector<void(*)(vector<queue>&, vector<int> &, char* , char **, vector<tuple<int,int,int,int,int>> &, unsigned int , vector<tuple<int,int>> &, vector<std::tuple<int,int,int,int,int,int>>&, vector<int> &, int* , vector<int>&, uint32_t , vector<arrayWrapper> &, vector<arrayWrapper> &, vector<arrayWrapper> &, vector<arrayWrapper>&)> f_cand;


	 void (*f_parallel_embedding_batched_wrapper)(vector<queue>&, vector<int> &, char (*)[LEN_INPUT], char** &, unsigned int , uint32_t , vector<int> &, uint32_t &){parallel_embedding_batched_2dev_wrapper};
	 void (*f_parallel_embedding_USM_wrapper)(queue&, vector<int>&, char (*)[LEN_INPUT], char** , unsigned int , uint32_t , vector<int>&, uint32_t&){parallel_embedding_USM_wrapper};
	 void (*f_parallel_embedding_while_loop)(vector<queue>&, vector<int> &, char (*)[LEN_INPUT], char** &, unsigned int , uint32_t , vector<int> &, uint32_t &){parallel_embedding_while_loop_2dev_wrapper};




	 void(*f_create_buckets)(vector<queue>&, char **, vector<tuple<int,int,int,int,int>> &, uint32_t , uint32_t , int* , vector<int> &, vector<int> &, uint32_t ){create_buckets_2dev_wrapper};

	 void(*f_create_buckets_without_offset)(vector<queue>&, char **, vector<tuple<int,int,int,int,int>> &, uint32_t , uint32_t , int* , vector<int> &, vector<int> &, uint32_t ){create_buckets_without_lshnumber_offset_2dev_wrapper};



	 void(*f_generate_candidates)(vector<queue>&, vector<int> &, char* , char **, vector<tuple<int,int,int,int,int>> &, unsigned int , vector<tuple<int,int>> &, vector<std::tuple<int,int,int,int,int,int>>&, vector<int> &, int* , vector<int>&, uint32_t , vector<arrayWrapper> &, vector<arrayWrapper> &, vector<arrayWrapper> &, vector<arrayWrapper> &){generate_candidates_2dev_wrapper};

	 void(*f_generate_candidates_without_offset)(vector<queue>&, vector<int> &, char* , char **, vector<tuple<int,int,int,int,int>> &, unsigned int , vector<tuple<int,int>> &, vector<std::tuple<int,int,int,int,int,int>>& , vector<int> &, int * , vector<int> &, uint32_t , vector<arrayWrapper> &, vector<arrayWrapper> &, vector<arrayWrapper> &, vector<arrayWrapper> &){generate_candidates_without_lshnumber_offset_2dev_wrapper};


	 f_emb.push_back(f_parallel_embedding_batched_wrapper); // 0

//	 f_emb.push_back(f_parallel_embedding_USM_wrapper); // 1
	 f_emb.push_back(f_parallel_embedding_batched_wrapper); // 1

	 f_emb.push_back(f_parallel_embedding_while_loop); // 2



	 f_bucket.push_back(f_create_buckets); // 0
	 f_bucket.push_back(f_create_buckets_without_offset); // 1

	 f_cand.push_back(f_generate_candidates); // 0
	 f_cand.push_back(f_generate_candidates_without_offset); // 1


	/**
	 *
	 * INITIALIZATION
	 *
	 * */

	start_alg=std::chrono::system_clock::now();

	start = std::chrono::system_clock::now();

	srand(11110);
	initialization(len_oristrings, oristrings, oridata_modified, hash_lsh, a, lshnumber);


	end = std::chrono::system_clock::now();


	time_init=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();


	std::cerr<< "Start parallel algorithm..."<<std::endl<<std::endl;




    start=std::chrono::system_clock::now();

	/**
	 *
	 *
	 * EMBEDDING STEP
	 *
	 *
	 **/


    f_emb[alg_number[0]](queues, len_oristrings, oristrings, set_embdata_dev, batch, n_batches, lshnumber, len_output);

//    parallel_embedding_USM_wrapper(queues.back(), len_oristrings, oristrings, set_embdata_dev, batch, n_batches, lshnumber, len_output);

//    parallel_embedding_batched_2dev_wrapper(queues, len_oristrings, oristrings, set_embdata_dev, batch, n_batches, lshnumber, len_output);;

//    parallel_embedding_while_loop_2dev_wrapper(queues, len_oristrings, oristrings, set_embdata_dev, batch, n_batches, lshnumber, len_output);

    //device_queue.wait();

    for(auto &q : queues){
    	q.wait();
    }


	end=std::chrono::system_clock::now();

	time_embedding_data=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

	cout<<"Time: "<<(float)time_embedding_data/1000<<"sec"<<std::endl;




#if PRINT_EACH_STEP
	print_embedded(set_embdata_dev, len_output, batch);
#endif



	start_tot=std::chrono::system_clock::now();


//	/**
//	 *
//	 *
//	 * CREATE BUCKETS STEP
//	 *
//	 *
//	 * **/
//
//
//
	start=std::chrono::system_clock::now();

	f_bucket[alg_number[1]](queues, (char**)set_embdata_dev, buckets, n_batches, batch, (int*)hash_lsh, a, lshnumber, len_output);


//	create_buckets_2dev_wrapper(queues, (char**)set_embdata_dev, buckets, n_batches, batch, (int*)hash_lsh, a, lshnumber, len_output);

//	create_buckets_without_lshnumber_offset_2dev_wrapper(queues, (char**)set_embdata_dev, buckets, n_batches, batch, (int*)hash_lsh, a, lshnumber, len_output);

	end=std::chrono::system_clock::now();

	time_create_buckets=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

	cout<<"Time buckets creation: "<<(float)time_create_buckets/1000<<"sec"<<std::endl;



	start=std::chrono::system_clock::now();

//TODO:	Parallel



//	 auto policy_sort = make_device_policy<class PolicySort>(device_queue);

//	 cl::sycl::buffer<tuple<int,int,int,int,int>,1> buf(buckets.data(),range<1>{buckets.size()});
//	 auto buf_begin = dpstd::begin(buf);
//	 auto buf_end   = dpstd::end(buf);

	 tbb::parallel_sort(buckets.begin(), buckets.end(), [](std::tuple<int,int,int,int,int>& e1, std::tuple<int,int,int,int,int>& e2) {
	 		 return ( ( get<0>(e1)<get<0>(e2) ) ||
	 				 ( get<0>(e1)==get<0>(e2) && get<1>(e1)<get<1>(e2) ) ||
	 				 ( get<0>(e1)==get<0>(e2) && get<1>(e1)==get<1>(e2) && get<2>(e1)<get<2>(e2) )  ||
	 				 ( get<0>(e1)==get<0>(e2) && get<1>(e1)==get<1>(e2) && get<2>(e1)==get<2>(e2) && get<3>(e1)<get<3>(e2) ) ||
	 				 ( get<0>(e1)==get<0>(e2) && get<1>(e1)==get<1>(e2) && get<2>(e1)==get<2>(e2) && get<3>(e1)==get<3>(e2) && get<4>(e1)<get<4>(e2) )); } );



//	 std::sort( buckets.begin(), buckets.end(), [](std::tuple<int,int,int,int,int> t1, std::tuple<int,int,int,int,int> t2){return (t1 < t2);});

	 end=std::chrono::system_clock::now();

	 time_sorting_buckets=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

//	 print_buckets(buckets);

#if PRINT_EACH_STEP
	print_buckets(buckets);
#endif
	 /**
	  *
	  * INITIALIZATION FOR CANDIDATE GENERATION
	  *
	  *
	  * **/

	 start=std::chrono::system_clock::now();

	 //TODO: Change the "2" among parameter
	 vector<arrayWrapper> partitionsBuckets;
	 vector<arrayWrapper> partitionsBucketsDelimiter;
	 vector<arrayWrapper> partitionsCandStart;
	 vector<arrayWrapper> partitionsCandidates;


	 initialize_candidate_pairs(queues, buckets_delimiter, buckets, candidates, candidates_start, partitionsBucketsDelimiter, partitionsCandStart, partitionsBuckets, partitionsCandidates);

	 end=std::chrono::system_clock::now();

	 time_candidate_initialization=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

	 std::cout<<time_candidate_initialization<<std::endl;




	 //exit(0);

	 /**
	 *
	 *
	 * GENERATE CANDIDATE PAIRS STEP
	 *
	 *
	 * **/


	 start=std::chrono::system_clock::now();


	 f_cand[alg_number[2]](queues, len_oristrings, (char*)oristrings, (char**)set_embdata_dev, buckets, batch, buckets_delimiter, candidates, candidates_start, (int *)hash_lsh, lshnumber, len_output, partitionsBucketsDelimiter, partitionsCandStart, partitionsBuckets, partitionsCandidates);

// 	 generate_candidates_2dev_wrapper(queues, len_oristrings, (char*)oristrings, (char**)set_embdata_dev, buckets, batch, buckets_delimiter, candidates, candidates_start, (int *)hash_lsh, lshnumber, len_output, partitionsBucketsDelimiter, partitionsCandStart, partitionsBuckets, partitionsCandidates);
//	 generate_candidates_without_lshnumber_offset_2dev_wrapper(queues, len_oristrings, (char*)oristrings, (char**)set_embdata_dev, buckets, batch, buckets_delimiter, candidates, candidates_start, (int *)hash_lsh, lshnumber, len_output, partitionsBucketsDelimiter, partitionsCandStart, partitionsBuckets, partitionsCandidates);

	 end=std::chrono::system_clock::now();

	 time_generate_candidates=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();


	 cout<<"\n\nStarting candidate processing analysis..."<<std::endl;

	 start=std::chrono::system_clock::now();


	 auto start_2=std::chrono::system_clock::now();

	 	 vector<std::tuple<int,int>> verifycan;

	// auto policy_rem = make_device_policy(device_queue);

	 	 candidates.erase(remove_if(dpstd::execution::par_unseq, candidates.begin(), candidates.end(),[](std::tuple<int,int,int,int,int,int> e){return (get<0>(e)==-1 || get<4>(e)>K_INPUT || get<5>(e)!=0 || get<0>(e)==get<2>(e));}), candidates.end());

	 auto end_2=std::chrono::system_clock::now();
	 std::cout<<std::endl;
	 std::cout<<"Remove some candidates: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end_2-start_2).count()/1000<<std::endl;



	 start_2=std::chrono::system_clock::now();

//	 sort(candidates.begin(),candidates.end());

//	 auto policy_sort = make_device_policy<class PolicySort>(device_queue);

//TODO: Check comparing!
	 tbb::parallel_sort(candidates.begin(),candidates.end(), [](std::tuple<int,int,int,int,int,int> e1, std::tuple<int,int,int,int,int,int> e2) {
		 return ( ( get<0>(e1)<get<0>(e2) ) ||
				 ( get<0>(e1)==get<0>(e2) && get<1>(e1)<get<1>(e2) ) ||
				 ( get<0>(e1)==get<0>(e2) && get<1>(e1)==get<1>(e2) && get<2>(e1)<get<2>(e2) )  ||
				 ( get<0>(e1)==get<0>(e2) && get<1>(e1)==get<1>(e2) && get<2>(e1)==get<2>(e2) && get<3>(e1)<get<3>(e2) )/* ||
				 ( get<0>(e1)==get<0>(e2) && get<1>(e1)==get<1>(e2) && get<2>(e1)==get<2>(e2) && get<3>(e1)==get<3>(e2) && get<4>(e1)<get<4>(e2) ) ||
				 ( get<0>(e1)==get<0>(e2) && get<1>(e1)==get<1>(e2) && get<2>(e1)==get<2>(e2) && get<3>(e1)==get<3>(e2) && get<4>(e1)==get<4>(e2) && get<5>(e1)<get<5>(e2) )*/ ); } );

	 end_2=std::chrono::system_clock::now();

	 time_sorting_candidates=std::chrono::duration_cast<std::chrono::milliseconds>(end_2-start_2).count();


 	std::cout<<std::endl;
 	std::cout<<"Time sorting candidates freq: "<<(float)time_sorting_candidates/1000<<std::endl;


	std::cerr<<"\n\tCandidate after filter out: "<<candidates.size()<<std::endl;


#if PRINT_EACH_STEP
	print_candidate_pairs(candidates);
#endif


	/*
	 *
	 * COUNTING FREQUENCIES
	 *
	 * **/




	start_2=std::chrono::system_clock::now();

		vector<int> freq_uv;
		if (!candidates.empty())
		{
			freq_uv.push_back(0);
			auto prev = candidates[0];        // you should ensure !uv.empty() if previous code did not already ensure it.
			for (auto const & x : candidates)
			{
				if (prev != x)
				{
					freq_uv.push_back(0);
					prev = x;
				}
				++freq_uv.back();
			}
		}

	end_2=std::chrono::system_clock::now();
	std::cout<<std::endl;
	std::cout<<"Counting freq: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end_2-start_2).count()/1000<<std::endl;

	start_2=std::chrono::system_clock::now();

		candidates.erase(unique( candidates.begin(), candidates.end() ), candidates.end());

	end_2=std::chrono::system_clock::now();
	std::cout<<"Make uniq: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end_2-start_2).count()/1000<<std::endl;

	start_2=std::chrono::system_clock::now();

		for (int i = 0; i < candidates.size(); i++)
		{
			if (freq_uv[i] > countfilter )
			{
				verifycan.emplace_back(get<0>(candidates[i]),get<2>(candidates[i]));
			}
		}

	end_2=std::chrono::system_clock::now();
	std::cout<<"Filter out candidates: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end_2-start_2).count()/1000<<std::endl;

	int num_candidate=0;

	start_2=std::chrono::system_clock::now();

		sort(verifycan.begin(), verifycan.end());

	end_2=std::chrono::system_clock::now();
	std::cout<<"Sort verifycan: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end_2-start_2).count()/1000<<std::endl;


	start_2=std::chrono::system_clock::now();

		verifycan.erase(unique(verifycan.begin(), verifycan.end()), verifycan.end());

	end_2=std::chrono::system_clock::now();
	std::cout<<"Uniq verifycan: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end_2-start_2).count()/1000<<std::endl;

	end=std::chrono::system_clock::now();

	time_candidate_processing=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

	cout<<"\nEnd candidates processing"<<std::endl;

	start=std::chrono::system_clock::now();


	/**
	 *
	 * EDIT DISTANCE CALCULATION
	 *
	 * */


	unsigned int num_threads = std::thread::hardware_concurrency();

	cout<<"\nNumber of threads for edit distance: "<<num_threads<<std::endl;



	vector<std::thread> workers;
	std::atomic<int> verified(0);
	int to_verify=verifycan.size();

	int cont=0;
	std::mutex mt;

//	vector<int> solutions(to_verify,0);


	for(int t=0; t<num_threads; t++){


		workers.push_back(std::thread([&](){

			while(true){

				int j=verified.fetch_add(1);

				if(j<to_verify){

//					solutions[j]++;

					int first_str;
					int second_str;

					first_str=get<0>(verifycan[j]);
					second_str=get<1>(verifycan[j]);
//					int ed = edit_distance(oridata_modified[second_str].data(), len_oristrings[second_str]/*tmp_oridata[second_str].size()*/,oridata_modified[first_str].data(), len_oristrings[first_str] /*tmp_oridata[first_str].size()*/, K_INPUT);
					string tmp_str1=oridata_modified[first_str];
					string tmp_str2=oridata_modified[second_str];

					int ed = edit_distance(tmp_str2.data(), len_oristrings[second_str], tmp_str1.data(), len_oristrings[first_str] /*tmp_oridata[first_str].size()*/, K_INPUT);

					// Critical section
					std::unique_lock<std::mutex> lk(mt);


					if(ed != -1) {
						cont++;
						//cout<<j<<std::endl;
						outputs.push_back(make_tuple(first_str, second_str));
					}

					num_candidate++;

				}
				else{
					break;
				}

			}

		}));


	}


	for(auto &t:workers){
			if(t.joinable()){
				t.join();
		}
	}
//
//	for(int t=0; t<solutions.size(); t++){
//		if(solutions[t]!=1){
//			cout<<"ERROR!"<<std::endl;
//		}
//	}

	cout<<"\n\tCont: "<<cont<<std::endl;


	end=std::chrono::system_clock::now();
	time_edit_distance=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

	end_tot=std::chrono::system_clock::now();

	total_time_join=std::chrono::duration_cast<std::chrono::milliseconds>(end_tot-start_tot).count();

	end_alg=std::chrono::system_clock::now();
	total_time=std::chrono::duration_cast<std::chrono::milliseconds>(end_alg-start_alg).count();


	delete[] hash_lsh;
	cout<<"\nDelete hash_lsh"<<std::endl;

	delete[] oristrings;
	cout<<"\nDelete oristrings"<<std::endl;

	for(int i=0; i<n_batches; i++){
		if(set_embdata_dev[i]==nullptr){
			cout<<"ERROR: Null pointer!"<<std::endl;
		}else{
			free(set_embdata_dev[i], queues.back());
			cout<<"Delete embdata["<<i<<"]"<<std::endl;

		}
	}
	if(set_embdata_dev==nullptr){
				cout<<"ERROR: Null pointer!"<<std::endl;
	}else{
		free(set_embdata_dev, queues.back());
		cout<<"Delete embdata"<<std::endl;
	}


	std::string distinguisher="";

	if(device==0){
		distinguisher+="-GPU-";
	}else if(device==1){
		distinguisher+="-CPU-";
	}else if(device==2){
		distinguisher+="-BOTH-";
	}
	else{
		distinguisher+="-ERROR";
	}

	if(alg_number[0]==0 && alg_number[1]==0 && alg_number[2]==0){
		distinguisher+="BATCHED-NO_WHILE-";
	}
	else if(alg_number[0]==2 && alg_number[1]==1 && alg_number[2]==1){
		distinguisher+="BATCHED-WHILE-";
	}
	else{
		distinguisher+="ERROR";
	}

	distinguisher+=std::to_string(len_output);

	std::cout<<std::endl<<std::endl<<std::endl;
	std::cout<<"Report:"<<std::endl<<std::endl;
	std::cout<<"Time read data: "<<(float)time_init/1000<<std::endl;
	std::cout<<"Time PARALLEL embedding data:\t"<< (float)time_embedding_data/1000<<"sec"<<std::endl;
	std::cout<<"Time PARALLEL buckets generation:\t"<< (float)time_create_buckets/1000<<"sec"<<std::endl;
	std::cout<<"Time buckets sorting: "<< (float)time_sorting_buckets/1000<<"sec"<<std::endl;
	std::cout<<"Time candidate initialization:\t"<< (float)time_candidate_initialization/1000<<"sec"<<std::endl;
	std::cout<<"Time PARALLEL candidates generation:\t"<< (float)time_generate_candidates/1000<<"sec"<<std::endl;
	std::cout<<"Time candidates processing:\t"<< (float)time_candidate_processing/1000<<"sec"<<std::endl;
	std::cout<<"Time candidates sorting(within cand-processing):\t"<< (float)time_sorting_candidates/1000<<"sec"<<std::endl;
	std::cout<<"Time compute edit distance:\t"<<(float) time_edit_distance/1000<<"sec"<<std::endl;
	std::cout<<"Total time parallel join:\t"<< (float)total_time_join/1000<<"sec"<<std::endl;
	std::cout<<"Total elapsed time :\t"<< (float)total_time/1000<<"sec"<<std::endl;

	//std::cout<<"Number of candidates: "<<num_candidate<<std::endl;

	{

		ofstream outFile;

		outFile.open("report-"+filename+distinguisher, ios::out | ios::trunc);

		std::string dev=queues.back().get_device().get_info<info::device::name>();
			if (outFile.is_open()) {

				//outFile<</*get<4>(candidates[i])<<", "<<get<5>(candidates[i])<<", "<<*/get<0>(candidates[i])<<", "<<get<1>(candidates[i])<<", "<<get<2>(candidates[i])<<", "<<get<3>(candidates[i])<<std::endl;
				outFile<<"Step,Time(sec),Device"<<std::endl;
				outFile<<"Read Data,"<<(float)time_init/1000<<std::endl;
				outFile<<"Embedding,"<<(float)time_embedding_data/1000<<","<<dev<<std::endl;
				outFile<<"Create Buckets,"<< (float)time_create_buckets/1000<<","<<dev<<std::endl;
				outFile<<"Sort Buckets,"<< (float)time_sorting_buckets/1000<<std::endl;
				outFile<<"Candidate Initialization,"<<(float)time_candidate_initialization/1000<<std::endl;
				outFile<<"Generate Candidate,"<< (float)time_generate_candidates/1000<<","<<dev<<std::endl;
				outFile<<"Sort candidates: "<<(float)time_sorting_candidates/1000<<std::endl;
				outFile<<"Candidates processing,"<< (float)time_candidate_processing/1000<<std::endl;
				outFile<<"Edit Distance,"<< (float)time_edit_distance/1000<<std::endl;
				outFile<<"Total Join time (w/o embedding),"<< (float)total_time_join/1000<<std::endl;
				outFile<<"Total Alg time,"<< (float)total_time/1000<<std::endl;

			}
			if(PRINT_EACH_STEP==1){
				std::cerr<<"Attention, join time include the print on file time"<<std::endl;
			}
	}


	print_output("join_output_parallel.txt");

	cout<<"Back to main"<<std::endl;
	return 0;

}


