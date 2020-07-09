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
	#define ALLOUTPUTRESULT 1
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




vector<int> indices;
//vector<vector<vector<string>>> embdata;// embedded strings

vector<idpair> outputs;

//TODO: Remove or manage better
vector<string> tmp_oridata;



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

	std::cout<<"Reading in read function: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<std::endl;


	for (int i = 0; i < tmp_oridata.size(); i++)
		indices.push_back(i);

	start=std::chrono::system_clock::now();

	tbb::parallel_sort(indices.begin(), indices.end(), [&](int i1, int i2) { return tmp_oridata[i1].size() <  tmp_oridata[i2].size();});
	tbb::parallel_sort(tmp_oridata.begin(), tmp_oridata.end(), [&](auto s1,auto s2){return s1.size()<s2.size();});

	end=std::chrono::system_clock::now();

	std::cout<<"Sorting in read function: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<std::endl;



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

	std::cout<<"Memory op in read function: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<std::endl;

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


	std::cout << "Device: " << device_queue.get_device().get_info<info::device::name>() << std::endl;

	cout << "\tTask: Embedding Data"<< std::endl;


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


	cout << "\tTask: Embedding Data\t";
	std::cout << "Device: " << device_queue.get_device().get_info<info::device::name>() << std::endl;

	unsigned int size_p=NUM_STR*NUM_CHAR*(samplingrange+1);

	buffer<int,1> buffer_p(reinterpret_cast<int*>(p),range<1>{size_p});

	buffer<char, 2> buffer_oristrings(reinterpret_cast<char*>(oristrings),range<2>{batch_size,LEN_INPUT});

	buffer<int, 1> buffer_lshnumber(lshnumber.data(),range<1>{lshnumber.size()});



	buffer<char, 1> buffer_embdata(embdata, range<1>{static_cast<unsigned int>(batch_size*NUM_STR*NUM_REP*len_output)});

	buffer<int,1> buffer_dict(dictionary,range<1>{256});

	buffer<int,1> buffer_len_oristrings(len_oristrings,range<1>(batch_size));


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



void create_bucket(queue &device_queue, char **embdata, vector<tuple<int,int,int,int,int>> &buckets, unsigned int batch_size, int* hash_lsh, vector<int> &a, vector<int> &lshnumber, uint32_t len_output){

	int dictionary[256]={0};
	inititalize_dictionary(dictionary);

    std::cout<<std::endl;
	std::cout<< "Selected: Create buckets"<<std::endl;
	cout << "\tTask: Buckets Generation\t";
	std::cout << "Device: " << device_queue.get_device().get_info<info::device::name>() << std::endl;
	
    {

//	buffer<char, 1> buffer_embeddata(embdata, range<1>{static_cast<unsigned int>(batch_size*NUM_STR*NUM_REP*LEN_OUTPUT)});

	buffer<tuple<int,int,int,int,int>> buffer_buckets(buckets.data(),range<1>{buckets.size()}); // Wrong dimension
	buffer<int, 1> buffer_a(a.data(),range<1>{a.size()});
	buffer<int,2> buffer_hash_lsh(hash_lsh,range<2>{NUM_HASH,NUM_BITS});
	buffer<int,1> buffer_dict(dictionary,range<1>{256});
	buffer<int, 1> buffer_lshnumber(lshnumber.data(),range<1>{lshnumber.size()});

	buffer<unsigned int, 1> buffer_batch_size(&batch_size,range<1>{1});

    buffer<uint32_t, 1> buffer_len_output(&len_output,range<1>{1});



	device_queue.submit([&](handler &cgh){


		//Executing kernel

//		auto acc_embeddata = buffer_embeddata.get_access<access::mode::read>(cgh);

		auto acc_buckets = buffer_buckets.get_access<access::mode::write>(cgh);
		auto acc_dict = buffer_dict.get_access<access::mode::read>(cgh);
		auto acc_hash_lsh = buffer_hash_lsh.get_access<access::mode::read>(cgh);

		auto acc_a = buffer_a.get_access<access::mode::read>(cgh);
		auto acc_lshnumber = buffer_lshnumber.get_access<access::mode::read>(cgh);

		auto acc_batch_size=buffer_batch_size.get_access<access::mode::read>(cgh);

        auto acc_len_output=buffer_len_output.get_access<access::mode::read>(cgh);

        
		cgh.parallel_for<class CreateBuckets>(range<3>{NUM_STRING,NUM_STR,NUM_REP}, [=](item<3> index){

			int i=index[0];
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
					digit=acc_lshnumber[acc_hash_lsh[k][j]];
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



void create_bucket_without_lshnumber_offset(queue &device_queue, char **embdata, vector<tuple<int,int,int,int,int>> &buckets, unsigned int batch_size, int* hash_lsh, vector<int> &a, vector<int> &lshnumber, uint32_t len_output){

	int dictionary[256]={0};
	inititalize_dictionary(dictionary);

	std::cout<<std::endl;
	std::cout<< "Selected: Create buckets - without lshnumber offset"<<std::endl;
	cout << "\tTask: Buckets Generation\t";


	std::cout << "Device: " << device_queue.get_device().get_info<info::device::name>() << std::endl;


    {

	buffer<tuple<int,int,int,int,int>> buffer_buckets(buckets.data(),range<1>{buckets.size()}); // Wrong dimension
	buffer<int, 1> buffer_a(a.data(),range<1>{a.size()});
	buffer<int,2> buffer_hash_lsh(hash_lsh,range<2>{NUM_HASH,NUM_BITS});
	buffer<int,1> buffer_dict(dictionary,range<1>{256});


	buffer<unsigned int, 1> buffer_batch_size(&batch_size,range<1>{1});

    buffer<uint32_t, 1> buffer_len_output(&len_output,range<1>{1});



	device_queue.submit([&](handler &cgh){


		//Executing kernel

//		auto acc_embeddata = buffer_embeddata.get_access<access::mode::read>(cgh);

		auto acc_buckets = buffer_buckets.get_access<access::mode::write>(cgh);
		auto acc_dict = buffer_dict.get_access<access::mode::read>(cgh);
		auto acc_hash_lsh = buffer_hash_lsh.get_access<access::mode::read>(cgh);

		auto acc_a = buffer_a.get_access<access::mode::read>(cgh);

		auto acc_batch_size=buffer_batch_size.get_access<access::mode::read>(cgh);

        auto acc_len_output=buffer_len_output.get_access<access::mode::read>(cgh);


		cgh.parallel_for<class CreateBuckets>(range<3>{NUM_STRING,NUM_STR,NUM_REP}, [=](item<3> index){

			int i=index[0];
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


void generate_candidates(queue &device_queue, vector<int> &len, char* oristrings, char **embdata, vector<tuple<int,int,int,int,int>> &buckets, unsigned int batch_size, vector<tuple<int,int>> &bucket_delimiter, vector<std::tuple<int,int,int,int,int,int>>& candidate, vector<int> &candidates_start, int * local_hash_lsh, vector<int> &lshnumber, uint32_t len_output){

    std::cout<<std::endl;
	cout << "Selected: Generate candidates"<< std::endl;
	cout << "\tTask: Candidate Pairs Generation\t";
	std::cout << "Device: " << device_queue.get_device().get_info<info::device::name>() << std::endl;


//	std::cout << "Max allocation size: " << device_queue.get_device().get_info<info::device::max_mem_alloc_size>() << std::endl;


	{

		buffer<int, 1> buffer_lshnumber(lshnumber.data(),range<1>{lshnumber.size()});

//		buffer<char,1> buffer_embeddata(reinterpret_cast<char*>(embdata), range<1>{static_cast<unsigned int>(batch_size*NUM_STR*NUM_REP*LEN_OUTPUT)});

		buffer<char,2> buffer_oristrings(oristrings,range<2>{NUM_STRING,LEN_INPUT});

		buffer<int, 1> buffer_candidate_start(candidates_start.data(),range<1>{candidates_start.size()});

		buffer<tuple<int,int,int,int,int>> buffer_buckets(buckets.data(),range<1>{buckets.size()});

		buffer<tuple<int,int>> buffer_delimiter(bucket_delimiter.data(),range<1>{bucket_delimiter.size()});

		cout << bucket_delimiter.size()<< " "<< candidate.size()<< std::endl;

		buffer<int, 2> buffer_hash_lsh(reinterpret_cast<int*>(local_hash_lsh),range<2>{NUM_HASH,NUM_BITS});

		buffer<tuple<int,int,int,int,int,int>> buffer_candidates(candidate.data(),range<1>{candidate.size()});

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

			auto acc_lshnumber = buffer_lshnumber.get_access<access::mode::read>(cgh);

            auto acc_batch_size=buffer_batch_size.get_access<access::mode::read>(cgh);

           

            auto acc_len_output=buffer_len_output.get_access<access::mode::read>(cgh);


            
			cgh.parallel_for<class GenerateCandidates>(range<1>{bucket_delimiter.size()}, [=](item<1> index){

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

void generate_candidates_without_lshnumber_offset(queue &device_queue, vector<int> &len, char* oristrings, char **embdata, vector<tuple<int,int,int,int,int>> &buckets, unsigned int batch_size, vector<tuple<int,int>> &bucket_delimiter, vector<std::tuple<int,int,int,int,int,int>>& candidate, vector<int> &candidates_start, int * local_hash_lsh, vector<int> &lshnumber, uint32_t len_output){

	std::cout<<std::endl;
		cout << "Selected: Generate candidates - without lshnumber offset"<< std::endl;
		cout << "\tTask: Candidate Pairs Generation\t";
		std::cout << "Device: " << device_queue.get_device().get_info<info::device::name>() << std::endl;

//	std::cout << "Max allocation size: " << device_queue.get_device().get_info<info::device::max_mem_alloc_size>() << std::endl;


	{


//		buffer<char,1> buffer_embeddata(reinterpret_cast<char*>(embdata), range<1>{static_cast<unsigned int>(batch_size*NUM_STR*NUM_REP*LEN_OUTPUT)});

		buffer<char,2> buffer_oristrings(oristrings,range<2>{NUM_STRING,LEN_INPUT});

		buffer<int, 1> buffer_candidate_start(candidates_start.data(),range<1>{candidates_start.size()});

		buffer<tuple<int,int,int,int,int>> buffer_buckets(buckets.data(),range<1>{buckets.size()});

		buffer<tuple<int,int>> buffer_delimiter(bucket_delimiter.data(),range<1>{bucket_delimiter.size()});

		cout << bucket_delimiter.size()<< " "<< candidate.size()<< std::endl;

		buffer<int, 2> buffer_hash_lsh(reinterpret_cast<int*>(local_hash_lsh),range<2>{NUM_HASH,NUM_BITS});

		buffer<tuple<int,int,int,int,int,int>> buffer_candidates(candidate.data(),range<1>{candidate.size()});

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



			cgh.parallel_for<class GenerateCandidates>(range<1>{bucket_delimiter.size()}, [=](item<1> index){

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

/*

void print_embedded( char *output, std::string filename=file_embed_strings ){

	ofstream outFile;

	outFile.open(filename, ios::out | ios::trunc);

	if (outFile.is_open()) {
		for(int i=0; i<NUM_STRING; i++){
				for(int j=0; j<NUM_STR; j++ ){
					for(int k=0; k<NUM_REP; k++){
						for(int t=0; t<LEN_OUTPUT; t++){

							if(output[ABSPOS(i,j,k,t)]==0){
								break;
							}
							outFile<<output[ABSPOS(i,j,k,t)];

							//outFile<<output[i][j][k][t];
						}
						outFile<<std::endl;

					}
				}
			}
	}
}*/

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





void initialize_candidate_pairs(std::vector<tuple<int,int>> &buckets_delimiter, vector<tuple<int,int,int,int,int>> &buckets, unsigned int batch_size, vector<std::tuple<int,int,int,int,int,int>> &candidates, vector<int> &candidates_start){


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


	start=std::chrono::system_clock::now();


	auto new_end=std::remove_if(buckets_delimiter.begin(),buckets_delimiter.end(),[](std::tuple<int,int> &e){return std::get<1>(e)<2;});

	buckets_delimiter.erase( new_end, buckets_delimiter.end());
	end=std::chrono::system_clock::now();

	std::cout<<"Time cand-init: remove element: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;


	start=std::chrono::system_clock::now();

	size=0;

	for(auto b=0; b<buckets_delimiter.size(); b++){

		candidates_start.emplace_back(size);
		int n=get<1>(buckets_delimiter[b]);
		size+=((n*(n-1))/2);

	}

	end=std::chrono::system_clock::now();

	std::cout<<"Time cand-init: count max pairs: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;


	start=std::chrono::system_clock::now();

	candidates.resize(size, tuple<int,int,int,int,int,int>(-1,-1,-1,-1,-1,-1));

	end=std::chrono::system_clock::now();

	std::cout<<"Time cand-init: resize: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;

//	for(int i=0; i<size; i++){
//		candidates.emplace_back();
//		std::get<0>(candidates[i])=-1;
//	}

}




void gathering_wrapper(queue &device_queue, char** set_embdata_dev, int (*hash_lsh)[NUM_BITS], unsigned int batch_size, uint32_t n_batches, vector<int> &lshnumber, uint32_t &len_output){



	char **set_embdata_reduced_dev=(char**)malloc_shared(n_batches*sizeof(char*), device_queue.get_device(), device_queue.get_context());

	for(int n=0; n<n_batches; n++){

		uint32_t new_size=lshnumber.size();

		set_embdata_reduced_dev[n]=(char*)malloc(batch_size*NUM_STR*NUM_REP*lshnumber.size()*sizeof(char));


		for(int id=0; id<batch_size; id++){

			for(int l=0; l<NUM_STR; l++ ){

				for(int k=0; k<NUM_REP; k++){

					int digit=0;

					for(int j=0; j<len_output; j++){

						while (digit < lshnumber.size() && j > lshnumber[digit]){

							set_embdata_reduced_dev[n][ABSPOS(id,l,k,digit,lshnumber.size())] = set_embdata_dev[n][ABSPOS(id,l,k,j,len_output)];
							digit++;
						}
					}
				}
			}
		}

		free(set_embdata_dev[n], device_queue.get_context());

		set_embdata_dev[n]=(char*)malloc_shared(batch_size*NUM_STR*NUM_REP*lshnumber.size()*sizeof(char), device_queue.get_device(), device_queue.get_context());

		memcpy(set_embdata_dev[n], set_embdata_reduced_dev[n], lshnumber.size());

	}



	len_output=lshnumber.size();
	set_embdata_dev=set_embdata_reduced_dev;
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


void parallel_embedding_batched_2dev_wrapper(queue &cpu_queue, queue &gpu_queue, vector<int> &len_oristrings, char (*oristrings)[LEN_INPUT], char** set_embdata_dev, unsigned int batch_size,uint32_t n_batches, vector<int> &lshnumber, uint32_t &len_output){

	std::cout<< "Selected: Parallel embedding - batched version"<<std::endl;


	// DICTIONARY

	int dictionary[256]={0};
	inititalize_dictionary(dictionary);


	for(int n=0; n<n_batches/2; n++){
		set_embdata_dev[n]=(char*)malloc_shared(batch_size*NUM_STR*NUM_REP*len_output*sizeof(char), cpu_queue.get_device(), cpu_queue.get_context());
	}

	for(int n=n_batches/2; n<n_batches; n++){
			set_embdata_dev[n]=(char*)malloc_shared(batch_size*NUM_STR*NUM_REP*len_output*sizeof(char), gpu_queue.get_device(), gpu_queue.get_context());
	}

	uint32_t len_p=samplingrange+1;

	int (*p)=new int[NUM_STR*NUM_CHAR*len_p];

	generate_random_string(p, len_p);

	for(int n=0; n<n_batches/2; n++){
		parallel_embedding_batched(cpu_queue, len_oristrings.data()+n*batch_size, (char*) oristrings[n*batch_size], (char*)set_embdata_dev[n], batch_size, lshnumber, p, len_p, len_output, dictionary);
	}

	for(int n=n_batches/2; n<n_batches; n++){
		parallel_embedding_batched(gpu_queue, len_oristrings.data()+n*batch_size, (char*) oristrings[n*batch_size], (char*)set_embdata_dev[n], batch_size, lshnumber, p, len_p, len_output, dictionary);
	}

	delete[] p;
}

void parallel_embedding_while_loop_wrapper(queue &device_queue, vector<int> &len_oristrings, char (*oristrings)[LEN_INPUT], char** set_embdata_dev, unsigned int batch_size,uint32_t n_batches, vector<int> &lshnumber, uint32_t &len_output){

	std::cout<< "Selected: Parallel embedding - while loop"<<std::endl;

	int dictionary[256]={0};
	inititalize_dictionary(dictionary);

	len_output=lshnumber.size();

	for(int n=0; n<n_batches; n++){
		set_embdata_dev[n]=(char*)malloc_shared(batch_size*NUM_STR*NUM_REP*len_output*sizeof(char), device_queue.get_device(), device_queue.get_context());
	}

	uint32_t len_p=samplingrange+1;

	int (*p)=new int[NUM_STR*NUM_CHAR*len_p];

	generate_random_string(p, len_p);

	for(int n=0; n<n_batches; n++){
		parallel_embedding_while_loop(device_queue, len_oristrings.data()+n*batch_size, (char*) oristrings[n*batch_size], (char*)set_embdata_dev[n], batch_size, lshnumber, p, len_p, lshnumber.size(), dictionary);
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



//#if GPU


	cpu_selector device_selector1;

	gpu_selector device_selector2;

	queue cpu_queue(device_selector1, asyncHandler, property::queue::in_order());

	queue gpu_queue(device_selector2, asyncHandler, property::queue::in_order());

	queue device_queue;

	if(device==0){
		device_queue=gpu_queue;
	}
	else{
		device_queue=cpu_queue;
	}

    char **set_embdata_dev=(char**)malloc_shared(n_batches*sizeof(char*), device_queue.get_device(), device_queue.get_context());


	 /**
	  *
	  * FOR BENCHMARK
	  *
	  * **/


	 std::vector<void(*)(queue&, vector<int>&, char(*)[LEN_INPUT], char** , unsigned int , uint32_t , vector<int>&, uint32_t&)> f_emb;
	 std::vector<void(*)(queue&, char**, vector<tuple<int,int,int,int,int>>&, unsigned int, int* , vector<int>&, vector<int>&, uint32_t)> f_bucket;
	 std::vector<void(*)(queue&, vector<int>&, char*, char**, vector<tuple<int,int,int,int,int>>&, unsigned int, vector<tuple<int,int>>&, vector<std::tuple<int,int,int,int,int,int>>&, vector<int>&, int*, vector<int>&, uint32_t)> f_cand;


	 void (*f_parallel_embedding_batched_wrapper)(queue&, vector<int>&, char (*)[LEN_INPUT] , char** , unsigned int , uint32_t , vector<int>&, uint32_t&){parallel_embedding_batched_wrapper};
	 void (*f_parallel_embedding_USM_wrapper)(queue&, vector<int>&, char (*)[LEN_INPUT], char** , unsigned int , uint32_t , vector<int>&, uint32_t&){parallel_embedding_USM_wrapper};
	 void (*f_parallel_embedding_while_loop)(queue&, vector<int>&, char(*)[LEN_INPUT], char** , unsigned int , uint32_t , vector<int>&, uint32_t&){parallel_embedding_while_loop_wrapper};


//

	 void(*f_create_buckets)(queue&, char**, vector<tuple<int,int,int,int,int>>&, unsigned int, int* , vector<int>&, vector<int>&, uint32_t){create_bucket};

	 void(*f_create_buckets_without_offset)(queue&, char**, vector<tuple<int,int,int,int,int>>&, unsigned int, int* , vector<int>&, vector<int>&, uint32_t){create_bucket_without_lshnumber_offset};


	 void(*f_generate_candidates)(queue&, vector<int>&, char*, char**, vector<tuple<int,int,int,int,int>>&, unsigned int, vector<tuple<int,int>>&, vector<std::tuple<int,int,int,int,int,int>>&, vector<int>&, int*, vector<int>&, uint32_t){generate_candidates};

	 void(*f_generate_candidates_without_offset)(queue&, vector<int>&, char*, char**, vector<tuple<int,int,int,int,int>>&, unsigned int, vector<tuple<int,int>>&, vector<std::tuple<int,int,int,int,int,int>>&, vector<int>&, int*, vector<int>&, uint32_t){generate_candidates_without_lshnumber_offset};


	 f_emb.push_back(f_parallel_embedding_batched_wrapper); // 0
	 f_emb.push_back(f_parallel_embedding_USM_wrapper); // 1
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


    f_emb[alg_number[0]](device_queue, len_oristrings, oristrings, set_embdata_dev, batch, n_batches, lshnumber, len_output);


    device_queue.wait();

	end=std::chrono::system_clock::now();

	time_embedding_data=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();


#if PRINT_EACH_STEP
	print_embedded(embdata);
#endif


	start_tot=std::chrono::system_clock::now();


	/**
	 *
	 *
	 * CREATE BUCKETS STEP
	 *
	 *
	 * **/



	start=std::chrono::system_clock::now();

	f_bucket[alg_number[1]](device_queue, (char**)set_embdata_dev, buckets, batch, (int*)hash_lsh, a, lshnumber, len_output);

	end=std::chrono::system_clock::now();

	time_create_buckets=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();


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

	 initialize_candidate_pairs(buckets_delimiter, buckets, batch, candidates, candidates_start);

	 end=std::chrono::system_clock::now();

	 time_candidate_initialization=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();



	 /**
	 *
	 *
	 * GENERATE CANDIDATE PAIRS STEP
	 *
	 *
	 * **/


	 start=std::chrono::system_clock::now();

	 f_cand[alg_number[2]](device_queue, len_oristrings, (char*)oristrings, (char**)set_embdata_dev, buckets, batch, buckets_delimiter, candidates, candidates_start, (int *)hash_lsh, lshnumber, len_output);;


	 end=std::chrono::system_clock::now();

	 time_generate_candidates=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

	 start=std::chrono::system_clock::now();

	 vector<std::tuple<int,int>> verifycan;


	 candidates.erase(remove_if(candidates.begin(), candidates.end(),[](std::tuple<int,int,int,int,int,int> e){return (get<0>(e)==-1 || get<4>(e)>K_INPUT || get<5>(e)!=0 || get<0>(e)==get<2>(e));}), candidates.end());

	 auto start_2=std::chrono::system_clock::now();

//	 sort(candidates.begin(),candidates.end());

//	 auto policy_sort = make_device_policy<class PolicySort>(device_queue);


	 tbb::parallel_sort(candidates.begin(),candidates.end(), [](std::tuple<int,int,int,int,int,int> e1, std::tuple<int,int,int,int,int,int> e2) {
		 return ( ( get<0>(e1)<get<0>(e2) ) ||
				 ( get<0>(e1)==get<0>(e2) && get<1>(e1)<get<1>(e2) ) ||
				 ( get<0>(e1)==get<0>(e2) && get<1>(e1)==get<1>(e2) && get<2>(e1)<get<2>(e2) )  ||
				 ( get<0>(e1)==get<0>(e2) && get<1>(e1)==get<1>(e2) && get<2>(e1)==get<2>(e2) && get<3>(e1)<get<3>(e2) ) ||
				 ( get<0>(e1)==get<0>(e2) && get<1>(e1)==get<1>(e2) && get<2>(e1)==get<2>(e2) && get<3>(e1)==get<3>(e2) && get<4>(e1)<get<4>(e2) ) ||
				 ( get<0>(e1)==get<0>(e2) && get<1>(e1)==get<1>(e2) && get<2>(e1)==get<2>(e2) && get<3>(e1)==get<3>(e2) && get<4>(e1)==get<4>(e2) && get<5>(e1)<get<5>(e2) ) ); } );

	 auto end_2=std::chrono::system_clock::now();

	 time_sorting_candidates=std::chrono::duration_cast<std::chrono::milliseconds>(end_2-start_2).count();

	 std::cerr<<"Candidate after filter out: "<<candidates.size()<<std::endl;


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
	std::cout<<"Uniq: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end_2-start_2).count()/1000<<std::endl;

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

	start=std::chrono::system_clock::now();


	/**
	 *
	 * EDIT DISTANCE CALCULATION
	 *
	 * */


	int first_str;
	int second_str;
	int j=0;

	for ( j = 0; j < verifycan.size(); j++){
		num_candidate++;
		first_str=get<0>(verifycan[j]);
		second_str=get<1>(verifycan[j]);
		int ed = edit_distance(oridata_modified[second_str].data(), len_oristrings[second_str]/*tmp_oridata[second_str].size()*/,oridata_modified[first_str].data(), len_oristrings[first_str] /*tmp_oridata[first_str].size()*/, K_INPUT);

		if(ed != -1) {
			outputs.push_back(make_tuple(first_str, second_str));
		}

	}

	end=std::chrono::system_clock::now();
	time_edit_distance=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

	end_tot=std::chrono::system_clock::now();

	total_time_join=std::chrono::duration_cast<std::chrono::milliseconds>(end_tot-start_tot).count();

	end_alg=std::chrono::system_clock::now();
	total_time=std::chrono::duration_cast<std::chrono::milliseconds>(end_alg-start_alg).count();


	delete[] hash_lsh;
	delete[] oristrings;

	for(int i=0; i<n_batches; i++){
		free(set_embdata_dev[i], device_queue.get_context());
	}
	free(set_embdata_dev, device_queue.get_context());




	std::string distinguisher="";

	if(device==0){
		distinguisher+="-GPU-";
	}else if(device==1){
		distinguisher+="-CPU-";
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

		std::string dev=device_queue.get_device().get_info<info::device::name>();
			if (outFile.is_open()) {

				//outFile<</*get<4>(candidates[i])<<", "<<get<5>(candidates[i])<<", "<<*/get<0>(candidates[i])<<", "<<get<1>(candidates[i])<<", "<<get<2>(candidates[i])<<", "<<get<3>(candidates[i])<<std::endl;
				outFile<<"Step,Time(sec),Device"<<std::endl;
				outFile<<"Read Data,"<<(float)time_init/1000<<std::endl;
				outFile<<"Embedding,"<<(float)time_embedding_data/1000<<","<<dev<<std::endl;
				outFile<<"Create Buckets,"<< (float)time_create_buckets/1000<<","<<dev<<std::endl;
				outFile<<"Sort Buckets,"<< (float)time_sorting_buckets/1000<<std::endl;
				outFile<<"Candidate Initialization,"<<(float)time_candidate_initialization/1000<<std::endl;
				outFile<<"Generate Candidate,"<< (float)time_generate_candidates/1000<<","<<dev<<std::endl;
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

	return 0;

}


