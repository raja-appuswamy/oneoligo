#include <CL/sycl.hpp>
//#include <ittnotify.h>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>
#include "tbb/parallel_sort.h"
#include <exception>
#include <optional>

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
#include <thread>
#include <mutex>
#include <atomic>
#include <memory>

#include "tbb/global_control.h"
#include "Time.cpp"


using namespace cl::sycl;
using namespace oneapi::std;
using namespace std;




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


#ifndef PRINT_EMB
	#define PRINT_EMB 0
#endif

#ifndef PRINT_BUCK
	#define PRINT_BUCK 0
#endif

#ifndef PRINT_CAND
	#define PRINT_CAND 0
#endif

//embedrange: the length of truncation, recommended to be the average length of strings (you could try smaller values to further save the embedding time)




int samplingrange=5000; //the maximum digit to embed, the range to sample

int countfilter=1;// Number of required matches (>T) for a pair of substrings to be considered as candidate




int test_batches=2;

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


#ifndef OLD_VERSION
	#define OLD_VERSION 0
#endif


//TODO Remove
class arrayWrapper{

public:
	size_t size;
	size_t offset; //In number of element

	arrayWrapper(size_t size, size_t offset){
		this->size=size;
		this->offset=offset;
	}



};



Time timer;



std::vector<int> indices;

std::vector<idpair> outputs;

std::vector<std::string> tmp_oridata;



void setuplsh(int (*hash_lsh)[NUM_BITS], std::vector<int> &a, std::vector<int> &lshnumber)
{

	for (int i = 0; i < NUM_HASH; i++)
	for (int j = 0; j < NUM_BITS; j++)
		hash_lsh[i][j] = rand() % (samplingrange);


	for (int i = 0; i < NUM_BITS; i++)
		a.push_back(rand() % (M - 1));


	for (int i = 0; i < NUM_HASH; i++){
		for(int j=0; j < NUM_BITS; j++){
			lshnumber.emplace_back(hash_lsh[i][j]);
		}
	}

	tbb::parallel_sort(lshnumber.begin(), lshnumber.end());
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
	std::cout<<std::endl;
}




void readdata(std::vector<int> &len_oristrings, char (*oristrings)[LEN_INPUT], std::vector<std::string> &oridata_modified )
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



void initialization( std::vector<int> &len_oristrings, char (*oristrings)[LEN_INPUT], std::vector<string> &oridata_modified, int (*hash_lsh)[NUM_BITS], std::vector<int> &a, std::vector<int> &lshnumber )
{



	//ORIGINAL VERSION

	timer.start_time(0,0,1);

	readdata(len_oristrings, oristrings, oridata_modified);

	timer.end_time(0,0,1);



	timer.start_time(0,0,2);

	setuplsh(hash_lsh, a, lshnumber);

	timer.end_time(0,0,2);


}

void inititalize_dictionary(uint8_t* dictionary){

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



void parallel_embedding_while_loop_gath(queue &device_queue, buffer<int,1> &buffer_len_oristrings, buffer<char,2> &buffer_oristrings, buffer<char,1> &buffer_embdata, uint32_t batch_size, buffer<int,1> &buffer_lshnumber, buffer<int,1> &buffer_p, buffer<uint32_t,1> &buffer_len_output, buffer<uint32_t,1> &buffer_samplingrange, buffer<uint8_t,1> &buffer_dict, buffer<std::tuple<int,int>> &buffer_rev_hash){


		std::cout << "\n\t\tTask: Embedding Data";
		std::cout << "\tDevice: " << device_queue.get_device().get_info<info::device::name>() << std::endl;

	device_queue.submit([&](handler &cgh){


		  auto acc_oristrings = buffer_oristrings.get_access<access::mode::read>(cgh);
		  auto acc_lshnumber = buffer_lshnumber.get_access<access::mode::read,access::target::constant_buffer>(cgh);
		  auto acc_embdata = buffer_embdata.get_access<access::mode::write>(cgh);
		  auto acc_dict = buffer_dict.get_access<access::mode::read>(cgh);
		  auto acc_samplingrange = buffer_samplingrange.get_access<access::mode::read>(cgh);
		  auto acc_len_oristrings = buffer_len_oristrings.get_access<access::mode::read>(cgh);
		  auto acc_p=buffer_p.get_access<access::mode::read>(cgh);

          auto acc_len_output=buffer_len_output.get_access<access::mode::read>(cgh);

          auto acc_rev_hash=buffer_rev_hash.get_access<access::mode::read>(cgh);


          std::cout<<"\t\t\tBatch size: "<<batch_size<<std::endl;
          std::cout<<"\t\t\tRange: ("<<batch_size<<", "<<NUM_STR<<", "<<NUM_REP<<")"<<std::endl;



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


//					  acc_embdata[ABSPOS(id,l,k,partdigit,acc_len_output[0])]=s;


					  acc_embdata[ABSPOS(id,l,k,std::get<0>(acc_rev_hash[partdigit]),acc_len_output[0])]=s;



					  int next=std::get<1>(acc_rev_hash[partdigit]);

					  while(next!=-1){
						  acc_embdata[ABSPOS(id,l,k,std::get<0>(acc_rev_hash[next]),acc_len_output[0])]=s;
						  next=get<1>(acc_rev_hash[next]);
					  }


					  partdigit++;


				 }
			  }

		  });

		});

}

void parallel_embedding_while_loop(queue &device_queue, buffer<int,1> &buffer_len_oristrings, buffer<char,2> &buffer_oristrings, buffer<char,1> &buffer_embdata, uint32_t batch_size, buffer<int,1> &buffer_lshnumber, buffer<int,1> &buffer_p, buffer<uint32_t,1> &buffer_len_output, buffer<uint32_t,1> &buffer_samplingrange, buffer<uint8_t,1> &buffer_dict, buffer<tuple<int,int>> &buffer_rev_hash){


		cout << "\n\t\tTask: Embedding Data";
		cout << "\tDevice: " << device_queue.get_device().get_info<info::device::name>() << std::endl;

		device_queue.submit([&](handler &cgh){


		  auto acc_oristrings = buffer_oristrings.get_access<access::mode::read>(cgh);
		  auto acc_lshnumber = buffer_lshnumber.get_access<access::mode::read,access::target::constant_buffer>(cgh);
		  auto acc_embdata = buffer_embdata.get_access<access::mode::write>(cgh);
		  auto acc_dict = buffer_dict.get_access<access::mode::read>(cgh);
		  auto acc_samplingrange = buffer_samplingrange.get_access<access::mode::read>(cgh);
		  auto acc_len_oristrings = buffer_len_oristrings.get_access<access::mode::read>(cgh);
		  auto acc_p=buffer_p.get_access<access::mode::read>(cgh);

          auto acc_len_output=buffer_len_output.get_access<access::mode::read>(cgh);

          auto acc_rev_hash=buffer_rev_hash.get_access<access::mode::read>(cgh);


          std::cout<<"\t\t\tBatch size: "<<batch_size<<std::endl;
          std::cout<<"\t\t\tRange: ("<<batch_size<<", "<<NUM_STR<<", "<<NUM_REP<<")"<<std::endl;



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


					  acc_embdata[ABSPOS(id,l,k,partdigit,acc_len_output[0])]=s;


					  partdigit++;

				 }
			  }

		  });

		});

}




void parallel_embeddingUSM(queue &device_queue, int* len_oristrings, char *oristrings, char** embdata, unsigned int batch_size, std::vector<int> &lshnumber, int *p, int len_p, uint32_t len_output, uint8_t* dictionary){



	unsigned int size_p=NUM_STR*NUM_CHAR*(samplingrange+1);

	buffer<int,1> buffer_p(reinterpret_cast<int*>(p),range<1>{size_p});

	buffer<char, 2> buffer_oristrings(reinterpret_cast<char*>(oristrings),range<2>{NUM_STRING,LEN_INPUT});

	buffer<int, 1> buffer_lshnumber(lshnumber.data(),range<1>{lshnumber.size()});


	buffer<uint8_t,1> buffer_dict(dictionary,range<1>{256});
	buffer<int,1> buffer_len_oristrings(len_oristrings,range<1>(NUM_STRING));

	uint32_t samprange=samplingrange;
	buffer<uint32_t,1> buffer_samplingrange(&samprange,range<1>(1));
    
    buffer<unsigned int, 1> buffer_batch_size(&batch_size,range<1>{1});
    
    buffer<uint32_t, 1> buffer_len_output(&len_output,range<1>{1});



	cout << "\tTask: Embedding Data";
	cout << "\tDevice: " << device_queue.get_device().get_info<info::device::name>() << std::endl;

	cout<<"size_p"<<size_p<<std::endl;
	cout<<"samp_range"<<samprange<<std::endl;
	cout<<"bacth_size"<<batch_size<<std::endl;
	cout<<"len_output"<<len_output<<std::endl;

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



void sequential_embedding_if(queue &device_queue, int* len_oristrings, char *oristrings, char* embdata, unsigned int batch_size, std::vector<int> &lshnumber, int *p, int len_p, uint32_t len_output, uint8_t* dictionary){

			cout << "\tTask: Embedding Data\t";
			std::cout << "Device: " << device_queue.get_device().get_info<info::device::name>() << std::endl;
			std::cout<<"\n\tLen output: "<<len_output<<std::endl;

		unsigned int size_p=NUM_STR*NUM_CHAR*(samplingrange+1);

		buffer<int,1> buffer_p(reinterpret_cast<int*>(p),range<1>{size_p}, {property::buffer::use_host_ptr()});

		buffer<char, 2> buffer_oristrings(reinterpret_cast<char*>(oristrings),range<2>{batch_size,LEN_INPUT}, {property::buffer::use_host_ptr()});

		buffer<int, 1> buffer_lshnumber(lshnumber.data(),range<1>{lshnumber.size()});

		buffer<char, 1> buffer_embdata(embdata, range<1>{static_cast<unsigned int>(batch_size*NUM_STR*NUM_REP*len_output)}, {property::buffer::use_host_ptr()});

		buffer<uint8_t,1> buffer_dict(dictionary,range<1>{256});

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


				int increment=0;
				int bidx=0;
				int oidx=0;
				int saved_idx=0;
				int take_it=0;

					for (int oidx = 0; i < size && oidx <= acc_samplingrange[0] && bidx<acc_lshnumber.get_range()[0]; oidx++){

						char s = acc_oristrings[id][i];

						r=acc_dict[s];

						i += (acc_p[ABSPOS_P(l,r,oidx,len)] ? 0:1);

//						while (partdigit < lshnumber.size() && j > lshnumber[partdigit]){

						if(oidx==acc_lshnumber[bidx]){
							acc_embdata[ABSPOS(id,l,k,saved_idx,acc_len_output[0])]=s;
							saved_idx++;
							bidx++;
						}

//						increment=(oidx==acc_lshnumber[bidx]?1:0);
//						+=increment;
//						bidx+=increment;

//						oidx++;



//						partdigit++;


				}





			  });

		});


}


void parallel_embeddingUSM_while(queue &device_queue, int* len_oristrings, char *oristrings, char** embdata, unsigned int batch_size, std::vector<int> &lshnumber, int *p, int len_p, uint32_t len_output, uint8_t* dictionary){


    std::cout<<std::endl;
	std::cout << "Device: " << device_queue.get_device().get_info<info::device::name>() << std::endl;

//	tuple<int,int,int,int>* tmp=(tuple<int,int,int,int>*)malloc_shared<tuple<int,int,int,int>>(batch_size*NUM_STR*NUM_REP, device_queue);


	unsigned int size_p=NUM_STR*NUM_CHAR*(samplingrange+1);

	buffer<int,1> buffer_p(reinterpret_cast<int*>(p),range<1>{size_p});

	buffer<char, 2> buffer_oristrings(reinterpret_cast<char*>(oristrings),range<2>{NUM_STRING,LEN_INPUT});

	buffer<int, 1> buffer_lshnumber(lshnumber.data(),range<1>{lshnumber.size()});


	buffer<uint8_t,1> buffer_dict(dictionary,range<1>{256});
	buffer<int,1> buffer_len_oristrings(len_oristrings,range<1>(NUM_STRING));

	uint32_t samprange=samplingrange;
	buffer<uint32_t,1> buffer_samplingrange(&samprange,range<1>(1));

    buffer<unsigned int, 1> buffer_batch_size(&batch_size,range<1>{1});

    buffer<uint32_t, 1> buffer_len_output(&len_output,range<1>{1});



	cout << "\tTask: Embedding Data"<< std::endl;

	cout<<"size_p"<<size_p<<std::endl;
		cout<<"samp_range"<<samprange<<std::endl;
		cout<<"bacth_size"<<batch_size<<std::endl;
		cout<<"len_output"<<len_output<<std::endl;


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
		  cgh.parallel_for<class EmbedString>(range<3>{NUM_STRING,NUM_STR,NUM_REP}, [=](item<3> index){


			  int id = index[0];
			  int l = index[1];
			  int k = index[2];


			  int partdigit = 0;
			  int size = acc_len_oristrings[id];

			  int r=0;

			  int len_out=acc_lshnumber.get_range()[0];

			  int i = SHIFT*k;

			  int len=acc_samplingrange[0]+1;

//			  get<0>(tmp[index.get_linear_id()])=id;
//			  get<1>(tmp[index.get_linear_id()])=l;
//			  get<2>(tmp[index.get_linear_id()])=k;
//			  get<3>(tmp[index.get_linear_id()])=ABSPOS(id,l,k,0,acc_len_output[0]);





			  for (int j = 0; i < size && j <= acc_samplingrange[0]; i++){

				  char s = acc_oristrings[id][i];

				  r=acc_dict[s];

				  j += ( acc_p[ABSPOS_P(l,r,j,len)] + 1 );

				  while (partdigit < acc_lshnumber.get_range()[0] && j > acc_lshnumber[partdigit]){


					  embdata[(int)(id/acc_batch_size[0])][ABSPOS((int)(id%acc_batch_size[0]),l,k,partdigit,acc_len_output[0])]=s;
					  partdigit++;


				 }
			  }

		  });

		});

	// ensure any asynchronous exceptions caught are handled before proceeding
	device_queue.wait_and_throw();

}


void parallel_embedding_batched(queue &device_queue, int* len_oristrings, char *oristrings, char* embdata, unsigned int batch_size, std::vector<int> &lshnumber, int *p, int len_p, uint32_t len_output, uint8_t* dictionary){


		cout << "\tTask: Embedding Data\t";
		std::cout << "Device: " << device_queue.get_device().get_info<info::device::name>() << std::endl;
		std::cout<<"\n\tLen output: "<<len_output<<std::endl;

	unsigned int size_p=NUM_STR*NUM_CHAR*(samplingrange+1);

	buffer<int,1> buffer_p(reinterpret_cast<int*>(p),range<1>{size_p}, {property::buffer::use_host_ptr()});

	buffer<char, 2> buffer_oristrings(reinterpret_cast<char*>(oristrings),range<2>{batch_size,LEN_INPUT}, {property::buffer::use_host_ptr()});

	buffer<int, 1> buffer_lshnumber(lshnumber.data(),range<1>{lshnumber.size()});

	buffer<char, 1> buffer_embdata(embdata, range<1>{static_cast<unsigned int>(batch_size*NUM_STR*NUM_REP*len_output)}, {property::buffer::use_host_ptr()});

	buffer<uint8_t,1> buffer_dict(dictionary,range<1>{256});

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


// To use when call embedding WITH while loop.
void create_bucket_without_lshnumber_offset(queue &device_queue, char **embdata, buffer<std::tuple<int,int,int,int,int>,1> &buffer_buckets, buffer<uint32_t,1> &buffer_batch_size, uint32_t split_size, buffer<uint32_t,1> &buffer_split_offset, buffer<uint32_t,2> &buffer_hash_lsh, buffer<uint32_t,1> &buffer_a, buffer<uint32_t,1> &buffer_len_output, buffer<uint8_t,1> &buffer_dict){




		std::cout << "\n\tTask: Buckets Generation\t";
		std::cout << "Device: " << device_queue.get_device().get_info<info::device::name>() << std::endl;

		std::cout<<"\t\tSplit size: "<<split_size<<std::endl;
		std::cout<<"\t\tRange: "<<"("<<split_size<<", "<<NUM_STR<<", "<<NUM_HASH*NUM_REP<<")"<<std::endl;


    {


	    auto start=std::chrono::system_clock::now();


		device_queue.submit([&](handler &cgh){


		//Executing kernel


			auto acc_buckets = buffer_buckets.get_access<access::mode::write>(cgh);
			auto acc_dict = buffer_dict.get_access<access::mode::read>(cgh);
			auto acc_hash_lsh = buffer_hash_lsh.get_access<access::mode::read>(cgh);

			auto acc_a = buffer_a.get_access<access::mode::read>(cgh);

			auto acc_batch_size=buffer_batch_size.get_access<access::mode::read>(cgh);

			auto acc_len_output=buffer_len_output.get_access<access::mode::read>(cgh);

	        auto acc_split_offset=buffer_split_offset.get_access<access::mode::read>(cgh);




			auto start=std::chrono::system_clock::now();





			cgh.parallel_for<class CreateBuckets>(range<3>{split_size,NUM_STR,NUM_REP*NUM_HASH}, [=](item<3> index){

				int i=index[0]+acc_split_offset[0];
				int t=index[1];


				int output_position;

				size_t linear_id=index.get_linear_id();

				int id=0;
				char dict_index=0;
				int id_mod=0;
				int digit=-1;

				int kq=index[2];


				int k=kq/NUM_REP;
				int q=kq%NUM_REP;


					id = 0;
					id_mod=0;


					for (int j = 0; j < NUM_BITS; j++){

						digit=acc_hash_lsh[k][j];
						dict_index=embdata[(int)(i/acc_batch_size[0])][ABSPOS((int)(i%acc_batch_size[0]),t,q,digit,acc_len_output[0])];
						if ( dict_index!=0 ) {

							id += (acc_dict[dict_index]) * acc_a[j];
						}

					}

					id_mod=id % M;


					output_position=linear_id;


					get<0>(acc_buckets[output_position])=t;
					get<1>(acc_buckets[output_position])=k;
					get<2>(acc_buckets[output_position])=id_mod;
					get<3>(acc_buckets[output_position])=i;
					get<4>(acc_buckets[output_position])=q;

			//				}

			});
		});

		auto end=std::chrono::system_clock::now();

		cout<<"Submission duration: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()<<"\n";

	}

}


void create_bucket_without_lshnumber_offset_NEW_impr(queue &device_queue, char **embdata, buffer<std::tuple<int,int,int,int,int>,1> &buffer_buckets, buffer<uint32_t,1> &buffer_batch_size, uint32_t split_size, buffer<uint32_t,1> &buffer_split_offset, buffer<uint32_t,2> &buffer_hash_lsh, buffer<uint32_t,1> &buffer_a, buffer<uint32_t,1> &buffer_len_output, buffer<uint8_t,1> &buffer_dict){

		std::cout << "\n\tTask: Buckets Generation\t";
		std::cout << "Device: " << device_queue.get_device().get_info<info::device::name>() << std::endl;

		std::cout<<"\t\tSplit size: "<<split_size<<std::endl;

		range<2> glob_range(split_size*NUM_STR*NUM_REP,NUM_HASH);
		range<3> local_range(250,1,1);

		std::cout<<"\t\tGlobal range: "<<"("<<glob_range[0]<<", "<<glob_range[1]<<")"<<std::endl;


    {


	    auto start=std::chrono::system_clock::now();


		device_queue.submit([&](handler &cgh){


		//Executing kernel


			auto acc_buckets = buffer_buckets.get_access<access::mode::write>(cgh);
			auto acc_dict = buffer_dict.get_access<access::mode::read>(cgh);
			auto acc_hash_lsh = buffer_hash_lsh.get_access<access::mode::read>(cgh);

			auto acc_a = buffer_a.get_access<access::mode::read>(cgh);

			auto acc_batch_size=buffer_batch_size.get_access<access::mode::read>(cgh);

			auto acc_len_output=buffer_len_output.get_access<access::mode::read>(cgh);

	        auto acc_split_offset=buffer_split_offset.get_access<access::mode::read>(cgh);




			auto start=std::chrono::system_clock::now();


			cgh.parallel_for<class CreateBuckets>(range<2>(glob_range), [=](item<2> index){

				int itq=index[0];//.get_global_id(0);

				int i=itq/(NUM_STR*NUM_REP)+acc_split_offset[0];

				int tq=itq%(NUM_STR*NUM_REP);

				int t=tq/NUM_REP;
				int q=tq%NUM_REP;

				int k=index[1];//.get_global_id(1);


				int output_position;

				size_t linear_id=index.get_linear_id();//.get_global_linear_id();

				int id=0;
				char dict_index=0;
				int id_mod=0;
				int digit=-1;


				id = 0;
				id_mod=0;

				for (int j = 0; j < NUM_BITS; j++){

					digit=k*NUM_BITS+j;
					dict_index=embdata[(int)(i/acc_batch_size[0])][ABSPOS((int)(i%acc_batch_size[0]),t,q,digit,acc_len_output[0])];

						id += (acc_dict[dict_index]) * acc_a[j];

				}

				id_mod=id % M;


				output_position=linear_id;


				get<0>(acc_buckets[output_position])=t;
				get<1>(acc_buckets[output_position])=k;
				get<2>(acc_buckets[output_position])=id_mod;
				get<3>(acc_buckets[output_position])=i;
				get<4>(acc_buckets[output_position])=q;


			});
		});

		auto end=std::chrono::system_clock::now();

		cout<<"Submission duration: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()<<"\n";

	}

}



void create_buckets_without_lshnumber_offset_2dev_NEW_wrapper(vector<queue> &queues, char **embdata, vector<tuple<int,int,int,int,int>> &buckets, uint32_t n_batches, uint32_t batch_size, int* hash_lsh, vector<int> &a, vector<int> &lshnumber, uint32_t len_output){

	std::cout<< "Selected: Create buckets - without lshnumber offset"<<std::endl;

	std::cout<<"Len output: "<<len_output<<std::endl;

	int num_dev=queues.size();


	int n_max;
	int n_min;


	int idx_max;
	int idx_min;

	int number_of_testing_batches=2*num_dev;//test_batches;


	vector<long> times;
	vector<vector<long>> time_on_dev(num_dev,vector<long>());


	{
		vector<uint32_t> split_size;//=9*batch_size;

		int thread_num=0;

		auto start_timeline=std::chrono::system_clock::now();


		uint8_t dictionary[256]={0};

		inititalize_dictionary(dictionary);


		vector<uint32_t> offset;

		vector<sycl::buffer<tuple<int,int,int,int,int>>> buffers_buckets;
		vector<sycl::buffer<uint32_t,1>> buffers_batch_size;
		vector<sycl::buffer<uint32_t,1>> buffers_split_size;
		vector<sycl::buffer<uint32_t,1>> buffers_split_offset;
		vector<sycl::buffer<uint32_t,2>> buffers_hash_lsh;
		vector<sycl::buffer<uint32_t,1>> buffers_a;
		vector<sycl::buffer<uint32_t,1>> buffers_lshnumber;
		vector<sycl::buffer<uint32_t,1>> buffers_len_output;
		vector<sycl::buffer<uint8_t,1>>  buffers_dict;


		timer.start_time(0,2,1);

		int n=0;

		int  dev=0;

		cout<<"\tStart profiling on devices..."<<std::endl<<std::endl;

		auto start=std::chrono::system_clock::now();


		for(auto &q:queues){


			for(int i=0; i<2; i++){

				auto start=std::chrono::system_clock::now();

				offset.emplace_back(2*batch_size*dev+i*batch_size);


				uint32_t loc_split_size=batch_size;

				cout<<"\n\tSet offset to: "<<offset[n]<<std::endl;

				buffers_buckets.emplace_back(sycl::buffer<tuple<int,int,int,int,int>,1>(static_cast<tuple<int,int,int,int,int>*>(buckets.data()+offset.back()*NUM_REP*NUM_HASH*NUM_STR),range<1>{loc_split_size*NUM_STR*NUM_HASH*NUM_REP}, {sycl::property::buffer::use_host_ptr()})); // Wrong dimension

				buffers_a.emplace_back(buffer<uint32_t,1>((uint32_t*)a.data(),range<1>{a.size()}));

				buffers_hash_lsh.emplace_back(buffer<uint32_t,2>((uint32_t*)hash_lsh, range<2>{NUM_HASH,NUM_BITS}));

				buffers_dict.emplace_back(buffer<uint8_t,1>(dictionary,range<1>{256}));

				buffers_batch_size.emplace_back(buffer<uint32_t,1>(&batch_size, range<1>{1}));

				buffers_len_output.emplace_back(buffer<uint32_t,1>(&len_output, range<1>{1}));

				buffers_split_offset.emplace_back(buffer<uint32_t,1> (&offset[n], range<1>{1}));


				// n*split offset w.r.t. the batches processed by the other device (ex. 3 batches)
				// Inside these batches there is another offset, that is iter*batch_size


				// The first one is longer then others
				create_bucket_without_lshnumber_offset_NEW_impr(q, embdata, buffers_buckets[n], buffers_batch_size[n], loc_split_size, buffers_split_offset[n], buffers_hash_lsh[n], buffers_a[n], buffers_len_output[n], buffers_dict[n]);

				q.wait();

				auto end=std::chrono::system_clock::now();

				if(i>0){
					times.emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
				}
				n++;
			}

			dev++;


		}


		for(auto t:times){
			cout<<"Times: "<<(float)t/1000<<"sec"<<std::endl;
		}


		if(num_dev>1){
			auto max_iter = std::max_element(times.begin(),times.end());
			auto min_iter = std::min_element(times.begin(),times.end());

			long max=*max_iter;
			long min=*min_iter;

			idx_max=max_iter-times.begin();
			idx_min=min_iter-times.begin();


			n_max=floor(((float)max/(float)(min+max))*(n_batches-number_of_testing_batches));


			n_min=n_batches-number_of_testing_batches-n_max;


		}else if(num_dev==1){
			n_max=0;
			n_min=(n_batches-number_of_testing_batches);
			idx_max=0;
			idx_min=0;

		}


		cout<<"n_max: "<<n_max<<std::endl;
		cout<<"n_min: "<<n_min<<std::endl;

		cout<<"id_max: "<<idx_max<<std::endl;
		cout<<"id_min: "<<idx_min<<std::endl;


		dev=0;

		auto end=std::chrono::system_clock::now();


		timer.end_time(0,2,1);

		cout<<"Time for measure computation: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<std::endl;


		timer.start_time(0,2,2);

		start=std::chrono::system_clock::now();

		offset.emplace_back(number_of_testing_batches*batch_size);

		cout<<"\n\tStart computation..."<<std::endl<<std::endl;


		for(int i=0; i<num_dev; i++){

			// Create

			split_size.emplace_back((i==idx_max?n_min:n_max)*batch_size);


//			while(n<n_batches/num_dev+dev*n_batches/num_dev){


			auto start=std::chrono::system_clock::now();

			{

				cout<<"t "<<n<<" start: "<<std::chrono::duration_cast<std::chrono::milliseconds>(start-start_timeline).count()<<std::endl;

			}


//			uint32_t offset=n*batch_size;//split_size;//+iter*batch_size;

			offset.emplace_back(offset.back()+(dev==0?0:split_size[dev-1]));

// 			int loc_split_size=(iter==0?1*batch_size:((number_of_testing_batches-1)*batch_size));

			uint32_t loc_split_size=split_size[dev];//(dev==0?split_size:((n_batches-9)*batch_size));//batch_size;

			cout<<"Offset: "<<offset.back()<<std::endl;


			buffers_buckets.emplace_back(sycl::buffer<tuple<int,int,int,int,int>,1>(static_cast<tuple<int,int,int,int,int>*>(buckets.data()+offset.back()*NUM_REP*NUM_HASH*NUM_STR),range<1>{loc_split_size*NUM_STR*NUM_HASH*NUM_REP}, {sycl::property::buffer::use_host_ptr()})); // Wrong dimension

			buffers_a.emplace_back(buffer<uint32_t,1>((uint32_t*)a.data(),range<1>{a.size()}));

			buffers_hash_lsh.emplace_back(buffer<uint32_t,2>((uint32_t*)hash_lsh, range<2>{NUM_HASH,NUM_BITS}));

			buffers_dict.emplace_back(buffer<uint8_t,1>(dictionary,range<1>{256}));

			buffers_batch_size.emplace_back(buffer<uint32_t,1>(&batch_size, range<1>{1}));

			buffers_len_output.emplace_back(buffer<uint32_t,1>(&len_output, range<1>{1}));

			buffers_split_offset.emplace_back(buffer<uint32_t,1> (&offset.back(), range<1>{1}));


				// n*split offset w.r.t. the batches processed by the other device (ex. 3 batches)
				// Inside these batches there is another offset, that is iter*batch_size


			create_bucket_without_lshnumber_offset_NEW_impr(queues[i], embdata, buffers_buckets[n], buffers_batch_size[n], loc_split_size, buffers_split_offset[n], buffers_hash_lsh[n], buffers_a[n], buffers_len_output[n], buffers_dict[n]);

//			queues[i].wait();

			auto end=std::chrono::system_clock::now();

			time_on_dev[thread_num].emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());

			{

				cout<<"t "<<n<<" end: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start_timeline).count()<<std::endl;

			}

			n++;

			dev++;

		}

//		queues[0].wait();

		for(int i=0; i<time_on_dev.size(); i++){
			for(int t=0; t<time_on_dev[i].size(); t++ ){

				cout<<"\nTime "<<t<<" device "<<i<<": "<<time_on_dev[i][t]<<std::endl;

			}
			cout<<std::endl;
		}
		end=std::chrono::system_clock::now();

		cout<<"Time for actual computation: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<std::endl;

	}

	timer.end_time(0,2,2);




	cout<<"End of scope"<<std::endl;

}

void create_buckets_without_lshnumber_offset_2dev_wrapper(vector<queue> &queues, char **embdata, vector<tuple<int,int,int,int,int>> &buckets, uint32_t n_batches, uint32_t batch_size, int* hash_lsh, vector<int> &a, vector<int> &lshnumber, uint32_t len_output){

	std::cout<< "Selected: Create buckets - without lshnumber offset"<<std::endl;


	cout<<"Len output: "<<len_output<<std::endl;

	int num_dev=queues.size();


	auto start=std::chrono::system_clock::now();

	auto end=std::chrono::system_clock::now();



	int n_max;
	int n_min;


	int idx_max;
	int idx_min;

	int number_of_testing_batches=2*num_dev;//test_batches;


	vector<long> times;
	vector<vector<long>> time_on_dev(num_dev,vector<long>());

	auto start_timeline=std::chrono::system_clock::now();


	{
		vector<uint32_t> split_size;//=9*batch_size;

		int thread_num=0;

		start_timeline=std::chrono::system_clock::now();


		uint8_t dictionary[256]={0};

		inititalize_dictionary(dictionary);


		vector<uint32_t> offset;

		vector<sycl::buffer<tuple<int,int,int,int,int>>> buffers_buckets;
		vector<sycl::buffer<uint32_t,1>> buffers_batch_size;
		vector<sycl::buffer<uint32_t,1>> buffers_split_size;
		vector<sycl::buffer<uint32_t,1>> buffers_split_offset;
		vector<sycl::buffer<uint32_t,2>> buffers_hash_lsh;
		vector<sycl::buffer<uint32_t,1>> buffers_a;
		vector<sycl::buffer<uint32_t,1>> buffers_lshnumber;
		vector<sycl::buffer<uint32_t,1>> buffers_len_output;
		vector<sycl::buffer<uint8_t,1>>  buffers_dict;



		int n=0;

		int  dev=0;

		cout<<"\tStart profiling on devices..."<<std::endl<<std::endl;
		start=std::chrono::system_clock::now();


		for(auto &q:queues){


			for(int i=0; i<2; i++){

				auto start=std::chrono::system_clock::now();

				offset.emplace_back(2*batch_size*dev+i*batch_size);


				uint32_t loc_split_size=batch_size;

				cout<<"\n\tSet offset to: "<<offset[n]<<std::endl;

				buffers_buckets.emplace_back(sycl::buffer<tuple<int,int,int,int,int>,1>(static_cast<tuple<int,int,int,int,int>*>(buckets.data()+offset.back()*NUM_REP*NUM_HASH*NUM_STR),range<1>{loc_split_size*NUM_STR*NUM_HASH*NUM_REP}, {sycl::property::buffer::use_host_ptr()})); // Wrong dimension

				buffers_a.emplace_back(buffer<uint32_t,1>((uint32_t*)a.data(),range<1>{a.size()}));

				buffers_hash_lsh.emplace_back(buffer<uint32_t,2>((uint32_t*)hash_lsh, range<2>{NUM_HASH,NUM_BITS}));

				buffers_dict.emplace_back(buffer<uint8_t,1>(dictionary,range<1>{256}));

				buffers_batch_size.emplace_back(buffer<uint32_t,1>(&batch_size, range<1>{1}));

				buffers_len_output.emplace_back(buffer<uint32_t,1>(&len_output, range<1>{1}));

				buffers_split_offset.emplace_back(buffer<uint32_t,1> (&offset[n], range<1>{1}));


				// n*split offset w.r.t. the batches processed by the other device (ex. 3 batches)
				// Inside these batches there is another offset, that is iter*batch_size


				// The first one is longer then others
				create_bucket_without_lshnumber_offset(q, embdata, buffers_buckets[n], buffers_batch_size[n], loc_split_size, buffers_split_offset[n], buffers_hash_lsh[n], buffers_a[n], buffers_len_output[n], buffers_dict[n]);

				q.wait();

				auto end=std::chrono::system_clock::now();

				if(i>0){
					times.emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
				}
				n++;
			}

			dev++;


		}


		for(auto t:times){
			cout<<"Times: "<<(float)t/1000<<"sec"<<std::endl;
		}


		if(num_dev>1){
			auto max_iter = std::max_element(times.begin(),times.end());
			auto min_iter = std::min_element(times.begin(),times.end());

			long max=*max_iter;
			long min=*min_iter;

			idx_max=max_iter-times.begin();
			idx_min=min_iter-times.begin();


			n_max=floor(((float)max/(float)(min+max))*(n_batches-number_of_testing_batches));


			n_min=n_batches-number_of_testing_batches-n_max;


		}else if(num_dev==1){
			n_max=0;
			n_min=(n_batches-number_of_testing_batches);
			idx_max=0;
			idx_min=0;

		}


		cout<<"n_max: "<<n_max<<std::endl;
		cout<<"n_min: "<<n_min<<std::endl;

		cout<<"id_max: "<<idx_max<<std::endl;
		cout<<"id_min: "<<idx_min<<std::endl;


		dev=0;

		end=std::chrono::system_clock::now();

		cout<<"Time for measure computation: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<std::endl;


		start=std::chrono::system_clock::now();

		offset.emplace_back(number_of_testing_batches*batch_size);

		cout<<"\n\tStart computation..."<<std::endl<<std::endl;


		for(int i=0; i<num_dev; i++){

			// Create

			split_size.emplace_back((i==idx_max?n_min:n_max)*batch_size);


//			while(n<n_batches/num_dev+dev*n_batches/num_dev){


			auto start=std::chrono::system_clock::now();

			{

				cout<<"t "<<n<<" start: "<<std::chrono::duration_cast<std::chrono::milliseconds>(start-start_timeline).count()<<std::endl;

			}


//			uint32_t offset=n*batch_size;//split_size;//+iter*batch_size;

			offset.emplace_back(offset.back()+(dev==0?0:split_size[dev-1]));

// 			int loc_split_size=(iter==0?1*batch_size:((number_of_testing_batches-1)*batch_size));

			uint32_t loc_split_size=split_size[dev];//(dev==0?split_size:((n_batches-9)*batch_size));//batch_size;

			cout<<"Offset: "<<offset.back()<<std::endl;


			buffers_buckets.emplace_back(sycl::buffer<tuple<int,int,int,int,int>,1>(static_cast<tuple<int,int,int,int,int>*>(buckets.data()+offset.back()*NUM_REP*NUM_HASH*NUM_STR),range<1>{loc_split_size*NUM_STR*NUM_HASH*NUM_REP}, {sycl::property::buffer::use_host_ptr()})); // Wrong dimension

			buffers_a.emplace_back(buffer<uint32_t,1>((uint32_t*)a.data(),range<1>{a.size()}));

			buffers_hash_lsh.emplace_back(buffer<uint32_t,2>((uint32_t*)hash_lsh, range<2>{NUM_HASH,NUM_BITS}));

			buffers_dict.emplace_back(buffer<uint8_t,1>(dictionary,range<1>{256}));

			buffers_batch_size.emplace_back(buffer<uint32_t,1>(&batch_size, range<1>{1}));

			buffers_len_output.emplace_back(buffer<uint32_t,1>(&len_output, range<1>{1}));

			buffers_split_offset.emplace_back(buffer<uint32_t,1> (&offset.back(), range<1>{1}));


				// n*split offset w.r.t. the batches processed by the other device (ex. 3 batches)
				// Inside these batches there is another offset, that is iter*batch_size


			create_bucket_without_lshnumber_offset(queues[i], embdata, buffers_buckets[n], buffers_batch_size[n], loc_split_size, buffers_split_offset[n], buffers_hash_lsh[n], buffers_a[n], buffers_len_output[n], buffers_dict[n]);

//			queues[i].wait();

			auto end=std::chrono::system_clock::now();

			time_on_dev[thread_num].emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());

			{

				cout<<"t "<<n<<" end: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start_timeline).count()<<std::endl;

			}

			n++;

			dev++;

		}

//		queues[0].wait();

		for(int i=0; i<time_on_dev.size(); i++){
			for(int t=0; t<time_on_dev[i].size(); t++ ){

				cout<<"\nTime "<<t<<" device "<<i<<": "<<time_on_dev[i][t]<<std::endl;

			}
			cout<<std::endl;
		}

	}


	end=std::chrono::system_clock::now();

	cout<<"Time for real computation: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<std::endl;


	cout<<"End of scope"<<std::endl;

}



/*
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
*/



void generate_candidates_without_lshnumber_offset(queue &device_queue, buffer<int,1> &buffer_len, buffer<char,2> &buffer_oristrings, char **embdata, buffer<tuple<int,int,int,int,int>,1> &buffer_buckets, buffer<uint32_t,1> &buffer_batch_size, buffer<tuple<int,int>,1> &buffer_bucket_delimiter, buffer<tuple<int,int,int,int,int,int>,1> &buffer_candidates, uint32_t candidate_size, buffer<int,1> &buffer_candidate_start, buffer<int,2> &buffer_hash_lsh, buffer<uint32_t,1> &buffer_len_output, buffer<int,1> &buffer_reverse_index){

	{
//		std::lock_guard<std::mutex> lock(output);
		cout << "\n\tTask: Candidate Pairs Generation\t";
		std::cout << "Device: " << device_queue.get_device().get_info<info::device::name>() << std::endl;
	}


//

	{
		device_queue.submit([&](handler &cgh){

			auto acc_reverse_index = buffer_reverse_index.get_access<access::mode::read>(cgh);


			auto acc_delimiter = buffer_bucket_delimiter.get_access<access::mode::read>(cgh);
			auto acc_buckets = buffer_buckets.get_access<access::mode::read>(cgh);
			auto acc_oridata = buffer_oristrings.get_access<access::mode::read>(cgh);

//			auto acc_embdata = buffer_embeddata.get_access<access::mode::read>(cgh);

			auto acc_hash_lsh = buffer_hash_lsh.get_access<access::mode::read>(cgh);
			auto acc_candidate_start = buffer_candidate_start.get_access<access::mode::read>(cgh);
			auto acc_candidate = buffer_candidates.get_access<access::mode::write>(cgh);

			auto acc_len = buffer_len.get_access<access::mode::read>(cgh);

            auto acc_batch_size=buffer_batch_size.get_access<access::mode::read>(cgh);

            auto acc_len_output=buffer_len_output.get_access<access::mode::read>(cgh);




           std::cout<<"Candidate size: "<<candidate_size<<std::endl;

            cgh.parallel_for<class GenerateCandidates>(range<1>(candidate_size), [=](item<1> index){

				int ij=index[0];



				int found_at=acc_reverse_index[ij];

				int begin=get<0>(acc_delimiter[found_at]);
				int size=get<1>(acc_delimiter[found_at]);
				int end=begin+size;

				int max_comb=size*(size-1)/2;

				int ij_norm=ij-acc_candidate_start[found_at];

				int res=ij_norm/(size-1);

				int i_norm=0;

				int tmp_index=ij_norm;

				while(res>0) {
					tmp_index -= (size - 1);
					size--;
					res = tmp_index / (size - 1);
					i_norm++;
				}



				int j_norm=tmp_index%(size)+i_norm+1;


				int index_output=index[0];//acc_candidate_start[b];


				int i=begin+i_norm;
				int j=begin+j_norm;

				int t1=get<0>(acc_buckets[i]);
				int k1=get<1>(acc_buckets[i]);
				int i1=get<3>(acc_buckets[i]);
				int q1=get<4>(acc_buckets[i]);

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

					// TODO Handle better

					c1=embdata[i1/acc_batch_size[0]][ ABSPOS(i1%acc_batch_size[0],t1,q1,acc_hash_lsh[k1][j],acc_len_output[0]) ];
					c2=embdata[i2/acc_batch_size[0]][ ABSPOS(i2%acc_batch_size[0],t1,q2,acc_hash_lsh[k1][j],acc_len_output[0]) ];


					if(c1!=0 && c2!=0){
						sum+=abs_diff(c1,c2);
					}
				}

				get<5>(acc_candidate[index_output])=sum;


			});

		});

//		device_queue.wait_and_throw();

	}
}


void generate_candidates_without_lshnumber_offset_NEW(queue &device_queue, buffer<int,1> &buffer_len, buffer<char,2> &buffer_oristrings, char **embdata, buffer<tuple<int,int,int,int,int>,1> &buffer_buckets, buffer<uint32_t,1> &buffer_buckets_offset, buffer<uint32_t,1> &buffer_batch_size, buffer<tuple<int,int,int,int,int,int>,1> &buffer_candidates, uint32_t candidate_size, buffer<int,2> &buffer_hash_lsh, buffer<uint32_t,1> &buffer_len_output ){


		cout << "\n\tTask: Candidate Pairs Generation\t";
		std::cout << "Device: " << device_queue.get_device().get_info<info::device::name>() << std::endl;

		device_queue.submit([&](handler &cgh){

			auto acc_buckets = buffer_buckets.get_access<access::mode::read>(cgh);

			auto acc_candidate = buffer_candidates.get_access<access::mode::write>(cgh);

			auto acc_len = buffer_len.get_access<access::mode::read>(cgh);

            auto acc_batch_size=buffer_batch_size.get_access<access::mode::read>(cgh);

            auto acc_len_output=buffer_len_output.get_access<access::mode::read>(cgh);

            auto acc_buckets_offset=buffer_buckets_offset.get_access<access::mode::read>(cgh);


           std::cout<<"Candidate size: "<<candidate_size<<std::endl;

            cgh.parallel_for<class GenerateCandidates>(range<1>(candidate_size), [=](item<1> index){

            	int ij = index[0];


				int index_output=ij;

				int sum=0;

				int i=get<0>(acc_candidate[ij])-acc_buckets_offset[0];//begin+i_norm;
				int j=get<1>(acc_candidate[ij])-acc_buckets_offset[0];//begin+i_norm;

				int t1=get<0>(acc_buckets[i]);
				int k1=get<1>(acc_buckets[i]);
				int i1=get<3>(acc_buckets[i]);
				int q1=get<4>(acc_buckets[i]);

//				int t2=get<0>(acc_buckets[j]);
//				int k2=get<1>(acc_buckets[j]);
				int i2=get<3>(acc_buckets[j]);
				int q2=get<4>(acc_buckets[j]);


				__int8_t c1;
				__int8_t c2;
//				__int64 s1=0;
//				__int64 s2=0;

//				uint8_t l=0;
//				for (int j = k1*NUM_BITS; j < k1*NUM_BITS+NUM_BITS; j+=8){

				for (int j = k1*NUM_BITS; j < k1*NUM_BITS+NUM_BITS; j++){

//					__int64 s1=(*(__int64*)( embdata[i1/acc_batch_size[0]] + ABSPOS(i1%acc_batch_size[0],t1,q1,j,acc_len_output[0]) ));
//					__int64 s2=(*(__int64*)( embdata[i2/acc_batch_size[0]] + ABSPOS(i2%acc_batch_size[0],t1,q2,j,acc_len_output[0]) ));

					c1=embdata[i1/acc_batch_size[0]][ ABSPOS(i1%acc_batch_size[0],t1,q1,j,acc_len_output[0]) ];
//					str1[l]=embdata[i1/acc_batch_size[0]][ ABSPOS(i1%acc_batch_size[0],t1,q1,j,acc_len_output[0]) ];
					c2=embdata[i2/acc_batch_size[0]][ ABSPOS(i2%acc_batch_size[0],t1,q2,j,acc_len_output[0]) ];

//					s1 = ( s1 << 8 );
//					s1 = ( s1 | c1 );
//
//					s2 = ( s2 << 8 );
//					s2 = ( s2 | c2 );
//
//
//					if((l+1)%8==0){
//						sum+=(s1==s2?0:1);
//						s1=0;
//						s2=0;
//					}
//					l++;


					if(c1!=0 && c2!=0){
						sum+=abs_diff(c1,c2);
					}

//					l++;
//
				}
//				sum+=(s1==s2?0:1);

				get<0>(acc_candidate[index_output])=i1;
				get<1>(acc_candidate[index_output])=q1;
				get<2>(acc_candidate[index_output])=i2;
				get<3>(acc_candidate[index_output])=q2;
				get<4>(acc_candidate[index_output])=abs_diff(acc_len[i1], acc_len[i2]);
				get<5>(acc_candidate[index_output])=sum;


			});

		});

}



void generate_candidates_without_lshnumber_offset_2dev_wrapper(vector<queue>& queues, vector<int> &len, char* oristrings, char **embdata, vector<tuple<int,int,int,int,int>> &buckets, unsigned int batch_size,/* vector<tuple<int,int>> &bucket_delimiter,*/ vector<std::tuple<int,int,int,int,int,int>>& candidate,/* vector<int> &candidates_start,*/ int * local_hash_lsh, vector<int> &lshnumber, uint32_t len_output/*, vector<arrayWrapper> &partitionsBucketsDelimiter, vector<arrayWrapper> &partitionsCandStart, vector<arrayWrapper> &partitionsBuckets, vector<arrayWrapper> &partitionsCandidates*/){

	cout << "Selected: Generate candidates - without lshnumber offset"<< std::endl;

	cout<<"Len output: "<<len_output<<std::endl;
//	ofstream outFile;
//
//
//	outFile.open("cand_before_with_end", ios::out | ios::trunc);
//
//
//	int i=0;
//	for(auto &c:candidate){
//		outFile<<get<0>(c)<<" "<<get<1>(c)<<" "<<get<2>(c)<<std::endl;
//		i++;
//	}



//	int y_dim=NUM_BITS;
//	int x_dim=static_cast<int>(250/*/NUM_BITS*/);
//
//	range<2> local_range(x_dim,y_dim);
	{

	int num_dev=queues.size();


	vector<vector<uint32_t>> size_cand(num_dev,vector<uint32_t>());

	vector<uint32_t> number_of_iter(num_dev);

	vector<uint32_t> buckets_offset;

	vector<vector<int>> reverse_index;

	vector<buffer<int,1>> buffers_reverse_index;


	vector<buffer<char,2>> buffers_oristrings;

	vector<buffer<int, 1>> buffers_candidate_start;

	vector<buffer<tuple<int,int,int,int,int>>> buffers_buckets;

	vector<buffer<tuple<int,int>>> buffers_bucket_delimiter;

	vector<buffer<int, 2>> buffers_hash_lsh;

	vector<buffer<tuple<int,int,int,int,int,int>>> buffers_candidates;

	vector<buffer<int,1>> buffers_len;

	vector<buffer<unsigned int, 1>> buffers_batch_size;

	vector<buffer<uint32_t, 1>> buffers_len_output;

	vector<buffer<uint32_t,1>> buffers_buckets_offset;



//	int num_dev=queues.size();

	vector<long> times;
	vector<vector<long>> time_on_dev(num_dev,vector<long>());


	uint32_t size_for_test=0.01*candidate.size();

	cout<<"Size (num candidates) for profiling: "<<size_for_test<<std::endl;

//	if(size_for_test%x_dim){
//		cout<<"\tAlign size for profiling to "<<x_dim<<std::endl;
//	}
//	while(size_for_test%x_dim!=0){
//		size_for_test--;
//	}

	cout<<"\n\tNew size for test: "<<size_for_test<<std::endl;


	timer.start_time(0,5,1);

	int n_max;
	int n_min;


	int idx_max;
	int idx_min;


	int dev=0;
	int n=0;


	cout<<"\n\tStart profiling..."<<std::endl;



	for(auto &q:queues){


		for(int i=0; i<2; i++){

			auto start=std::chrono::system_clock::now();

			int iter=0;

//			size_for_test+=(dev>0?(candidate.size()%num_dev):0);

			uint32_t start_b=get<0>(candidate[size_for_test*n]);
			uint32_t end_b=get<2>(candidate[size_for_test*n+size_for_test-1])-1;

			uint32_t size_buckets=end_b-start_b+1;

			buckets_offset.emplace_back(start_b);

			cout<<"\n\tIter "<<dev<<". Start buckets at "<<size_for_test*n<<": "<<start_b<<std::endl;
			cout<<"\tIter "<<dev<<". End buckets at "<<size_for_test*n + size_for_test-1<<": "<<end_b<<std::endl;
			cout<<"\n\tBuckets size: "<<size_buckets<<std::endl;

			buffers_oristrings.emplace_back( buffer<char,2>(oristrings,range<2>{NUM_STRING,LEN_INPUT}/*, {property::buffer::use_host_ptr()}*/));


			buffers_buckets.emplace_back( buffer<tuple<int,int,int,int,int>>(buckets.data()+start_b,range<1>{size_buckets}/*, {property::buffer::use_host_ptr()}*/));


			cout<<"\tCand size: "<<size_for_test<<std::endl;

			buffers_hash_lsh.emplace_back( buffer<int, 2>(reinterpret_cast<int*>(local_hash_lsh),range<2>{NUM_HASH,NUM_BITS}/*, {property::buffer::use_host_ptr()}*/));

			buffers_candidates.emplace_back( buffer<tuple<int,int,int,int,int,int>>(candidate.data()+n*size_for_test,range<1>{size_for_test}/*, {property::buffer::use_host_ptr()}*/));

			buffers_len.emplace_back( buffer<int,1>(len.data(),range<1>{len.size()}/*, {property::buffer::use_host_ptr()}*/));

			buffers_batch_size.emplace_back( buffer<unsigned int, 1>(&batch_size,range<1>{1}));

			buffers_len_output.emplace_back( buffer<uint32_t, 1>(&len_output,range<1>{1}));

			buffers_buckets_offset.emplace_back( buffer<uint32_t,1>(&buckets_offset.back(),range<1>{1}));

			generate_candidates_without_lshnumber_offset_NEW(q, buffers_len[n], buffers_oristrings[n], embdata, buffers_buckets[n], buffers_buckets_offset[n], buffers_batch_size[n], buffers_candidates[n], size_for_test, buffers_hash_lsh[n], buffers_len_output[n]);

//			generate_candidates_without_lshnumber_offset(q, buffers_len[n], buffers_oristrings[n], embdata, buffers_buckets[n], buffers_buckets_offset[n], buffers_batch_size[n], buffers_candidates[n], size_for_test, buffers_hash_lsh[n], buffers_len_output[n],local_range);

			q.wait();

			auto end=std::chrono::system_clock::now();

			if(i>0){
				times.emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
			}
			n++;
		}

		dev++;


	}


	for(auto t:times){
		cout<<"Times: "<<(float)t/1000<<"sec"<<std::endl;
	}

	uint32_t remaining_size=candidate.size()-size_for_test*2*num_dev;





	cout<<"\tRemaining size: "<<remaining_size<<std::endl;


	if(num_dev>1){
		auto max_iter = std::max_element(times.begin(),times.end());
		auto min_iter = std::min_element(times.begin(),times.end());

		long max=*max_iter;
		long min=*min_iter;

		idx_max=max_iter-times.begin();
		idx_min=min_iter-times.begin();


		n_max=floor(((float)max/(float)(min+max))*remaining_size);

		cout<<"\n\tNumber of candidates to assign to the faster device: "<<n_max<<std::endl;

		n_min=remaining_size-n_max;

		cout<<"\n\tNumber of candidates to assign to other device: "<<n_min<<std::endl;



		uint64_t s=n_min*sizeof(candidate[0]);

		cout<<"size of 6-int tuple: "<<sizeof(candidate[0])<<std::endl;

		int num_kernels=1;

		cout<<s<<std::endl;

		while(s>0xFFFFFFFF){
			cout<<"Warning: too much data for a buffer."<<std::endl;

			num_kernels=num_kernels*2;

			s=s/2;

		}

		for(int l=0; l<num_kernels; l++){

			if(l==num_kernels-1){

				int remind=n_min%(num_kernels);

				size_cand[idx_max].emplace_back(n_min/(num_kernels)+remind);

			}else{

				size_cand[idx_max].emplace_back(n_min/(num_kernels));

			}

		}





		s=n_max*sizeof(candidate[0]);

		cout<<"size of 6-int tuple: "<<sizeof(candidate[0])<<std::endl;

		num_kernels=1;

		cout<<s<<std::endl;

		while(s>0xFFFFFFFF){
			cout<<"Warning: too much data for a buffer."<<std::endl;

			num_kernels=num_kernels*2;

			s=s/2;

		}

		for(int l=0; l<num_kernels; l++){

			if(l==num_kernels-1){

				int remind=n_max%(num_kernels);

				size_cand[idx_min].emplace_back(n_max/(num_kernels)+remind);

			}else{

				size_cand[idx_min].emplace_back(n_max/(num_kernels));

			}

		}


//		size_cand[idx_max].emplace_back(n_min);


//		size_cand[idx_min].emplace_back(n_max); // At this point n_max is multiple of local_range[0] for sure


	}else if(num_dev==1){
		vector<int> tmp_sizes;

		n_max=0;

//		if(remaining_size*sizeof(candidate[0])>0xFFFFFFFF){

		//sizeof(candidate[0]);

		uint64_t s=remaining_size*sizeof(candidate[0]);
		cout<<"size of 6-int tuple: "<<sizeof(candidate[0])<<std::endl;

		int num_kernels=1;

		cout<<s<<std::endl;

		while(s>0xFFFFFFFF){

			cout<<"Warning: too much data for a buffer."<<std::endl;

			num_kernels=num_kernels*2;

			s=s/2;

		}

		for(int l=0; l<num_kernels; l++){
			if(l==num_kernels-1){
				int remind=remaining_size%(num_kernels);
				size_cand[idx_max].emplace_back(remaining_size/(num_kernels)+remind);
			}else{
				size_cand[idx_max].emplace_back(remaining_size/(num_kernels));
			}
		}

		//To manage the remainder


// Useless
		n_min=remaining_size;
		idx_max=0;
		idx_min=0;

		cout<<"\n\tNumber of candidates to assign to device: "<<n_min<<std::endl;

		//size_cand[idx_max].insert(size_cand[idx_max].end(), tmp_sizes.begin(), tmp_sizes.end());

	}


	cout<<std::endl;
	for(auto d:size_cand){
		for(auto s:d){
			cout<<"\tSize: "<<s<<std::endl;
		}
	}


	cout<<"\nn_max: "<<n_max<<std::endl;
	cout<<"n_min: "<<n_min<<std::endl;

	cout<<"id_max: "<<idx_max<<std::endl;
	cout<<"id_min: "<<idx_min<<std::endl;


	dev=0;

	timer.end_time(0,5,1);

	timer.start_time(0,5,2);

	uint32_t offset_cand=size_for_test*2*num_dev;


	for(auto &q : queues){

		int iter=0;

		while(iter<size_cand[dev].size()){

			cout<<"\n\tSize cand[dev]: "<<size_cand[dev][iter]<<std::endl;

			uint32_t start_b=get<0>(candidate[offset_cand]);
			uint32_t end_b=get<2>((candidate.data()+offset_cand)[size_cand[dev][iter]-1])-1;

			uint32_t size_buckets=end_b-start_b+1;

			buckets_offset.emplace_back(start_b);

			cout<<"\n\tIter "<<dev<<". Start buckets at "<<offset_cand<<": "<<start_b<<std::endl;
			cout<<"\tIter "<<dev<<". End buckets at "<<offset_cand + size_cand[dev][iter]-1<<": "<<end_b<<std::endl;
			cout<<"\n\tBuckets size: "<<size_buckets<<std::endl;


			buffers_oristrings.emplace_back( buffer<char,2>(oristrings,range<2>{NUM_STRING,LEN_INPUT}, {property::buffer::use_host_ptr()}));


			buffers_buckets.emplace_back( buffer<tuple<int,int,int,int,int>>(buckets.data()+start_b,range<1>{size_buckets}, {property::buffer::use_host_ptr()}));

			cout<<"\tCand size: "<<size_cand[dev][iter]<<std::endl;

			buffers_hash_lsh.emplace_back( buffer<int, 2>(reinterpret_cast<int*>(local_hash_lsh),range<2>{NUM_HASH,NUM_BITS}, {property::buffer::use_host_ptr()}));

			buffers_candidates.emplace_back( buffer<tuple<int,int,int,int,int,int>>(candidate.data()+offset_cand,range<1>{size_cand[dev][iter]}, {property::buffer::use_host_ptr()}));

			buffers_len.emplace_back( buffer<int,1>(len.data(),range<1>{len.size()}, {property::buffer::use_host_ptr()}));

			buffers_batch_size.emplace_back( buffer<unsigned int, 1>(&batch_size,range<1>{1}));

			buffers_len_output.emplace_back( buffer<uint32_t, 1>(&len_output,range<1>{1}));

			buffers_buckets_offset.emplace_back( buffer<uint32_t,1>(&buckets_offset.back(),range<1>{1}));



//			if(size_cand[dev][iter]<local_range[0]){
//				local_range[0]=size_cand[dev][iter];
//			}

			generate_candidates_without_lshnumber_offset_NEW(q, buffers_len[n], buffers_oristrings[n], embdata, buffers_buckets[n], buffers_buckets_offset[n], buffers_batch_size[n], buffers_candidates[n], size_cand[dev][iter], buffers_hash_lsh[n], buffers_len_output[n]);
//			generate_candidates_without_lshnumber_offset(q, buffers_len[n], buffers_oristrings[n], embdata, buffers_buckets[n], buffers_buckets_offset[n], buffers_batch_size[n], buffers_candidates[n], size_for_test, buffers_hash_lsh[n], buffers_len_output[n],local_range);


			offset_cand+=size_cand[dev][iter];

			n++;
			iter++;
		}

		dev++;


	}

	}

	timer.end_time(0,5,2);


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
	tbb::parallel_sort(outputs.begin(), outputs.end());
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

//		for(int i=0; i<20; i++){
//			for(int j=0; j<batch_size*NUM_STR*NUM_REP*len_output; j++){
//				outFile<<output[i][j];
//			}
//			outFile<<std::endl;
//		}
		for(int i=0; i<NUM_STRING; i++){
			for(int j=0; j<NUM_STR; j++ ){
				for(int k=0; k<NUM_REP; k++){
//					outFile<<"("<<i<<","<<j<<","<<k<<"): ";
					for(int t=0; t<len_output; t++){

						if(output[(int)(i/batch_size)][ABSPOS((int)(i%batch_size),j,k,t,len_output)]==0){
							break;
						}
						outFile<<output[(int)(i/batch_size)][ABSPOS((int)(i%batch_size),j,k,t,len_output)];

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

			outFile<</*get<4>(candidates[i])<<", "<<get<5>(candidates[i])<<", "<<*/get<0>(candidates[i])<<", "<<get<1>(candidates[i])<<", "<<get<2>(candidates[i])<<", "<<get<3>(candidates[i])<<", "<<get<4>(candidates[i])<<", "<<get<5>(candidates[i])<<std::endl;

		}
	}
}




void initialize_candidate_pairs(vector<queue>& queues, vector<tuple<int,int,int,int,int>> &buckets, vector<std::tuple<int,int,int,int,int,int>> &candidates ){


	cout<<"\nInitialize candidate vector"<<std::endl;
	/*
	 * Compute the boundary ( starting index and size ) of each buckets in the 1-D vector
	 *
	 * */
	vector<tuple<int,int>> buckets_delimiter;

	auto start=std::chrono::system_clock::now();

		int j=0;
		uint64_t size=0;

		buckets_delimiter.emplace_back(make_tuple(0,0));


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

//	sub_time_compute_buckets_delim=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

	std::cout<<"\n\tTime cand-init: count element: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;


	/**
	 *
	 * Remove buckets having size == 1, since no candidates are possible
	 *
	 * */



	start=std::chrono::system_clock::now();
	std::cout<<"Size before remove: "<<buckets_delimiter.size()<<std::endl;

//		auto remove_policy = oneapi::dpl::execution::make_device_policy(queues.back());
		auto new_end=remove_if(/*oneapi::dpl::execution::par, */buckets_delimiter.begin(),buckets_delimiter.end(),[](std::tuple<int,int> e){return std::get<1>(e)<2;});

		buckets_delimiter.erase( new_end, buckets_delimiter.end());
		std::cout<<"Size after remove: "<<buckets_delimiter.size()<<std::endl;

	end=std::chrono::system_clock::now();

//	sub_time_filterout_one_elem_buckets=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();


	std::cout<<"\tTime cand-init: remove element: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;







	int num_splits=queues.size();


	start=std::chrono::system_clock::now();


	/**
	 *
	 * Since each buckets has a variable number of possible candidates,
	 * count the maximum number of candiadtes by scanning the array.
	 * This allows us to set approximately the same number of candidates for each kernel.
	 * This is not possible if the split was based ony on buckets delimiters
	 *
	 *
	 * */

	std::vector<int> n_values;

		for(int b=0; b<buckets_delimiter.size(); b++){
			int n=get<1>(buckets_delimiter[b]);
			size+=((n*(n-1))/2);
			n_values.emplace_back(n);
		}

		auto p=std::max_element(n_values.begin(), n_values.end());


		std::cout<<"Max n: "<<*p<<std::endl;
		start=std::chrono::system_clock::now();

		std::cout<<"Size: "<<size<<std::endl;

		candidates.resize(size/*,make_tuple(-1,-1,-1,-1,-1,-1)*/);
		end=std::chrono::system_clock::now();

		std::cout<<"\tTime cand-init: resize vector: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;

//		start=std::chrono::system_clock::now();
//
//
//		tuple<int,int,int,int,int,int> *arr_cand=new tuple<int,int,int,int,int,int>[size];
//
//		end=std::chrono::system_clock::now();
//
//
//
//
//		std::cout<<"\nTime cand-init: resize array: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;

//		cout<<get<0>(arr_cand[size-1])<<std::endl;

		start=std::chrono::system_clock::now();

		int c=0;
		for(auto &b:buckets_delimiter ){
			int start=get<0>(b);
			int size=get<1>(b);
			int end=start+size;

			for(int i=start; i<end-1; i++){
				for(int j=i+1; j<end; j++ ){
					get<0>(candidates[c])=i;
					get<1>(candidates[c])=j;
					get<2>(candidates[c])=end;
//					get<0>(arr_cand[c])=i;
//					get<1>(arr_cand[c])=j;
				    c++;
				}
			}
		}

		if(c!=size){
			cout<<c<<" != "<<size<<std::endl;
			cout<<"Exit"<<std::endl;
			exit(-1);
		}
		cout<<c<<" == "<<size<<std::endl;

		end=std::chrono::system_clock::now();

		std::cout<<"\tTime cand-init: assign i and j to candidates: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;


//		int i=0;
//		for(auto &c:candidates){
//			if(i<20){
//				cout<<get<0>(c)<<" "<<get<1>(c)<<std::endl;
//			}
//
//			if(i>candidates.size()-20){
//				cout<<get<0>(c)<<" "<<get<1>(c)<<std::endl;
//
//			}
//			i++;
//
//		}

}


void initialize_candidate_pairs_onDevice(vector<queue>& queues, vector<tuple<int,int,int,int,int>> &buckets, vector<std::tuple<int,int,int,int,int,int>> &candidates ){



	cout<<"Initialize candidates on device"<<std::endl;

		timer.start_time(0,4,1);

			auto start=std::chrono::system_clock::now();
			std::vector<tuple<int,int>> delimiter(buckets.size());
			get<0>(delimiter[0]) = 0;

			{
				cl::sycl::buffer<tuple<int,int>> buckets_delimiter_buf{ delimiter.data(), delimiter.size(), {property::buffer::use_host_ptr()}};
				cl::sycl::buffer<tuple<int,int,int,int,int>> array_buf{ buckets.data(), buckets.size(), {property::buffer::use_host_ptr()}};

				queues.front().submit([&](sycl::handler& cgh) {

					auto pv_acc = buckets_delimiter_buf.get_access<cl::sycl::access::mode::write>(cgh);
					auto array_acc = array_buf.get_access<cl::sycl::access::mode::read>(cgh);

					cgh.parallel_for<class partition_kernel>(cl::sycl::range<1>{buckets.size() - 1},
						[=](cl::sycl::id<1> idx) {
							if ( (get<0>(array_acc[idx[0]])!=get<0>(array_acc[idx[0] + 1]))
									|| (get<0>(array_acc[idx[0]])==get<0>(array_acc[idx[0] + 1]) && get<1>(array_acc[idx[0]])!=get<1>(array_acc[idx[0] + 1]) )
									|| (get<0>(array_acc[idx[0]])==get<0>(array_acc[idx[0] + 1]) && get<1>(array_acc[idx[0]])==get<1>(array_acc[idx[0] + 1]) && get<2>(array_acc[idx[0]])!=get<2>(array_acc[idx[0] + 1])) ) {
								get<0>(pv_acc[idx[0] + 1]) = idx[0] + 1;
							}
						});
				}).wait();
			} // For synch

//			auto remove_policy = dpl::execution::make_device_policy(queues.back());
			auto new_end=remove_if(oneapi::dpl::execution::par, delimiter.begin()+1,delimiter.end(),[](std::tuple<int,int> e){return std::get<0>(e)==0;});
				delimiter.erase( new_end, delimiter.end());
//				auto end=std::chrono::system_clock::now();
//			std::cout<<"Time cand-init: parallel count element: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;
			size_t size=0;

//			start=std::chrono::system_clock::now();
			{
				cl::sycl::buffer<tuple<int,int>> buckets_delimiter_buf{ delimiter.data(), delimiter.size(), {property::buffer::use_host_ptr()}};

				queues.front().submit([&](sycl::handler& cgh) {
					auto pv_acc = buckets_delimiter_buf.get_access<cl::sycl::access::mode::write>(cgh);
					cgh.parallel_for<class partition_kernel>(cl::sycl::range<1>{buckets.size()-1},
						[=](cl::sycl::id<1> idx) {
							get<1>(pv_acc[idx[0]]) = get<0>(pv_acc[idx[0]+1]) - get<0>(pv_acc[idx[0]]);
						});
					}).wait();
			} // For synch

			auto end=std::chrono::system_clock::now();
			timer.end_time(0,4,1);

			std::cout<<"\n\tTime cand-init: parallel count element: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;

			timer.start_time(0,4,2);
			start=std::chrono::system_clock::now();

			//auto remove_policy = dpl::execution::make_device_policy(queues.back());

			 new_end=std::remove_if(dpl::execution::par, delimiter.begin(),delimiter.end(),[](std::tuple<int,int> &e){return std::get<1>(e)<2;});

			 delimiter.erase( new_end, delimiter.end());

			 end=std::chrono::system_clock::now();

			 timer.end_time(0,4,2);

			 std::cout<<"\tTime cand-init: parallel remove element: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;
//			std::for_each(delimiter.begin(), delimiter.begin()+10,[](tuple<int,int>d){
//				std::cout<<get<0>(d)<<" "<<get<1>(d)<<std::endl;
//			});



		size=0;
		for(int b=0; b<delimiter.size(); b++){
			int n=get<1>(delimiter[b]);
			size+=((n*(n-1))/2);
		}

		cout<<"\t\tCandidate vector size: "<<size<<std::endl;

		timer.start_time(0,4,3);
		start=std::chrono::system_clock::now();


		candidates.resize(size,tuple<int,int,int,int,int,int>(-1,-1,-1,-1,-1,-1));
		end=std::chrono::system_clock::now();

		timer.end_time(0,4,3);

		std::cout<<"\tTime cand-init: resize: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;

//		start=std::chrono::system_clock::now();
//
//		int *mem = new int[size * sizeof(tuple<int, int, int, int, int, int>)];
//		tuple<int,int,int,int,int,int> *cand = reinterpret_cast<tuple<int,int,int,int,int,int>*>(mem);
//
//		end=std::chrono::system_clock::now();
//
//		std::cout<<"\nTime cand-init: alternative allocation: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;
//		get<0>(cand[size-1])=99;
//		cout<<get<0>(cand[size-1])<<std::endl;

		timer.start_time(0,4,4);

		start=std::chrono::system_clock::now();

		int c=0;

		for(auto &b:delimiter ){
			int start=get<0>(b);
			int size=get<1>(b);
			int end=start+size;

			for(int i=start; i<end-1; i++){
				for(int j=i+1; j<end; j++ ){
					get<0>(candidates[c])=i;
					get<1>(candidates[c])=j;
					get<2>(candidates[c])=end;
					c++;
				}
			}
		}


		if(c!=size){
			cout<<c<<" != "<<size<<std::endl;

			exit(-1);
		}
		cout<<c<<" == "<<size<<std::endl;

		end=std::chrono::system_clock::now();

		timer.end_time(0,4,4);
		std::cout<<"\tTime cand-init: assign i and j to candidates: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;



}



int* parallel_embedding_while_loop_2dev_wrapper(std::vector<queue> &queues, std::vector<int> &len_oristrings, char (*oristrings)[LEN_INPUT], char** &set_embdata_dev, unsigned int batch_size, uint32_t n_batches, std::vector<int> &lshnumber, uint32_t &len_output, std::vector<tuple<int,int>> &rev_hash, int *p){

	std::cout<< "Selected: Parallel embedding - while loop version"<<std::endl;



	std::atomic<int> task(0);

	// DICTIONARY

	uint8_t dictionary[256]={0};
	inititalize_dictionary(dictionary);

//	len_output=NUM_HASH*NUM_BITS;//lshnumber.size();

	len_output=lshnumber.size();

	cout<<"\n\tSet len_output to: "<<len_output<<std::endl;



	auto start=std::chrono::system_clock::now();

	for(int n=0; n<n_batches; n++){
			set_embdata_dev[n]=malloc_shared<char>(batch_size*NUM_STR*NUM_REP*len_output, queues.back());
			memset(set_embdata_dev[n],0,batch_size*NUM_STR*NUM_REP*len_output);
	}

	auto end=std::chrono::system_clock::now();

	auto time=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

	std::cout<<"\tTime to allocate embedded strings (shared malloc): "<<(float)time/1000<<"sec"<<std::endl<<std::endl;

	uint32_t len_p=samplingrange+1;


//	int (*p)=new int[NUM_STR*NUM_CHAR*len_p];

//	generate_random_string(p, len_p);

//	vector<std::thread> threads;
	int n_max;
	int n_min;


	int idx_max;
	int idx_min;

	int num_dev=queues.size();

	int number_of_testing_batches=2*num_dev;//test_batches;


	std::vector<long> times;
	std::vector<std::vector<long>> time_on_dev(num_dev,std::vector<long>());


	{
		std::vector<buffer<int,1>> buffers_p;

		std::vector<buffer<char,2>> buffers_oristrings;

		std::vector<buffer<int,1>> buffers_lshnumber;

		std::vector<buffer<char,1>> buffers_embdata;

		std::vector<buffer<uint8_t,1>> buffers_dict;

		std::vector<buffer<int,1>> buffers_len_oristrings;

		std::vector<buffer<uint32_t,1>> buffers_samplingrange;

		std::vector<buffer<uint32_t,1>> buffers_len_output;

		std::vector<buffer<tuple<int,int>>> buffers_rev_hash;


		int n=0;
		int dev=0;

		std::cout<<"\tStart profiling on devices..."<<std::endl<<std::endl;


		start=std::chrono::system_clock::now();



	// ---------------------------------------

		for(auto &q:queues){


			for(int i=0; i<2; i++){

				auto start=std::chrono::system_clock::now();

				uint32_t size_p=static_cast<unsigned int>(NUM_STR*NUM_CHAR*(samplingrange+1));

				buffers_p.emplace_back( buffer<int,1>(p,range<1>{size_p}) );

				buffers_oristrings.emplace_back( buffer<char, 2>(reinterpret_cast<char*>((char*)oristrings[n*batch_size]),range<2>{batch_size,LEN_INPUT}) );

				buffers_lshnumber.emplace_back( buffer<int, 1>(lshnumber.data(),range<1>{lshnumber.size()}) );

				unsigned int size_emb=static_cast<unsigned int>(batch_size*NUM_STR*NUM_REP*len_output);

				buffers_embdata.emplace_back( buffer<char, 1> (reinterpret_cast<char*>(set_embdata_dev[n]), range<1>{size_emb}, {property::buffer::use_host_ptr()}) );

				buffers_dict.emplace_back( buffer<uint8_t,1>(dictionary,range<1>{256}) );

				buffers_len_oristrings.emplace_back( buffer<int,1>(len_oristrings.data()+n*batch_size,range<1>(batch_size)) );

				uint32_t samprange=samplingrange;

				buffers_samplingrange.emplace_back( buffer<uint32_t,1>(&samprange,range<1>(1)) );

				buffers_len_output.emplace_back( buffer<uint32_t, 1>(&len_output,range<1>{1}) );

				buffers_rev_hash.emplace_back( buffer<tuple<int,int>>(rev_hash.data(),range<1>(rev_hash.size())));

				parallel_embedding_while_loop( q, buffers_len_oristrings[n], buffers_oristrings[n], buffers_embdata[n], batch_size, buffers_lshnumber[n], buffers_p[n], buffers_len_output[n], buffers_samplingrange[n], buffers_dict[n], buffers_rev_hash[n]);

				q.wait();

				auto end=std::chrono::system_clock::now();

				if(i>0){
					times.emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
				}

				n++;
			}

			dev++;


		}


		for(auto t:times){
			cout<<"\tTimes kernel: "<<(float)t/1000<<"sec"<<std::endl;
		}

		std::vector<int> iter_per_dev;


		if(num_dev>1){
			auto max_iter = std::max_element(times.begin(),times.end());
			auto min_iter = std::min_element(times.begin(),times.end());

			long max=*max_iter;
			long min=*min_iter;

			idx_max=max_iter-times.begin();
			idx_min=min_iter-times.begin();


			n_max=floor(((float)max/(float)(min+max))*(n_batches-number_of_testing_batches));


			n_min=n_batches-number_of_testing_batches-n_max;

			iter_per_dev.resize(num_dev);
			iter_per_dev[idx_max]=n_min;
			iter_per_dev[idx_min]=n_max;

		}else if(num_dev==1){

			n_max=0;
			n_min=(n_batches-number_of_testing_batches);
			idx_max=0;
			idx_min=0;
			iter_per_dev.emplace_back(n_min);

		}


		cout<<"n_max: "<<n_max<<std::endl;
		cout<<"n_min: "<<n_min<<std::endl;

		cout<<"id_max: "<<idx_max<<std::endl;
		cout<<"id_min: "<<idx_min<<std::endl;



		// --------------------

		end=std::chrono::system_clock::now();

		cout<<"\tTotal time for profiling: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<std::endl<<std::endl;;

		std::cout<<"\n\tStart computation..."<<std::endl<<std::endl;


		start=std::chrono::system_clock::now();



		dev=0;

		for(int i=0; i<num_dev; i++){

//			cout<<"From "<<n<<" to "<<n+iter_per_dev[i]<<std::endl;

			int iter=0;

			while(iter<iter_per_dev[i]){

				uint32_t size_p=static_cast<unsigned int>(NUM_STR*NUM_CHAR*(samplingrange+1));

				buffers_p.emplace_back( buffer<int,1>(p,range<1>{size_p}) );

				buffers_oristrings.emplace_back( buffer<char, 2>(reinterpret_cast<char*>(oristrings[n*batch_size]),range<2>{batch_size,LEN_INPUT}) );

				buffers_lshnumber.emplace_back( buffer<int, 1>(lshnumber.data(),range<1>{lshnumber.size()}) );

				unsigned int size_emb=static_cast<unsigned int>(batch_size*NUM_STR*NUM_REP*len_output);

				buffers_embdata.emplace_back( buffer<char, 1> (reinterpret_cast<char*>(set_embdata_dev[n]), range<1>{size_emb}, {property::buffer::use_host_ptr()}) );

				buffers_dict.emplace_back( buffer<uint8_t,1>(dictionary,range<1>{256}) );

				buffers_len_oristrings.emplace_back( buffer<int,1>(len_oristrings.data()+n*batch_size,range<1>(batch_size)) );

				uint32_t samprange=samplingrange;

				buffers_samplingrange.emplace_back( buffer<uint32_t,1>(&samprange,range<1>(1)) );

				buffers_len_output.emplace_back( buffer<uint32_t, 1>(&len_output,range<1>{1}) );

				buffers_rev_hash.emplace_back( buffer<tuple<int,int>>(rev_hash.data(),range<1>(rev_hash.size())));

				parallel_embedding_while_loop(queues[i], buffers_len_oristrings[n], buffers_oristrings[n], buffers_embdata[n], batch_size, buffers_lshnumber[n], buffers_p[n], buffers_len_output[n], buffers_samplingrange[n], buffers_dict[n], buffers_rev_hash[n]);

				n++;
				iter++;

			}

			dev++;
		}

	}

	end=std::chrono::system_clock::now();

	cout<<"\tTime for actual computation: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<std::endl;


//	delete[] p;
	return p;
}

int* parallel_embedding_while_loop_2dev_gath_wrapper(std::vector<queue> &queues, vector<int> &len_oristrings, char (*oristrings)[LEN_INPUT], char** &set_embdata_dev, unsigned int batch_size, uint32_t n_batches, std::vector<int> &lshnumber, uint32_t &len_output, std::vector<tuple<int,int>> &rev_hash/*, int*p*/){

	std::cout<< "Selected: Parallel embedding - while loop version"<<std::endl;


	// DICTIONARY

	uint8_t dictionary[256]={0};
	inititalize_dictionary(dictionary);

	len_output=NUM_HASH*NUM_BITS;

	cout<<"\n\tSet len_output to: "<<len_output<<std::endl;





	timer.start_time(0,1,1);

	auto start=std::chrono::system_clock::now();

	for(int n=0; n<n_batches; n++){
		set_embdata_dev[n]=malloc_shared<char>(batch_size*NUM_STR*NUM_REP*len_output, queues.back());
		memset(set_embdata_dev[n],0,batch_size*NUM_STR*NUM_REP*len_output);
	}

	timer.end_time(0,1,1);

	auto end=std::chrono::system_clock::now();

	auto time=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

	std::cout<<"\tTime to allocate embedded strings (shared malloc): "<<(float)time/1000<<"sec"<<std::endl<<std::endl;

	timer.start_time(0,1,2);


	uint32_t len_p=samplingrange+1;

	//TODO: Check this

	int (*p)=new int[NUM_STR*NUM_CHAR*len_p];

	generate_random_string(p, len_p);

	timer.end_time(0,1,2);





	timer.start_time(0,1,3);

	int n_fast=0;
	int n_slow=0;


	int idx_fastest=0;
	int idx_slowest=0;

	int num_dev=queues.size();

	int number_of_testing_batches=2*num_dev; // test_batches;


	std::vector<long> times;
	vector<std::vector<long>> time_on_dev(num_dev,std::vector<long>());


	{
		std::vector<buffer<int,1>> buffers_p;

		std::vector<buffer<char,2>> buffers_oristrings;

		std::vector<buffer<int,1>> buffers_lshnumber;

		std::vector<buffer<char,1>> buffers_embdata;

		std::vector<buffer<uint8_t,1>> buffers_dict;

		std::vector<buffer<int,1>> buffers_len_oristrings;

		std::vector<buffer<uint32_t,1>> buffers_samplingrange;

		std::vector<buffer<uint32_t,1>> buffers_len_output;

		std::vector<buffer<tuple<int,int>>> buffers_rev_hash;


		int n=0;
		int dev=0;

		std::cout<<"\tStart profiling on devices..."<<std::endl<<std::endl;


		start=std::chrono::system_clock::now();



	// ---------------------------------------

		for(auto &q:queues){


			for(int i=0; i<2; i++){

				auto start=std::chrono::system_clock::now();

				uint32_t size_p=static_cast<unsigned int>(NUM_STR*NUM_CHAR*(samplingrange+1));

				buffers_p.emplace_back( buffer<int,1>(p,range<1>{size_p}) );

				buffers_oristrings.emplace_back( buffer<char, 2>(reinterpret_cast<char*>((char*)oristrings[n*batch_size]),range<2>{batch_size,LEN_INPUT}) );

				buffers_lshnumber.emplace_back( buffer<int, 1>(lshnumber.data(),range<1>{lshnumber.size()}) );

				unsigned int size_emb=static_cast<unsigned int>(batch_size*NUM_STR*NUM_REP*len_output);

				buffers_embdata.emplace_back( buffer<char, 1> (reinterpret_cast<char*>(set_embdata_dev[n]), range<1>{size_emb}, {property::buffer::use_host_ptr()}) );

				buffers_dict.emplace_back( buffer<uint8_t,1>(dictionary,range<1>{256}) );

				buffers_len_oristrings.emplace_back( buffer<int,1>(len_oristrings.data()+n*batch_size,range<1>(batch_size)) );

				uint32_t samprange=samplingrange;

				buffers_samplingrange.emplace_back( buffer<uint32_t,1>(&samprange,range<1>(1)) );

				buffers_len_output.emplace_back( buffer<uint32_t, 1>(&len_output,range<1>{1}) );

				buffers_rev_hash.emplace_back( buffer<tuple<int,int>>(rev_hash.data(),range<1>(rev_hash.size())));

				parallel_embedding_while_loop_gath( q, buffers_len_oristrings[n], buffers_oristrings[n], buffers_embdata[n], batch_size, buffers_lshnumber[n], buffers_p[n], buffers_len_output[n], buffers_samplingrange[n], buffers_dict[n], buffers_rev_hash[n]);

				q.wait();

				auto end=std::chrono::system_clock::now();

				if(i>0){
					times.emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
				}

				n++;
			}

			dev++;


		}


		for(auto t:times){
			cout<<"\tTimes kernel: "<<(float)t/1000<<"sec"<<std::endl;
		}

		std::vector<int> iter_per_dev;


		if(num_dev>1){
			auto max_iter = std::max_element(times.begin(),times.end());
			auto min_iter = std::min_element(times.begin(),times.end());

			long slowest=*max_iter;
			long fastest=*min_iter;

			idx_slowest=max_iter-times.begin();
			idx_fastest=min_iter-times.begin();


			n_slow=floor(((float)fastest/(float)(fastest+slowest))*(n_batches-number_of_testing_batches));


			n_fast=n_batches-number_of_testing_batches-n_slow;

			iter_per_dev.resize(num_dev);
			iter_per_dev[idx_slowest]=n_slow;
			iter_per_dev[idx_fastest]=n_fast;

		}else if(num_dev==1){

			n_slow=0;
			n_fast=(n_batches-number_of_testing_batches);
			idx_fastest=0;
			idx_slowest=0;
			iter_per_dev.emplace_back(n_fast);

		}


		cout<<"\n\tn_fast: "<<n_fast<<std::endl;
		cout<<"\tn_slow: "<<n_slow<<std::endl;

		cout<<"\tid_fastest: "<<idx_fastest<<std::endl;
		cout<<"\tid_slowest: "<<idx_slowest<<std::endl;


		// --------------------

		timer.end_time(0,1,3);

		end=std::chrono::system_clock::now();

		cout<<"\tTotal time for profiling: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<std::endl<<std::endl;;

		std::cout<<"\n\tStart computation..."<<std::endl<<std::endl;


		timer.start_time(0,1,4);
		start=std::chrono::system_clock::now();



		dev=0;

		for(int i=0; i<num_dev; i++){

//			cout<<"From "<<n<<" to "<<n+iter_per_dev[i]<<std::endl;

			int iter=0;

			while(iter<iter_per_dev[i]){

				uint32_t size_p=static_cast<unsigned int>(NUM_STR*NUM_CHAR*(samplingrange+1));

				buffers_p.emplace_back( buffer<int,1>(p,range<1>{size_p}) );

				buffers_oristrings.emplace_back( buffer<char, 2>(reinterpret_cast<char*>(oristrings[n*batch_size]),range<2>{batch_size,LEN_INPUT}) );

				buffers_lshnumber.emplace_back( buffer<int, 1>(lshnumber.data(),range<1>{lshnumber.size()}) );

				unsigned int size_emb=static_cast<unsigned int>(batch_size*NUM_STR*NUM_REP*len_output);

				buffers_embdata.emplace_back( buffer<char, 1> (reinterpret_cast<char*>(set_embdata_dev[n]), range<1>{size_emb}, {property::buffer::use_host_ptr()}) );

				buffers_dict.emplace_back( buffer<uint8_t,1>(dictionary,range<1>{256}) );

				buffers_len_oristrings.emplace_back( buffer<int,1>(len_oristrings.data()+n*batch_size,range<1>(batch_size)) );

				uint32_t samprange=samplingrange;

				buffers_samplingrange.emplace_back( buffer<uint32_t,1>(&samprange,range<1>(1)) );

				buffers_len_output.emplace_back( buffer<uint32_t, 1>(&len_output,range<1>{1}) );

				buffers_rev_hash.emplace_back( buffer<tuple<int,int>>(rev_hash.data(),range<1>(rev_hash.size())));

				parallel_embedding_while_loop_gath(queues[i], buffers_len_oristrings[n], buffers_oristrings[n], buffers_embdata[n], batch_size, buffers_lshnumber[n], buffers_p[n], buffers_len_output[n], buffers_samplingrange[n], buffers_dict[n], buffers_rev_hash[n]);

				n++;
				iter++;

			}

			dev++;
		}

	}

	end=std::chrono::system_clock::now();

	cout<<"\tTime for actual computation: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<std::endl;

	timer.end_time(0,1,4);
//	delete[] p;
	return p;
}



void sequential_embedding_wrapper_if(std::vector<queue> &queues, std::vector<int> &len_oristrings, char (*oristrings)[LEN_INPUT], char** &set_embdata_dev, unsigned int batch_size, uint32_t n_batches, std::vector<int> &lshnumber, uint32_t &len_output){

	std::cout<< "Selected: Parallel embedding - no while improved version with if"<<std::endl;



	std::atomic<int> task(0);

	// DICTIONARY

	uint8_t dictionary[256]={0};
	inititalize_dictionary(dictionary);

	len_output=lshnumber.size();

	cout<<"\n\tlen output: "<<len_output<<std::endl;



	auto start=std::chrono::system_clock::now();

	for(int n=0; n<n_batches; n++){
			set_embdata_dev[n]=malloc_shared<char>(batch_size*NUM_STR*NUM_REP*len_output, queues.back());
	}

	auto end=std::chrono::system_clock::now();

	auto time=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

	std::cout<<"Time to allocate by shared malloc: "<<(float)time/1000<<"sec"<<std::endl;

	uint32_t len_p=samplingrange+1;

	int (*p)=new int[NUM_STR*NUM_CHAR*len_p];

	generate_random_string(p, len_p);

	std::vector<std::thread> threads;

	for(auto &q : queues){

		threads.push_back(thread([&](){


			while(true){

				int n=task.fetch_add(1);

				if(n<n_batches){
					// set_embdata_gpu[n]=(char*)malloc_shared(batch_size*NUM_STR*NUM_REP*len_output*sizeof(char), cpu_queue.get_device(), cpu_queue.get_context());
	//				set_embdata_dev[n]=malloc_shared<char>(batch_size*NUM_STR*NUM_REP*len_output, gpu_queue);

					sequential_embedding_if(q, len_oristrings.data()+n*batch_size, (char*) oristrings[n*batch_size], (char*)set_embdata_dev[n], batch_size, lshnumber, p, len_p, len_output, dictionary);
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



void parallel_embedding_USM_while_wrapper(std::vector<queue> &queues, std::vector<int> &len_oristrings, char (*oristrings)[LEN_INPUT], char** &set_embdata_dev, unsigned int batch_size, uint32_t n_batches, std::vector<int> &lshnumber, uint32_t &len_output){
	std::cout<< "Selected: Parallel embedding - USM while loop version"<<std::endl;

	uint8_t dictionary[256]={0};
	inititalize_dictionary(dictionary);



	uint32_t len_p=samplingrange+1;

	len_output=lshnumber.size();

	for(int n=0; n<n_batches; n++){
		set_embdata_dev[n]=(char*)malloc_shared(batch_size*NUM_STR*NUM_REP*len_output*sizeof(char), queues.back().get_device(), queues.back().get_context());
	}




	int (*p)=new int[NUM_STR*NUM_CHAR*len_p];

	generate_random_string(p, len_p);

//	parallel_embeddingUSM(device_queue, len_oristrings.data(), (char*) oristrings, (char**)set_embdata_dev, batch_size, lshnumber, p, len_p, len_output, dictionary);

	parallel_embeddingUSM_while(queues.back(), len_oristrings.data(), (char*) oristrings, (char**)set_embdata_dev, batch_size, lshnumber, p, len_p, len_output, dictionary);

	delete[] p;
}


void parallel_embedding_USM_wrapper(vector<queue> &queues, vector<int> &len_oristrings, char (*oristrings)[LEN_INPUT], char** &set_embdata_dev, unsigned int batch_size, uint32_t n_batches, std::vector<int> &lshnumber, uint32_t &len_output){
	std::cout<< "Selected: Parallel embedding - USM version"<<std::endl;

	uint8_t dictionary[256]={0};
	inititalize_dictionary(dictionary);

	len_output=lshnumber.size();

	cout<<"Len output"<<len_output<<std::endl;

	for(int n=0; n<n_batches; n++){
		set_embdata_dev[n]=(char*)malloc_shared(batch_size*NUM_STR*NUM_REP*len_output*sizeof(char), queues.back().get_device(), queues.back().get_context());
		memset(set_embdata_dev[n],0,batch_size*NUM_STR*NUM_REP*len_output*sizeof(char));
	}

	uint32_t len_p=samplingrange+1;



	int *p=new int[NUM_STR*NUM_CHAR*len_p];

	generate_random_string(p, len_p);

	ofstream outFile;

	outFile.open(string("p"+to_string(queues.front().get_device().is_gpu()?1:0)), ios::out | ios::trunc);

	if (outFile.is_open()) {
		for(int i=0; i<NUM_STR; i++){
			for(int j=0; j<NUM_CHAR; j++){

				for(int k=0; k<len_p; k++){

					outFile<<p[ABSPOS_P(i,j,k,len_p)]<<" ";

				}
			}
		}
	}


//	parallel_embeddingUSM(queues.back(), len_oristrings.data(), (char*) oristrings, (char**)set_embdata_dev, batch_size, lshnumber, p, len_p, len_output, dictionary);
	parallel_embeddingUSM_while(queues.back(), len_oristrings.data(), (char*) oristrings, (char**)set_embdata_dev, batch_size, lshnumber, p, len_p, len_output, dictionary);

	delete[] p;
}



int main(int argc, char **argv) {

//	__itt_pause();

	int device=0;
	int alg_number[3]={0,0,0};

	unsigned int batch=30000;
	unsigned int n_batches=10;



	if (argc==11){

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
		if(argc>10){
			test_batches=atoi(argv[10]);
		}



	}
	else{
		std::cerr<<"usage: ./embedjoin inputdata 0/1(cpu/gpu) [0-2]step1 [0-1]step2 [0-1]step3\n"<<std::endl;
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


	//HASH

	std::vector<int> a; // the random vector for second level hash table
	int (*hash_lsh)[NUM_BITS] = new int[NUM_HASH][NUM_BITS];

	std::vector<int> lshnumber;

	//INPUT STRINGS

	char (*oristrings)[LEN_INPUT];
	oristrings = new char[NUM_STRING][LEN_INPUT];
	std::vector<string> oridata_modified;
	std::vector<int> len_oristrings;



	std::vector<tuple<int,int,int,int,int>> buckets(NUM_STRING*NUM_STR*NUM_HASH*NUM_REP);

	std::vector<std::tuple<int,int,int,int,int,int>> candidates;

	std::vector<tuple<int,int>> buckets_delimiter;

	std::vector<int> candidates_start;


	std::vector<queue> queues;

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


	/**
	 *
	 * INITIALIZATION
	 *
	 * */



	timer.start_time(2,0,0);


	timer.start_time(0,0,0);


	srand(11110);
	initialization(len_oristrings, oristrings, oridata_modified, hash_lsh, a, lshnumber);

	timer.start_time(0,0,3);


	int k=0;

	vector<tuple<int,int>> rev_hash(lshnumber.size(), make_tuple(-1,-1));

	for(int i=0; i<NUM_HASH; i++){

		for(int j=0; j<NUM_BITS; j++){

			if(get<0>(rev_hash[hash_lsh[i][j]])!=-1){

				// Find last pos
				int t=hash_lsh[i][j];

				while(get<1>(rev_hash[t])!=-1){

					t=get<1>(rev_hash[t]);

				}
				rev_hash.emplace_back(make_tuple(k,-1));
				get<1>(rev_hash[t])=rev_hash.size()-1;
			}else {
				get<0>(rev_hash[hash_lsh[i][j]]) = k;
			}
			k++;
		}

	}

	timer.end_time(0,0,3);


	timer.end_time(0,0,0);


	std::cerr << "Start parallel algorithm..." << std::endl<<std::endl;



	timer.start_time(0,1,0);


	char **set_embdata_dev=(char**)malloc_shared<char*>(n_batches, queues.back());


	/**
	 *
	 *
	 * EMBEDDING STEP
	 *
	 *
	 **/


//	__itt_resume();

    int *p=parallel_embedding_while_loop_2dev_gath_wrapper(queues, len_oristrings, oristrings, set_embdata_dev, batch, n_batches, lshnumber, len_output, rev_hash);


    for(auto &q : queues){
    	q.wait();
    }
//    __itt_pause();

    timer.end_time(0,1,0);



	cout<<"Time: "<<timer.get_step_time(0,1,0)<<"sec"<<std::endl;



#if PRINT_EMB
		print_embedded( set_embdata_dev, len_output, batch, string("embedded"+to_string(device)+".txt"));
#endif

	timer.start_time(1,0,0);


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




	timer.start_time(0,2,0);



	len_output=NUM_HASH*NUM_BITS;

	create_buckets_without_lshnumber_offset_2dev_NEW_wrapper(queues, (char**)set_embdata_dev, buckets, n_batches, batch, (int*)hash_lsh, a, lshnumber, len_output);

	for(auto &q:queues){
		q.wait();
	}


	timer.end_time(0,2,0);


	cout<<"Time buckets creation: "<<timer.get_step_time(0,2,0)<<"sec"<<std::endl;

	timer.start_time(0,3,0);



//	 auto policy_sort = dpl::execution::make_device_policy<class PolicySort>(queues.front());

//	 cl::sycl::buffer<tuple<int,int,int,int,int>,1> buf(buckets.data(),range<1>{buckets.size()});
//	 auto buf_begin = dpl::begin(buf, sycl::read_write);
//	 auto buf_end   = dpl::end(buf, sycl::read_write);

	tbb::parallel_sort(buckets.begin(), buckets.end(), [](std::tuple<int,int,int,int,int> e1, std::tuple<int,int,int,int,int> e2) {
	 		 return ( ( get<0>(e1)<get<0>(e2) ) ||
	 				 ( get<0>(e1)==get<0>(e2) && get<1>(e1)<get<1>(e2) ) ||
	 				 ( get<0>(e1)==get<0>(e2) && get<1>(e1)==get<1>(e2) && get<2>(e1)<get<2>(e2) )  ||
	 				 ( get<0>(e1)==get<0>(e2) && get<1>(e1)==get<1>(e2) && get<2>(e1)==get<2>(e2) && get<3>(e1)<get<3>(e2) ) ||
	 				 ( get<0>(e1)==get<0>(e2) && get<1>(e1)==get<1>(e2) && get<2>(e1)==get<2>(e2) && get<3>(e1)==get<3>(e2) && get<4>(e1)<get<4>(e2) )); } );


	timer.end_time(0,3,0);



#if PRINT_BUCK
	 print_buckets(buckets, string("buckets"+to_string(device)+".txt"));
#endif

	 /**
	  *
	  * INITIALIZATION FOR CANDIDATE GENERATION
	  *
	  *
	  * **/




	 timer.start_time(0,4,0);


//	 initialize_candidate_pairs_onDevice( queues, buckets, candidates );

	 initialize_candidate_pairs( queues, buckets, candidates );

	 timer.end_time(0,4,0);



	 std::cout<<timer.get_step_time(0,4,0)<<std::endl;



	 /**
	 *
	 *
	 * GENERATE CANDIDATE PAIRS STEP
	 *
	 *
	 * **/



	 timer.start_time(0,5,0);


	 len_output=NUM_HASH*NUM_BITS;
	 generate_candidates_without_lshnumber_offset_2dev_wrapper(queues, len_oristrings, (char*)oristrings, (char**)set_embdata_dev, buckets, batch, /*buckets_delimiter,*/ candidates, /*candidates_start,*/ (int *)hash_lsh, lshnumber, len_output/*, partitionsBucketsDelimiter, partitionsCandStart, partitionsBuckets, partitionsCandidates*/);


	 timer.end_time(0,5,0);



	 timer.start_time(0,6,0);

	std::cout<<"\n\nStarting candidate processing analysis..."<<std::endl;




	std::cout<<"\n\t\tCandidates size: "<<candidates.size()<<std::endl;


	 timer.start_time(0,6,1);


	 	 vector<std::tuple<int,int>> verifycan;

//	 dpl::execution::device_policy par_policy = dpl::execution::make_device_policy(cpu_selector{});

//			auto remove_policy = dpl::execution::make_device_policy(queues.back());

	 	 candidates.erase(remove_if(oneapi::dpl::execution::par, candidates.begin(), candidates.end(),[](std::tuple<int,int,int,int,int,int> e){return (get<4>(e)>K_INPUT || (get<5>(e)!=0) || get<0>(e)==get<2>(e));}), candidates.end());


	 timer.end_time(0,6,1);


	 std::cout<<std::endl;
	 std::cout<<"\tRemove some candidates: "<<timer.get_step_time(0,6,1)<<std::endl;


	std::cout<<"\n\t\tCandidates to process (after filtering): "<<candidates.size();


	 timer.start_time(0,6,2);


	 tbb::parallel_sort( candidates.begin(), candidates.end(), [](tuple<int,int,int,int,int,int> e1, tuple<int,int,int,int,int,int> e2) {
		 return ( ( get<0>(e1)<get<0>(e2) ) ||
				 ( get<0>(e1)==get<0>(e2) && get<1>(e1)<get<1>(e2) ) ||
				 ( get<0>(e1)==get<0>(e2) && get<1>(e1)==get<1>(e2) && get<2>(e1)<get<2>(e2) )  ||
				 ( get<0>(e1)==get<0>(e2) && get<1>(e1)==get<1>(e2) && get<2>(e1)==get<2>(e2) && get<3>(e1)<get<3>(e2) )
				  );
	 });


	 timer.end_time(0,6,2);



 	std::cout<<std::endl;
 	std::cout<<"\n\tSorting candidates freq: "<<timer.get_step_time(0,6,2)<<std::endl;


	std::cerr<<"\n\t\tCandidate after filter out: "<<candidates.size()<<std::endl;

#if PRINT_CAND
	print_candidate_pairs(candidates, string("candidates"+to_string(device)));
#endif


	/*
	 *
	 * COUNTING FREQUENCIES
	 *
	 * **/


	timer.start_time(0,6,3);



		std::vector<int> freq_uv;

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

	timer.end_time(0,6,3);


	std::cout<<std::endl;
	std::cout<<"\n\tCounting freq: "<<timer.get_step_time(0,6,3)<<std::endl;


	timer.start_time(0,6,4);


		candidates.erase(unique( candidates.begin(), candidates.end() ), candidates.end());


	timer.end_time(0,6,4);


	std::cout<<"\n\tMake uniq: "<<timer.get_step_time(0,6,4)<<std::endl;


	timer.start_time(0,6,5);


		for (int i = 0; i < candidates.size(); i++)
		{
			if (freq_uv[i] > countfilter )
			{
				verifycan.emplace_back(get<0>(candidates[i]),get<2>(candidates[i]));
			}
		}


	timer.end_time(0,6,5);



	std::cout<<"\n\tFilter out candidates: "<<timer.get_step_time(0,6,5)<<std::endl;


	int num_candidate=0;

	timer.start_time(0,6,6);



		tbb::parallel_sort(verifycan.begin(), verifycan.end());


	timer.end_time(0,6,6);

	std::cout<<"\n\tSort verifycan: "<<timer.get_step_time(0,6,6)<<std::endl;

	timer.start_time(0,6,7);


		verifycan.erase(unique(verifycan.begin(), verifycan.end()), verifycan.end());



	timer.end_time(0,6,7);



	std::cout<<"\n\tUniq verifycan: "<<timer.get_step_time(0,6,7)<<std::endl;




	cout<<"\nEnd candidates processing"<<std::endl;

	timer.end_time(0,6,0);


	/**
	 *
	 * EDIT DISTANCE CALCULATION
	 *
	 * */


	timer.start_time(0,7,0);

	unsigned int num_threads = std::thread::hardware_concurrency();

	cout<<"\nNumber of threads for edit distance: "<<num_threads<<std::endl;



	std::vector<std::thread> workers;
	std::atomic<int> verified(0);
	int to_verify=verifycan.size();

	int cont=0;
	std::mutex mt;


	std::cout<<"\n\tTo verify: "<<to_verify<<std::endl;

	for(int t=0; t<num_threads; t++){


		workers.push_back(std::thread([&](){

			while(true){

				int j=verified.fetch_add(1);

				if(j<to_verify){


					int first_str;
					int second_str;

					first_str=get<0>(verifycan[j]);
					second_str=get<1>(verifycan[j]);
//					int ed = edit_distance(oridata_modified[second_str].data(), len_oristrings[second_str]/*tmp_oridata[second_str].size()*/,oridata_modified[first_str].data(), len_oristrings[first_str] /*tmp_oridata[first_str].size()*/, K_INPUT);
					string tmp_str1=oridata_modified[first_str];
					string tmp_str2=oridata_modified[second_str];

					int ed = edit_distance(tmp_str2.data(), len_oristrings[second_str], tmp_str1.data(), len_oristrings[first_str] /*tmp_oridata[first_str].size()*/, K_INPUT);

					std::unique_lock<std::mutex> lk(mt);


					if(ed != -1) {
						cont++;
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

	timer.end_time(0,7,0);

	cout<<"\n\t\tNum output: "<<cont<<std::endl;


	timer.end_time(2,0,0);
	timer.end_time(1,0,0);

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

	std::string dev="";
	int count_dev=0;
	for(auto &q : queues){
		dev+=q.get_device().get_info<info::device::name>();
		dev+=count_dev==(queues.size()-1)?"": " && ";
		count_dev++;
	}

	timer.print_report(dev);

	std::string distinguisher="";

	if(device==0){
		distinguisher+="-CPU-";
	}else if(device==1){
		distinguisher+="-GPU-";
	}else if(device==2){
		distinguisher+="-BOTH-";
	}
	else{
		distinguisher+="-ERROR";
	}
	distinguisher+=std::to_string(batch);

	std::cout<<std::endl<<std::endl<<std::endl;
	{
		std::cout<<"Report:"<<std::endl<<std::endl;

		double t=timer.get_step_time(0,0,0);
		std::cout<<"Time read data: "<<t<<std::endl;

		t=timer.get_step_time(0,1,0);
		std::cout<<"Time PARALLEL embedding data:\t"<<t<<"sec"<<std::endl;

		t=timer.get_step_time(0,2,0);
		std::cout<<"Time PARALLEL buckets generation:\t"<< t<<"sec"<<std::endl;

		t=timer.get_step_time(0,3,0);
		std::cout<<"Time buckets sorting:\t"<< t <<"sec"<<std::endl;

		t=timer.get_step_time(0,4,0);
		std::cout<<"Time candidate initialization:\t"<< t<<"sec"<<std::endl;

		t=timer.get_step_time(0,5,0);
		std::cout<<"Time PARALLEL candidates generation:\t"<< t<<"sec"<<std::endl;

		t=timer.get_step_time(0,6,0);
		std::cout<<"Time candidates processing:\t"<< t<<"sec"<<std::endl;

		t=timer.get_step_time(0,6,2);
		std::cout<<"Time candidates sorting (within cand-processing):\t"<< t<<"sec"<<std::endl;

		t=timer.get_step_time(0,7,0);
		std::cout<<"Time compute edit distance:\t"<<t <<"sec"<<std::endl;

		t=timer.get_step_time(1,0,0);
		std::cout<<"Total time parallel join:\t"<< t<<"sec"<<std::endl;

		t=timer.get_step_time(2,0,0);
		std::cout<<"Total elapsed time :\t"<< t<<"sec"<<std::endl;

	}
	cout<<std::endl;


	{

		ofstream outFile;

		outFile.open("report-"+filename+distinguisher, ios::out | ios::trunc);
		std::string dev="";

		if(device==2){

			int count_dev=0;
			for(auto &q : queues){
				dev+=q.get_device().get_info<info::device::name>();
				dev+=count_dev==(queues.size()-1)?"": " && ";
				count_dev++;
			}

		}else{

			dev=queues.back().get_device().get_info<info::device::name>();

		}
		if (outFile.is_open()) {
			timer.print_report(dev, outFile);
		}
		if(PRINT_EACH_STEP==1){
			std::cerr<<"Attention, join time include the print on file time"<<std::endl;
		}
	}


	print_output("join_output_parallel.txt");

	return 0;

}



