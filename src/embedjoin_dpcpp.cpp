
#include "embedjoin.hpp"
#include "Time.cpp"


using namespace cl::sycl;
using namespace oneapi::std;
using namespace std;



//embedrange: the length of truncation, recommended to be the average length of strings (you could try smaller values to further save the embedding time)




int samplingrange=5000; //the maximum digit to embed, the range to sample

int countfilter=1;// Number of required matches (>T) for a pair of substrings to be considered as candidate




int test_batches=2;

std::string filename="";

Time timer;


std::vector<int> indices;

std::vector<idpair> outputs;

std::vector<std::string> tmp_oridata;



void setuplsh(int (*hash_lsh)[NUM_BITS], std::vector<int> &a, std::vector<int> &lshnumber)
{

	for (int i = 0; i < NUM_HASH; i++){
		for (int j = 0; j < NUM_BITS; j++){
			hash_lsh[i][j] = rand() % (samplingrange);
		}
	}

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


	for (int i = 0; i < NUM_HASH; i++){
		for (int j = 0; j < NUM_BITS; j++){
			hash_lsh[i][j] = lower_bound(lshnumber.begin(), lshnumber.end(), hash_lsh[i][j]) - lshnumber.begin();
		}
	}
}



void readdata(std::vector<size_t> &len_oristrings, char (*oristrings)[LEN_INPUT], std::vector<std::string> &oridata_modified )
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



void initialization( std::vector<size_t> &len_oristrings, char (*oristrings)[LEN_INPUT], std::vector<string> &oridata_modified, int (*hash_lsh)[NUM_BITS], std::vector<int> &a, std::vector<int> &lshnumber )
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
		dictionary[static_cast<uint8_t>('A')]=0;
		dictionary[static_cast<uint8_t>('C')]=1;
		dictionary[static_cast<uint8_t>('G')]=2;
		dictionary[static_cast<uint8_t>('T')]=3;

	}else if (NUM_CHAR == 26 || NUM_CHAR == 25){
		int j=0;
		for(int i=(int)'A'; i<=(int)'Z'; i++){
			dictionary[i]=j;
			j++;
		}

	}

}



void parallel_embedding(queue &device_queue, buffer<size_t,1> &buffer_len_oristrings, buffer<char,2> &buffer_oristrings, buffer<char,1> &buffer_embdata, size_t batch_size, buffer<int,1> &buffer_lshnumber, buffer<int,1> &buffer_p, buffer<size_t,1> &buffer_len_output, buffer<uint32_t,1> &buffer_samplingrange, buffer<uint8_t,1> &buffer_dict, buffer<std::tuple<int,int>> &buffer_rev_hash){


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

				  while (partdigit < len_out && j > acc_lshnumber[partdigit]){


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



void create_buckets(queue &device_queue, char **embdata, buffer<std::tuple<int,int,int,int,int>,1> &buffer_buckets, buffer<size_t,1> &buffer_batch_size, size_t split_size, buffer<size_t,1> &buffer_split_offset, buffer<uint32_t,1> &buffer_a, buffer<size_t,1> &buffer_len_output, buffer<uint8_t,1> &buffer_dict){

		std::cout << "\n\tTask: Buckets Generation\t";
		std::cout << "Device: " << device_queue.get_device().get_info<info::device::name>() << std::endl;

		std::cout<<"\t\tSplit size: "<<split_size<<std::endl;

		range<2> glob_range(split_size*NUM_STR*NUM_REP,NUM_HASH);
		range<3> local_range(250,1,1);

		std::cout<<"\t\tGlobal range: "<<"("<<glob_range[0]<<", "<<glob_range[1]<<")"<<std::endl;


    {


		device_queue.submit([&](handler &cgh){


		//Executing kernel


			auto acc_buckets = buffer_buckets.get_access<access::mode::write>(cgh);
			auto acc_dict = buffer_dict.get_access<access::mode::read>(cgh);

			auto acc_a = buffer_a.get_access<access::mode::read>(cgh);

			auto acc_batch_size=buffer_batch_size.get_access<access::mode::read>(cgh);

			auto acc_len_output=buffer_len_output.get_access<access::mode::read>(cgh);

	        auto acc_split_offset=buffer_split_offset.get_access<access::mode::read>(cgh);






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



	}

}



void create_buckets_wrapper(vector<queue> &queues, char **embdata, vector<tuple<int,int,int,int,int>> &buckets, size_t n_batches, size_t batch_size, int* hash_lsh, vector<int> &a, vector<int> &lshnumber, size_t len_output){

	std::cout<< "Selected: Create buckets - without lshnumber offset"<<std::endl;

	std::cout<<"Len output: "<<len_output<<std::endl;

	int num_dev=queues.size();


	int n_fast=0; // Number of batches to allocate to the fastest device
	int n_slow=0; // Number of batches to allocate to the slowest device


	int idx_fastest=0; // Id of fastest device
	int idx_slowest=0; // Id of slowest device


	// Number batches to use for profiling
	// (2 batches per queue/device)

	int number_of_testing_batches=2*num_dev;



	vector<long> times;


	{
		vector<size_t> split_size;


		uint8_t dictionary[256]={0};

		inititalize_dictionary(dictionary);


		vector<size_t> offset;

		vector<sycl::buffer<tuple<int,int,int,int,int>>> buffers_buckets;
		vector<sycl::buffer<size_t,1>> buffers_batch_size;
		vector<sycl::buffer<size_t,1>> buffers_split_size;
		vector<sycl::buffer<size_t,1>> buffers_split_offset;
		vector<sycl::buffer<uint32_t,2>> buffers_hash_lsh;
		vector<sycl::buffer<uint32_t,1>> buffers_a;
		vector<sycl::buffer<uint32_t,1>> buffers_lshnumber;
		vector<sycl::buffer<size_t,1>> buffers_len_output;
		vector<sycl::buffer<uint8_t,1>>  buffers_dict;


		timer.start_time(0,2,1);

		int n=0; // Global number of iteration

		int  dev=0; // Device index

		cout<<"\tStart profiling on devices..."<<std::endl<<std::endl;

		/**
		 *
		 * Profiling kernel on devices by using the test batches
		 *
		 * */


		for(auto &q:queues){


			for(int i=0; i<2; i++){

				// Two kernel are chosen, since the first one
				// includes kernel compiling time

				auto start=std::chrono::system_clock::now();

				offset.emplace_back(2*batch_size*dev+i*batch_size);

				size_t loc_split_size=batch_size;

				cout<<"\n\tSet offset to: "<<offset[n]<<std::endl;

				buffers_buckets.emplace_back(sycl::buffer<tuple<int,int,int,int,int>,1>(static_cast<tuple<int,int,int,int,int>*>(buckets.data()+offset.back()*NUM_REP*NUM_HASH*NUM_STR),range<1>{loc_split_size*NUM_STR*NUM_HASH*NUM_REP}, {sycl::property::buffer::use_host_ptr()})); // Wrong dimension

				buffers_a.emplace_back(buffer<uint32_t,1>((uint32_t*)a.data(),range<1>{a.size()}));

				buffers_hash_lsh.emplace_back(buffer<uint32_t,2>((uint32_t*)hash_lsh, range<2>{NUM_HASH,NUM_BITS}));

				buffers_dict.emplace_back(buffer<uint8_t,1>(dictionary,range<1>{256}));

				buffers_batch_size.emplace_back(buffer<size_t,1>(&batch_size, range<1>{1}));

				buffers_len_output.emplace_back(buffer<size_t,1>(&len_output, range<1>{1}));

				buffers_split_offset.emplace_back(buffer<size_t,1> (&offset[n], range<1>{1}));

				create_buckets(q, embdata, buffers_buckets[n], buffers_batch_size[n], loc_split_size, buffers_split_offset[n], buffers_a[n], buffers_len_output[n], buffers_dict[n]);

				q.wait();

				auto end=std::chrono::system_clock::now();

				// Save the time only for the second kernel execution for each device
				// because the first run includes the compiling time

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

			// If there are 2 devices, compute the number of batches
			// to allocate to devices.
			// Note that at most 2 devices can be used handled


			// Get the max and min time measured during profiling.
			// The max time is associated with the slowest device.
			// The min time is associated with the fastest device.

			auto max_iter = std::max_element(times.begin(),times.end());
			auto min_iter = std::min_element(times.begin(),times.end());

			long slowest=*max_iter;
			long fastest=*min_iter;


			// Get the position in the time vector corresponding
			// to the min and max time.
			// These positions correspond to the device positions
			// in the device queues vector.

			idx_slowest=max_iter-times.begin();
			idx_fastest=min_iter-times.begin();


			// Compute the number of batches based on time measured

			n_slow=floor(((float)fastest/(float)(fastest+slowest))*(n_batches-number_of_testing_batches));

			n_fast=n_batches-number_of_testing_batches-n_slow;



		}else if(num_dev==1){

			// If there is only one device, all remaining batches
			// are given to the first (and only) device of the queue.

			// Assign remaining batches as n_slow or n_fast is
			// the same in this case since there is only 1 device;

			n_slow=(n_batches-number_of_testing_batches);
			idx_slowest=0;

			idx_fastest=0;
			n_fast=(n_batches-number_of_testing_batches);


		}


		cout<<"\n\tn_fast: "<<n_fast<<std::endl;
		cout<<"\tn_slow: "<<n_slow<<std::endl;

		cout<<"\tid_fastest: "<<idx_fastest<<std::endl;
		cout<<"\tid_slowest: "<<idx_slowest<<std::endl;


		dev=0;



		timer.end_time(0,2,1);

		cout<<"Time for measure computation: "<<(float)timer.get_step_time(0,2,1)<<std::endl;

		/**
		 *
		 * Start computation for remaining batches in parallel
		 * on all devices available
		 *
		 * **/

		timer.start_time(0,2,2);


		offset.emplace_back(number_of_testing_batches*batch_size);

		cout<<"\n\tStart computation..."<<std::endl<<std::endl;


		for(int i=0; i<num_dev; i++){

			split_size.emplace_back((i==idx_slowest?n_slow:n_fast)*batch_size);

			offset.emplace_back(offset.back()+(dev==0?0:split_size[dev-1]));

			size_t loc_split_size=split_size[dev];

			cout<<"Offset: "<<offset.back()<<std::endl;


			buffers_buckets.emplace_back(sycl::buffer<tuple<int,int,int,int,int>,1>(static_cast<tuple<int,int,int,int,int>*>(buckets.data()+offset.back()*NUM_REP*NUM_HASH*NUM_STR),range<1>{loc_split_size*NUM_STR*NUM_HASH*NUM_REP}, {sycl::property::buffer::use_host_ptr()})); // Wrong dimension

			buffers_a.emplace_back(buffer<uint32_t,1>((uint32_t*)a.data(),range<1>{a.size()}));

			buffers_hash_lsh.emplace_back(buffer<uint32_t,2>((uint32_t*)hash_lsh, range<2>{NUM_HASH,NUM_BITS}));

			buffers_dict.emplace_back(buffer<uint8_t,1>(dictionary,range<1>{256}));

			buffers_batch_size.emplace_back(buffer<size_t,1>(&batch_size, range<1>{1}));

			buffers_len_output.emplace_back(buffer<size_t,1>(&len_output, range<1>{1}));

			buffers_split_offset.emplace_back(buffer<size_t,1> (&offset.back(), range<1>{1}));


			create_buckets(queues[i], embdata, buffers_buckets[n], buffers_batch_size[n], loc_split_size, buffers_split_offset[n], buffers_a[n], buffers_len_output[n], buffers_dict[n]);


			n++;

			dev++;

		}


	}

	timer.end_time(0,2,2);


	cout<<"Time for actual computation: "<<timer.get_step_time(0,2,2)<<std::endl;


	cout<<"End of scope"<<std::endl;

}


void generate_candidates(queue &device_queue, buffer<size_t,1> &buffer_len_oristrings, buffer<char,2> &buffer_oristrings, char **embdata, buffer<tuple<int,int,int,int,int>,1> &buffer_buckets, buffer<size_t,1> &buffer_buckets_offset, buffer<size_t,1> &buffer_batch_size, buffer<tuple<int,int,int,int,int,int>,1> &buffer_candidates, size_t candidate_size, buffer<size_t,1> &buffer_len_output ){


		cout << "\n\tTask: Candidate Pairs Generation\t";
		std::cout << "Device: " << device_queue.get_device().get_info<info::device::name>() << std::endl;

		device_queue.submit([&](handler &cgh){

			auto acc_buckets = buffer_buckets.get_access<access::mode::read>(cgh);

			auto acc_candidate = buffer_candidates.get_access<access::mode::write>(cgh);

			auto acc_len = buffer_len_oristrings.get_access<access::mode::read>(cgh);

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



void generate_candidates_wrapper(vector<queue>& queues, vector<size_t> &len_oristrings, char* oristrings, char **embdata, vector<tuple<int,int,int,int,int>> &buckets, size_t batch_size, vector<std::tuple<int,int,int,int,int,int>>& candidate, int * local_hash_lsh, vector<int> &lshnumber, size_t len_output){

	cout << "Selected: Generate candidates - without lshnumber offset"<< std::endl;

	cout<<"Len output: "<<len_output<<std::endl;


	{

	int n_fast=0; // Number of batches to allocate to the fastest device
	int n_slow=0; // Number of batches to allocate to the slowest device


	int idx_fastest=0; // Id of fastest device
	int idx_slowest=0; // Id of slowest device

	int num_dev=queues.size();


	vector<vector<size_t>> size_cand(num_dev,vector<size_t>());

	vector<uint32_t> number_of_iter(num_dev);

	vector<size_t> buckets_offset;

	vector<vector<int>> reverse_index;

	vector<buffer<int,1>> buffers_reverse_index;


	vector<buffer<char,2>> buffers_oristrings;

	vector<buffer<int, 1>> buffers_candidate_start;

	vector<buffer<tuple<int,int,int,int,int>>> buffers_buckets;

	vector<buffer<tuple<int,int>>> buffers_bucket_delimiter;

	vector<buffer<int, 2>> buffers_hash_lsh;

	vector<buffer<tuple<int,int,int,int,int,int>>> buffers_candidates;

	vector<buffer<size_t,1>> buffers_len;

	vector<buffer<size_t, 1>> buffers_batch_size;

	vector<buffer<size_t, 1>> buffers_len_output;

	vector<buffer<size_t,1>> buffers_buckets_offset;


	vector<long> times;



	// Select a number of candidates to use for profiling.
	// The size can change:
	// too big can let to a big overhead
	// too small can reduce the quality of profiling

	size_t size_for_test=0.01*candidate.size();

	cout<<"Size (num candidates) for profiling: "<<size_for_test<<std::endl;



	cout<<"\n\tNew size for test: "<<size_for_test<<std::endl;


	timer.start_time(0,5,1);


	int dev=0;
	int n=0;


	cout<<"\n\tStart profiling..."<<std::endl;

	/**
	 *
	 * Profiling kernel on devices by using the test batches
	 *
	 * */


	for(auto &q:queues){


		for(int i=0; i<2; i++){

			auto start=std::chrono::system_clock::now();


			size_t start_b=get<0>(candidate[size_for_test*n]);
			size_t end_b=get<2>(candidate[size_for_test*n+size_for_test-1])-1;

			size_t size_buckets=end_b-start_b+1;

			buckets_offset.emplace_back(start_b);

			cout<<"\n\tIter "<<dev<<". Start buckets at "<<size_for_test*n<<": "<<start_b<<std::endl;
			cout<<"\tIter "<<dev<<". End buckets at "<<size_for_test*n + size_for_test-1<<": "<<end_b<<std::endl;
			cout<<"\n\tBuckets size: "<<size_buckets<<std::endl;

			buffers_oristrings.emplace_back( buffer<char,2>(oristrings,range<2>{NUM_STRING,LEN_INPUT}/*, {property::buffer::use_host_ptr()}*/));


			buffers_buckets.emplace_back( buffer<tuple<int,int,int,int,int>>(buckets.data()+start_b,range<1>{size_buckets}/*, {property::buffer::use_host_ptr()}*/));


			cout<<"\tCand size: "<<size_for_test<<std::endl;

			buffers_hash_lsh.emplace_back( buffer<int, 2>(reinterpret_cast<int*>(local_hash_lsh),range<2>{NUM_HASH,NUM_BITS}/*, {property::buffer::use_host_ptr()}*/));

			buffers_candidates.emplace_back( buffer<tuple<int,int,int,int,int,int>>(candidate.data()+n*size_for_test,range<1>{size_for_test}/*, {property::buffer::use_host_ptr()}*/));

			buffers_len.emplace_back( buffer<size_t,1>(len_oristrings.data(),range<1>{len_oristrings.size()}/*, {property::buffer::use_host_ptr()}*/));

			buffers_batch_size.emplace_back( buffer<size_t, 1>(&batch_size,range<1>{1}));

			buffers_len_output.emplace_back( buffer<size_t, 1>(&len_output,range<1>{1}));

			buffers_buckets_offset.emplace_back( buffer<size_t,1>(&buckets_offset.back(),range<1>{1}));

			generate_candidates(q, buffers_len[n], buffers_oristrings[n], embdata, buffers_buckets[n], buffers_buckets_offset[n], buffers_batch_size[n], buffers_candidates[n], size_for_test, buffers_len_output[n]);


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

	size_t remaining_size=candidate.size()-size_for_test*2*num_dev;





	cout<<"\tRemaining size: "<<remaining_size<<std::endl;


	if(num_dev>1){

		// If there are 2 devices, compute the number of batches
		// to allocate to devices.
		// Note that at most 2 devices can be used handled


		// Get the max and min time measured during profiling.
		// The max time is associated with the slowest device.
		// The min time is associated with the fastest device.

		auto max_iter = std::max_element(times.begin(),times.end());
		auto min_iter = std::min_element(times.begin(),times.end());

		long slowest=*max_iter;
		long fastest=*min_iter;


		// Get the position in the time vector corresponding
		// to the min and max time.
		// These positions correspond to the device positions
		// in the device queues vector.

		idx_slowest=max_iter-times.begin();
		idx_fastest=min_iter-times.begin();


		n_slow=floor(((float)fastest/(float)(fastest+slowest))*remaining_size);
		n_fast=remaining_size-n_slow;
		cout<<"\n\tNumber of candidates to assign to the faster device: "<<n_fast<<std::endl;



		cout<<"\n\tNumber of candidates to assign to other device: "<<n_slow<<std::endl;



		size_t s=n_fast*sizeof(candidate[0]);

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

				int remind=n_fast%(num_kernels);

				size_cand[idx_fastest].emplace_back(n_fast/(num_kernels)+remind);

			}else{

				size_cand[idx_fastest].emplace_back(n_fast/(num_kernels));

			}

		}


		s=n_slow*sizeof(candidate[0]);

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

				int remind=n_slow%(num_kernels);

				size_cand[idx_slowest].emplace_back(n_slow/(num_kernels)+remind);

			}else{

				size_cand[idx_slowest].emplace_back(n_slow/(num_kernels));

			}

		}

	}else if(num_dev==1){

		// If there is only one device, all remaining batches
		// are given to the first (and only) device of the queue.

		n_slow=0;
		idx_fastest=0;
		idx_slowest=0;

		vector<int> tmp_sizes;


		n_slow=0;


		size_t s=remaining_size*sizeof(candidate[0]);
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
				size_cand[idx_slowest].emplace_back(remaining_size/(num_kernels)+remind);
			}else{
				size_cand[idx_slowest].emplace_back(remaining_size/(num_kernels));
			}
		}


	}


	cout<<std::endl;
	for(auto d:size_cand){
		for(auto s:d){
			cout<<"\tSize: "<<s<<std::endl;
		}
	}


	cout<<"\n\tn_fast: "<<n_fast<<std::endl;
	cout<<"\tn_slow: "<<n_slow<<std::endl;

	cout<<"\tid_fastest: "<<idx_fastest<<std::endl;
	cout<<"\tid_slowest: "<<idx_slowest<<std::endl;


	dev=0;

	timer.end_time(0,5,1);

	timer.start_time(0,5,2);

	size_t offset_cand=size_for_test*2*num_dev;


	for(auto &q : queues){

		int iter=0;

		while(iter<size_cand[dev].size()){

			cout<<"\n\tSize cand[dev]: "<<size_cand[dev][iter]<<std::endl;

			size_t start_b=get<0>(candidate[offset_cand]);
			size_t end_b=get<2>((candidate.data()+offset_cand)[size_cand[dev][iter]-1])-1;

			size_t size_buckets=end_b-start_b+1;

			buckets_offset.emplace_back(start_b);

			cout<<"\n\tIter "<<dev<<". Start buckets at "<<offset_cand<<": "<<start_b<<std::endl;
			cout<<"\tIter "<<dev<<". End buckets at "<<offset_cand + size_cand[dev][iter]-1<<": "<<end_b<<std::endl;
			cout<<"\n\tBuckets size: "<<size_buckets<<std::endl;


			buffers_oristrings.emplace_back( buffer<char,2>(oristrings,range<2>{NUM_STRING,LEN_INPUT}, {property::buffer::use_host_ptr()}));


			buffers_buckets.emplace_back( buffer<tuple<int,int,int,int,int>>(buckets.data()+start_b,range<1>{size_buckets}, {property::buffer::use_host_ptr()}));

			cout<<"\tCand size: "<<size_cand[dev][iter]<<std::endl;

			buffers_hash_lsh.emplace_back( buffer<int, 2>(reinterpret_cast<int*>(local_hash_lsh),range<2>{NUM_HASH,NUM_BITS}, {property::buffer::use_host_ptr()}));

			buffers_candidates.emplace_back( buffer<tuple<int,int,int,int,int,int>>(candidate.data()+offset_cand,range<1>{size_cand[dev][iter]}, {property::buffer::use_host_ptr()}));

			buffers_len.emplace_back( buffer<size_t,1>(len_oristrings.data(),range<1>{len_oristrings.size()}/*, {property::buffer::use_host_ptr()}*/));

			buffers_batch_size.emplace_back( buffer<size_t, 1>(&batch_size,range<1>{1}));

			buffers_len_output.emplace_back( buffer<size_t, 1>(&len_output,range<1>{1}));

			buffers_buckets_offset.emplace_back( buffer<size_t,1>(&buckets_offset.back(),range<1>{1}));


			generate_candidates(q, buffers_len[n], buffers_oristrings[n], embdata, buffers_buckets[n], buffers_buckets_offset[n], buffers_batch_size[n], buffers_candidates[n], size_cand[dev][iter], buffers_len_output[n]);


			offset_cand+=size_cand[dev][iter];

			n++;
			iter++;
		}

		dev++;


	}

	}

	timer.end_time(0,5,2);

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










void initialize_candidate_pairs(vector<queue>& queues, vector<tuple<int,int,int,int,int>> &buckets, vector<std::tuple<int,int,int,int,int,int>> &candidates ){


	cout<<"\nInitialize candidate vector"<<std::endl;
	/*
	 * Compute the boundary ( starting index and size ) of each buckets in the 1-D vector
	 *
	 * */
	vector<tuple<int,int>> buckets_delimiter;

	auto start=std::chrono::system_clock::now();

		int j=0;
		size_t size=0;

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



	std::cout<<"\tTime cand-init: remove element: "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec"<<std::endl;

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


		for(int b=0; b<buckets_delimiter.size(); b++){
			int n=get<1>(buckets_delimiter[b]);
			size+=((n*(n-1))/2);
		}


		start=std::chrono::system_clock::now();

		std::cout<<"Size: "<<size<<std::endl;

		candidates.resize(size);
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



void parallel_embedding_wrapper(std::vector<queue> &queues, vector<size_t> &len_oristrings, char (*oristrings)[LEN_INPUT], char** set_embdata_dev, size_t batch_size, size_t n_batches, std::vector<int> &lshnumber, size_t &len_output, std::vector<tuple<int,int>> &rev_hash){

	std::cout<< "Selected: Parallel embedding - while loop version"<<std::endl;


	// DICTIONARY

	uint8_t dictionary[256]={0};
	inititalize_dictionary(dictionary);


	cout<<"\n\tLen output: "<<len_output<<std::endl;

	timer.start_time(0,1,2);


	uint32_t len_p=samplingrange+1;


	int *p=new int[NUM_STR*NUM_CHAR*len_p];

	generate_random_string(p, len_p);

	timer.end_time(0,1,2);


	timer.start_time(0,1,3);

	int n_fast=0; // Number of batches to allocate to the fastest device
	int n_slow=0; // Number of batches to allocate to the slowest device


	int idx_fastest=0; // Id of fastest device
	int idx_slowest=0; // Id of slowest device

	int num_dev=queues.size();


	// Number batches to use for profiling
	// (2 batches per queue/device)

	int number_of_testing_batches=2*num_dev;


	// Store the time taken by each device to run 1 kernel

	std::vector<long> times;


	{

		/**
		 *
		 * Each queue and each kernel has its own copy of data (sycl::buffer).
		 * Also read-only data shared by all kernels, are replicated once for
		 * each kernel, in order to reduce dependencies among different kernels
		 * and device queues.
		 * Thus, a vector of sycl::buffer is created for each array accessed
		 * in the kernel.
		 *
		 * Buffer are created inside the scope of this function, so buffer destructor
		 * is used as synch method to put back data in the host
		 *
		 * */

		std::vector<buffer<int,1>> buffers_p;

		std::vector<buffer<char,2>> buffers_oristrings;

		std::vector<buffer<int,1>> buffers_lshnumber;

		std::vector<buffer<char,1>> buffers_embdata;

		std::vector<buffer<uint8_t,1>> buffers_dict;

		std::vector<buffer<size_t,1>> buffers_len_oristrings;

		std::vector<buffer<uint32_t,1>> buffers_samplingrange;

		std::vector<buffer<size_t,1>> buffers_len_output;

		std::vector<buffer<tuple<int,int>>> buffers_rev_hash;


		int n=0;   // number of iterations
		int dev=0; // device index

		std::cout<<"\tStart profiling on devices..."<<std::endl<<std::endl;


		/**
		 *
		 * Profiling kernel on devices by using the test batches.
		 * The test is executed on the 2 devices sequentially, waiting
		 * at the end of each testing kernel.
		 *
		 * */


		for(auto &q:queues){

			// Two kernel are chosen, since the first one
			// includes kernel compiling time

			for(int i=0; i<2; i++){


				auto start=std::chrono::system_clock::now();

				size_t size_p=static_cast<size_t>(NUM_STR*NUM_CHAR*(samplingrange+1));

				buffers_p.emplace_back( buffer<int,1>(p, range<1>{size_p}) );

				buffers_oristrings.emplace_back( buffer<char, 2>(reinterpret_cast<char*>((char*)oristrings[n*batch_size]),range<2>{batch_size,LEN_INPUT}) );

				buffers_lshnumber.emplace_back( buffer<int, 1>(lshnumber.data(),range<1>{lshnumber.size()}) );

				size_t size_emb=static_cast<size_t>(batch_size*NUM_STR*NUM_REP*len_output);

				buffers_embdata.emplace_back( buffer<char, 1> (reinterpret_cast<char*>(set_embdata_dev[n]), range<1>{size_emb}, {property::buffer::use_host_ptr()}) );

				buffers_dict.emplace_back( buffer<uint8_t,1>(dictionary,range<1>{256}) );

				buffers_len_oristrings.emplace_back( buffer<size_t,1>(len_oristrings.data()+n*batch_size,range<1>(batch_size)) );

				uint32_t samprange=samplingrange;

				buffers_samplingrange.emplace_back( buffer<uint32_t,1>(&samprange,range<1>(1)) );

				buffers_len_output.emplace_back( buffer<size_t, 1>(&len_output,range<1>{1}) );

				buffers_rev_hash.emplace_back( buffer<tuple<int,int>>(rev_hash.data(),range<1>(rev_hash.size())));

				parallel_embedding( q, buffers_len_oristrings[n], buffers_oristrings[n], buffers_embdata[n], batch_size, buffers_lshnumber[n], buffers_p[n], buffers_len_output[n], buffers_samplingrange[n], buffers_dict[n], buffers_rev_hash[n]);

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

			// If there are 2 devices, compute the number of batches
			// to allocate to devices.
			// Note that at most 2 devices can be used handled


			// Get the max and min time measured during profiling.
			// The max time is associated with the slowest device.
			// The min time is associated with the fastest device.

			auto max_iter = std::max_element(times.begin(),times.end());
			auto min_iter = std::min_element(times.begin(),times.end());

			long slowest=*max_iter;
			long fastest=*min_iter;


			// Get the position in the time vector corresponding
			// to the min and max time.
			// These positions correspond to the device positions
			// in the device queues vector.

			idx_slowest=max_iter-times.begin();
			idx_fastest=min_iter-times.begin();


			// Compute the number of batches based on time measured

			n_slow=floor(((float)fastest/(float)(fastest+slowest))*(n_batches-number_of_testing_batches));

			n_fast=n_batches-number_of_testing_batches-n_slow;

			iter_per_dev.resize(num_dev);

			iter_per_dev[idx_slowest]=n_slow;
			iter_per_dev[idx_fastest]=n_fast;

		}else if(num_dev==1){

			// If there is only one device, all remaining batches
			// are given to the first (and only) device of the queue.

			n_slow=0;
			idx_fastest=0;
			idx_slowest=0;

			n_fast=(n_batches-number_of_testing_batches);

			iter_per_dev.emplace_back(n_fast);

		}


		cout<<"\n\tn_fast: "<<n_fast<<std::endl;
		cout<<"\tn_slow: "<<n_slow<<std::endl;

		cout<<"\tid_fastest: "<<idx_fastest<<std::endl;
		cout<<"\tid_slowest: "<<idx_slowest<<std::endl;


		/**
		 *
		 * Start computation for remaing batches in parallel
		 * on all devices available
		 *
		 * **/

		timer.end_time(0,1,3);


		cout<<"\tTotal time for profiling: "<<(float)timer.get_step_time(0,1,3)<<std::endl<<std::endl;;

		std::cout<<"\n\tStart computation..."<<std::endl<<std::endl;


		timer.start_time(0,1,4);



		dev=0;

		for(auto &q:queues){


			int iter=0;

			while(iter<iter_per_dev[dev]){

				size_t size_p=static_cast<size_t>(NUM_STR*NUM_CHAR*(samplingrange+1));

				buffers_p.emplace_back( buffer<int,1>(p,range<1>{size_p}) );

				buffers_oristrings.emplace_back( buffer<char, 2>(reinterpret_cast<char*>(oristrings[n*batch_size]),range<2>{batch_size,LEN_INPUT}) );

				buffers_lshnumber.emplace_back( buffer<int, 1>(lshnumber.data(),range<1>{lshnumber.size()}) );

				size_t size_emb=static_cast<size_t>(batch_size*NUM_STR*NUM_REP*len_output);

				buffers_embdata.emplace_back( buffer<char, 1> (reinterpret_cast<char*>(set_embdata_dev[n]), range<1>{size_emb}, {property::buffer::use_host_ptr()}) );

				buffers_dict.emplace_back( buffer<uint8_t,1>(dictionary,range<1>{256}) );

				buffers_len_oristrings.emplace_back( buffer<size_t,1>(len_oristrings.data()+n*batch_size,range<1>(batch_size)) );

				uint32_t samprange=samplingrange;

				buffers_samplingrange.emplace_back( buffer<uint32_t,1>(&samprange,range<1>(1)) );

				buffers_len_output.emplace_back( buffer<size_t, 1>(&len_output,range<1>{1}) );

				buffers_rev_hash.emplace_back( buffer<tuple<int,int>>(rev_hash.data(),range<1>(rev_hash.size())));

				parallel_embedding(q, buffers_len_oristrings[n], buffers_oristrings[n], buffers_embdata[n], batch_size, buffers_lshnumber[n], buffers_p[n], buffers_len_output[n], buffers_samplingrange[n], buffers_dict[n], buffers_rev_hash[n]);

				n++;
				iter++;

			}

			dev++;
		}

	}


	cout<<"\tTime for actual computation: "<<(float)timer.get_step_time(0,1,4)<<std::endl;

	timer.end_time(0,1,4);

	delete[] p;
}

void print_output( std::string file_name )
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


int main(int argc, char **argv) {

//	__itt_pause();

	int device=0;

	size_t batch_size=30000;
	size_t n_batches=10;




	if (argc==7){

		filename = argv[1];
		device=atoi(argv[2]);

		samplingrange=atoi(argv[3]);

		countfilter=atoi(argv[4]);

		batch_size=atoi(argv[5]);

		n_batches=atoi(argv[6]);



	}
	else{
		std::cerr<<"usage: ./embedjoin input_data 0/1/2(cpu/gpu/both) len_input_strings count_filter batch_size number_of_batches\n"<<std::endl;
		exit(-1);
	}


	//OUTPUT STRINGS

    size_t len_output=NUM_HASH*NUM_BITS;//samplingrange;

	print_configuration(batch_size, n_batches, len_output, countfilter, samplingrange);

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
	std::vector<size_t> len_oristrings;


	std::vector<tuple<int,int,int,int,int>> buckets(NUM_STRING*NUM_STR*NUM_HASH*NUM_REP);

	std::vector<std::tuple<int,int,int,int,int,int>> candidates;


	std::vector<queue> queues;

	if(device==0 || device==2){ // Selected CPU or both

		queues.push_back(queue(cpu_selector{}, asyncHandler, property::queue::in_order()));

	}

	if(device==1 || device==2){ // Selected GPU or both

		try{

			queue tmp_queue(gpu_selector{}, asyncHandler, property::queue::in_order());

			queues.push_back(std::move(tmp_queue));

		}catch(std::exception& e){ // No GPU available, use CPU if not selected yet
			std::cout<<"Attention: no GPU device detected. The program will run on CPU."<<std::endl;

			if( queues.size()==0 ){
				queues.push_back(queue(cpu_selector{}, asyncHandler, property::queue::in_order()));
			}
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


	/**
	 *
	 *  Compute the position in which to put each
	 *  selected char in embedding according the lsh bit
	 *
	 * */

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


	/**
	 *
	 *
	 * EMBEDDING STEP
	 *
	 *
	 **/



	timer.start_time(0,1,0);

	timer.start_time(0,1,1);





	char **set_embdata_dev=(char**)malloc_shared<char*>(n_batches, queues.back());

	for(int n=0; n<n_batches; n++){
		set_embdata_dev[n]=malloc_shared<char>(batch_size*NUM_STR*NUM_REP*len_output, queues.back());
		memset(set_embdata_dev[n],0,batch_size*NUM_STR*NUM_REP*len_output);
	}

	timer.end_time(0,1,1);




//	__itt_resume();

    parallel_embedding_wrapper(queues, len_oristrings, oristrings, set_embdata_dev, batch_size, n_batches, lshnumber, len_output, rev_hash);


    for(auto &q : queues){
    	q.wait();
    }
//    __itt_pause();

    timer.end_time(0,1,0);



	cout<<"Time: "<<timer.get_step_time(0,1,0)<<"sec"<<std::endl;



#if PRINT_EMB
		print_embedded( set_embdata_dev, len_output, batch_size, string("embedded"+to_string(device)+".txt"));
#endif

	timer.start_time(1,0,0);


//	/**
//	 *
//	 *
//	 * CREATE BUCKETS STEP
//	 *
//	 *
//	 * **/




	timer.start_time(0,2,0);


	create_buckets_wrapper(queues, (char**)set_embdata_dev, buckets, n_batches, batch_size, (int*)hash_lsh, a, lshnumber, len_output);

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


	 generate_candidates_wrapper(queues, len_oristrings, (char*)oristrings, (char**)set_embdata_dev, buckets, batch_size, /*buckets_delimiter,*/ candidates, /*candidates_start,*/ (int *)hash_lsh, lshnumber, len_output/*, partitionsBucketsDelimiter, partitionsCandStart, partitionsBuckets, partitionsCandidates*/);




	 timer.end_time(0,5,0);



	 timer.start_time(0,6,0);

	 for(auto &q:queues){
		 q.wait();
	 }

	 /**
	  *
	  *
	  * CANDIDATES PROCESSING
	  *
	  *
	  * */


	std::cout<<"\n\nStarting candidate processing analysis..."<<std::endl;




	std::cout<<"\n\t\tCandidates size: "<<candidates.size()<<std::endl;


	 timer.start_time(0,6,1);


	 	 vector<std::tuple<int,int>> verifycan;


	 candidates.erase(std::remove_if(oneapi::dpl::execution::par_unseq, candidates.begin(), candidates.end(),[](std::tuple<int,int,int,int,int,int> e){return (get<4>(e)>K_INPUT || (get<5>(e)!=0) || get<0>(e)==get<2>(e));}), candidates.end());


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
	 * EDIT DISTANCE
	 *
	 * */


	timer.start_time(0,7,0);

	uint32_t num_threads = std::thread::hardware_concurrency();

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
	distinguisher+=std::to_string(batch_size);

	std::cout<<std::endl<<std::endl<<std::endl;
	{
		std::cout<<"Summary:"<<std::endl<<std::endl;

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


