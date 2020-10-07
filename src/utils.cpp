#include "embedjoin.hpp"

using namespace std;

void print_oristrings( vector<string> &oristrings, vector<int> len )
{
	string filename_output="oristrings.txt";
	ofstream outFile;

	outFile.open(filename_output, ios::out | ios::trunc);

	if (outFile.is_open()) {
		for(auto &s:oristrings){
			outFile<<std::endl;
		}
	}
};


void print_embedded( char **output, size_t len_output, vector<batch_hdr> &batch_hdrs, size_t num_strings, std::string filename ){

	ofstream outFile;
	outFile.open(filename, ios::out | ios::trunc);
	size_t max_batch_size=batch_hdrs[0].size;
	if (outFile.is_open()) {
		for(int i=0; i<num_strings; i++){
			for(int j=0; j<NUM_STR; j++ ){
				for(int k=0; k<NUM_REP; k++){
					for(int t=0; t<len_output; t++){
						if(output[(int)(i/max_batch_size)][ABSPOS((int)(i%max_batch_size),j,k,t,len_output)]==0){
							break;
						}
						outFile<<output[(int)(i/max_batch_size)][ABSPOS((int)(i%max_batch_size),j,k,t,len_output)];
					}
					outFile<<std::endl;
				}
			}
		}
	}
};


void print_buckets( vector<buckets_t> &buckets, std::string filename){

	ofstream outFile;
	outFile.open(filename, ios::out | ios::trunc);

	if (outFile.is_open()) {
		
		for(int i=0; i<buckets.size(); i++){
			outFile<<get<0>(buckets[i])<<", "<<get<1>(buckets[i])<<", "<<get<2>(buckets[i])<<", "<<get<3>(buckets[i])<<", "<<get<4>(buckets[i])<<std::endl;
		}
	}
};

void print_candidate_pairs( vector<candidate_t> &candidates, std::string filename ){

	ofstream outFile;
	outFile.open(filename, ios::out | ios::trunc);

	if (outFile.is_open()) {
		for(int i=0; i<candidates.size(); i++){
			outFile<<get<0>(candidates[i])<<", "<<get<1>(candidates[i])<<", "<<get<2>(candidates[i])<<", "<<get<3>(candidates[i])<<std::endl;
		}
	}
};


void print_configuration(int batch_size,int n_batches, size_t len_output, size_t num_input_strings, int countfilter, int samplingrange){
	std::cout<<"\nParameter selected:"<<std::endl;
	std::cout<<"\tNum of strings:\t\t\t\t\t"<<num_input_strings<<std::endl;
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
};
