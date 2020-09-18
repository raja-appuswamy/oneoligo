#include "embedjoin.hpp"

using namespace std;


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
};







void print_embedded( char **output, int len_output, int batch_size, std::string filename ){

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
};

void print_buckets( vector<tuple<int,int,int,int,int>> &buckets, std::string filename){

	ofstream outFile;

	outFile.open(filename, ios::out | ios::trunc);

	if (outFile.is_open()) {
		for(int i=0; i<buckets.size(); i++){

			outFile<<get<0>(buckets[i])<<", "<<get<1>(buckets[i])<<", "<<get<2>(buckets[i])<<", "<<get<3>(buckets[i])<<", "<<get<4>(buckets[i])<<std::endl;

		}
	}
};

void print_candidate_pairs( vector<tuple<int,int,int,int,int,int>> &candidates, std::string filename ){

	ofstream outFile;

	outFile.open(filename, ios::out | ios::trunc);

	if (outFile.is_open()) {
		for(int i=0; i<candidates.size(); i++){

			outFile<</*get<4>(candidates[i])<<", "<<get<5>(candidates[i])<<", "<<*/get<0>(candidates[i])<<", "<<get<1>(candidates[i])<<", "<<get<2>(candidates[i])<<", "<<get<3>(candidates[i])<<", "<<get<4>(candidates[i])<<", "<<get<5>(candidates[i])<<std::endl;

		}
	}
};

void print_configuration(int batch_size,int n_batches, int len_output, int countfilter, int samplingrange){
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
};
