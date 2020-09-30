/*
 * main.cpp
 *
 *  Created on: 30 set 2020
 *      Author: eugenio
 */
#include "embedjoin.hpp"

int main(int argc, char **argv){

	int device=0;
	size_t batch_size=30000;
	size_t n_batches=10;
	string filename="";
	int samplingrange=0; // The maximum digit to embed, the range to sample
	int countfilter=1;   // Number of required matches (>T) for a pair of substrings to be considered as candidate


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

	embed_join(filename,batch_size,n_batches,device,samplingrange,countfilter);
}




