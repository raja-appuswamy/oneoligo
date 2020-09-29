/*
 * main.cpp
 *
 *  Created on: 28 set 2020
 *      Author: eugenio
 */



#include "embedjoin.hpp"

using namespace std;


int main(int argc, char**argv){

	int device=0;

	size_t batch_size=30000;
	size_t n_batches=10;


	string filename="";

	int samplingrange=5000;
	int countfilter=1;

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

//	device=0;
//
//	batch_size=15000;
//	n_batches=30;
//
//
//	filename="gen320ks.txt";
//
//	samplingrange=5000;
//	countfilter=1;

	int nPts=10;
//	for(int nPts=0; nPts<50; nPts++){
		DBSCAN(filename, device, samplingrange, countfilter, batch_size, n_batches, nPts);
//	}

//	device=1;
//
//	batch_size=15000;
//	n_batches=30;
//
//
//	filename="gen320ks.txt";
//
//	samplingrange=5000;
//	countfilter=1;


	std::vector<tuple<int,int>> tmp_v1;
	std::vector<string> tmp_v2;
//	embed_join(filename, device, samplingrange, countfilter, batch_size, n_batches, false,  tmp_v1,  tmp_v2);
//
//	device=2;
//
//	batch_size=15000;
//	n_batches=30;
//
//
//	filename="gen320ks.txt";
//
//	samplingrange=5000;
//	countfilter=1;
//
//	embed_join(filename, device, samplingrange, countfilter, batch_size, n_batches);




	return 0;
}



