#include "embedjoin.hpp"

int main(int argc, char **argv){

	Time timer;
	int device=0;
	size_t batch_size=0;
	size_t n_batches=0;
	string filename="";
	uint32_t samplingrange=0; // The maximum digit to embed, the range to sample
	uint32_t countfilter=0;   // Number of required matches (>T) for a pair of substrings to be considered as candidate

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

	vector<string> input_data;
	timer.start_time(init::read_dataset);
	read_dataset(input_data, filename);
	timer.end_time(init::read_dataset);

//	onejoin(input_data,batch_size,n_batches,device,samplingrange,countfilter,timer,"GEN320ks");

	vector<int> parameters={10,50,100,150,200,250,300,350};
//	vector<int> parameters={1,2,3,4,5,10,15,20,25,30,35,40};

	for(auto &min_pts:parameters){
		cout<<"oneDBSCAN min points: "<<min_pts<<std::endl;
		oneDBSCAN(input_data,batch_size,n_batches,device,samplingrange,countfilter,timer,min_pts,"GEN320ks");
	}
}
