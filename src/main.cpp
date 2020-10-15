#include "embedjoin.hpp"

int main(int argc, char **argv){

	Time timer;
	int device=0;
	size_t batch_size=0;
	size_t n_batches=0;
	string filename="";
	uint32_t samplingrange=0; // The maximum digit to embed, the range to sample
	uint32_t countfilter=0;   // Number of required matches (>T) for a pair of substrings to be considered as candidate
	uint32_t min_pts=10;

	if (argc==6){
		filename = argv[1];
		device=atoi(argv[2]);
		samplingrange=atoi(argv[3]);
		countfilter=atoi(argv[4]);
		batch_size=atoi(argv[5]);
	}
	else{
		std::cerr<<"usage: ./embedjoin input_data 0/1/2(cpu/gpu/both) samplingrange count_filter batch_size\n"<<std::endl;
		exit(-1);
	}

	vector<string> input_data;
	timer.start_time(init::read_dataset);
	read_dataset(input_data, filename);
	timer.end_time(init::read_dataset);
	OutputValues output_val;

	onejoin(input_data,batch_size,n_batches,device,samplingrange,countfilter,timer,output_val,"GEN320ks");

	string report_name=getReportFileName(device, batch_size);

	{
		ofstream out_file;
		out_file.open("report-GEN320ks"+report_name+".csv", ios::out | ios::trunc);

		if (out_file.is_open()) {
			timer.print_report(output_val.dev, output_val.num_candidates, output_val.num_outputs, out_file);
		}
	}

	return 0;
}
