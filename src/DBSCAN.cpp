#ifndef DBSCAN_H
#define DBSCAN_H

#include "embedjoin.hpp"
#include <unordered_map>

using namespace std;

constexpr int UNDEFINED=-2;
constexpr int NOISE=-1;

void get_consensus(vector<string> &input_dataset, vector<int> &label, int max_string_len, vector<string> &output_dataset){

	map<int,vector<int>> clusters;

	int string_idx=0;
	for(auto&c:label){
		//if(c!=NOISE){
			clusters[c].emplace_back(string_idx);
		//}
		string_idx++;
	}
	std::cout<<clusters.size()<<std::endl;

	vector<uint32_t> counter(256,0);
	counter['A']=0;
	counter['C']=0;
	counter['T']=0;
	counter['G']=0;
	counter['N']=0;

	for(auto&c:clusters){
		string true_string="";
		for(int digit=0; digit<max_string_len; digit++){
			for(auto &string_idx:c.second){
				char ch=input_dataset[string_idx][digit];
				counter[ch]++;
			}
			auto max_ch=max_element(counter.begin(), counter.end());
			char true_ch=std::distance(counter.begin(), max_ch);
			if(true_ch!='A' && true_ch!='C' && true_ch!='G' && true_ch!='T' &&  true_ch!='N'){
				BOOST_LOG_TRIVIAL(error)<<"Error character"<<std::endl;
				exit(-1);
			}
			true_string+=true_ch;
			counter['A']=0;
			counter['C']=0;
			counter['T']=0;
			counter['G']=0;
			counter['N']=0;
		}
		output_dataset.emplace_back(true_string);
	}
}


void get_indexes(vector<tuple<int,int>> &similarity_results, unordered_map<int,vector<int>> &indexes, int max_index_str){


	for(int i=0; i<max_index_str; i++){
			indexes[i].emplace_back(i);
	}
	for(int i=0; i<similarity_results.size(); i++){
		indexes[get<0>(similarity_results[i])].emplace_back(get<1>(similarity_results[i]));
	}
	BOOST_LOG_TRIVIAL(debug)<<"Size indexes: "<<indexes.size();
}

vector<int> DBSCAN(unordered_map<int,vector<int>> &indexes, int nPts, size_t size_dataset){

	vector<int> label(size_dataset, UNDEFINED);
	int min_points=nPts;
	int cluster=0;

	for(int str=0; str<size_dataset; str++){

		if(label[str]!=UNDEFINED){
			continue;
		}
		vector<int> &neigh=indexes.at(str);
		if(neigh.size()<min_points){
			label[str]=NOISE;
			continue;
		}
		cluster+=1;
		label[str]=cluster;

		set<int> seed_set(indexes.at(str).begin()+1,indexes.at(str).end()); // +1 to exclude the point "str"
		for( auto  itr=seed_set.begin(); itr!= seed_set.end(); ++itr ){

			int q=*itr;
			if(label[q]==NOISE){
				label[q]=cluster;
			}
			if(label[q]!=UNDEFINED){
				continue;
			}
			label[q]=cluster;
			vector<int> &neigh_2=indexes.at(q);
			if(neigh_2.size()>=min_points){
				int prev_size=seed_set.size();
				seed_set.insert(neigh_2.begin()+1,neigh_2.end());
				if(prev_size!=seed_set.size()){
					itr=seed_set.begin();
				}
			}
		}
	}
	BOOST_LOG_TRIVIAL(info)<<"Number of cluster: "<<cluster;
	BOOST_LOG_TRIVIAL(info)<<"Min points: "<<min_points<<std::endl;
	return label;
}

void oneCluster(vector<string> &input_data, size_t batch_size, int device, uint32_t new_samplingrange, uint32_t new_countfilter, Time &timer, int nPts, string dataset_name){

	BOOST_LOG_TRIVIAL(info) << "Starting OneCluster..."<<std::endl;

	size_t len_input=input_data[0].size();
	int min_points=nPts;
	vector<string> total_output_dataset;
	int chunk_num=0;
	bool end=false;

	while(!end){
		
		timer.start_time(cluster::total);

		timer.start_time(cluster::init);
		
		vector<idpair> similarity_results;
		unordered_map<int,vector<int>> indexes;
		vector<string> output_dataset;
		vector<int> labels;


		random_shuffle(input_data.begin(),input_data.end());

		size_t range=std::min(clustering_chunk_size,input_data.size());

		std::cout<<range<<std::endl;

		vector<string> input_chunk;

		if(input_data.size()==0 && total_output_dataset.size()>0){
			//Begins the last iteration
			end=true;
			if(chunk_num>1){
				input_chunk=move(total_output_dataset);
			}
		}
		else{
			input_chunk.insert(input_chunk.end(), make_move_iterator(input_data.begin()),make_move_iterator(input_data.begin()+range));
			input_data.erase(input_data.begin(),input_data.begin()+range);
			BOOST_LOG_TRIVIAL(debug) <<"Input chunk size: "<<input_chunk.size()<<std::endl;
		}

		if(input_chunk.size()==0){
			break;
		}

		timer.end_time(cluster::init);

		BOOST_LOG_TRIVIAL(info) <<"Computing OneJoin..."<<std::endl;
		
		timer.start_time(cluster::onejoin);
		OutputValues ov;
		similarity_results=onejoin( input_chunk, batch_size, device, new_samplingrange, new_countfilter, timer, ov, alg::cluster, 0, dataset_name);
		timer.end_time(cluster::onejoin);
		
		BOOST_LOG_TRIVIAL(debug)<<"\tSize of db: "<<input_data.size();
		BOOST_LOG_TRIVIAL(debug)<<"\tSize of results: "<<similarity_results.size();

		BOOST_LOG_TRIVIAL(info)<<"Creating indexes"<<std::endl;

		timer.start_time(cluster::create_indexes);

		size_t prev_size=similarity_results.size();

		similarity_results.reserve(similarity_results.size()*2);

		for(int idx=0; idx<prev_size; idx++){
			idpair p=similarity_results[idx];
			similarity_results.push_back(make_tuple(get<1>(p),get<0>(p)));
		}

		int max_index_str=input_chunk.size();
		
		timer.start_time(cluster::sort);

		tbb::parallel_sort(similarity_results.begin(), similarity_results.end(), [](idpair &e1, idpair &e2){
			return get<0>(e1)<get<0>(e2);
		});

		timer.end_time(cluster::sort);

		get_indexes(similarity_results,indexes,max_index_str);

		
		timer.end_time(cluster::create_indexes);

	
		BOOST_LOG_TRIVIAL(info)<<"Start DBSCAN algorithm"<<std::endl;
				
				
		timer.start_time(cluster::dbscan);

		labels=DBSCAN(indexes,min_points,input_chunk.size());

		timer.end_time(cluster::dbscan);
		BOOST_LOG_TRIVIAL(debug)<<"Time oneDBSCAN: "<<(float)timer.get_step_time(cluster::dbscan);

		timer.start_time(cluster::consensus);

		get_consensus(input_chunk, labels, len_input, output_dataset);

		timer.end_time(cluster::consensus);

		BOOST_LOG_TRIVIAL(debug)<<"Time consensus: "<<(float)timer.get_step_time(cluster::consensus);

		// ofstream out_file("consensus_results_chunk_"+to_string(chunk_num));
		if(end){
			BOOST_LOG_TRIVIAL(debug)<<"Saving final chunk results...";
			ofstream out_file("consensus_results_chunk_"+to_string(chunk_num));
			for(auto&s:output_dataset){
				out_file<<s<<std::endl;
			}
		}

		chunk_num++;

		total_output_dataset.insert(total_output_dataset.end(), make_move_iterator(output_dataset.begin()), make_move_iterator(output_dataset.end()));
		
		timer.end_time(cluster::total);
	}
	
}

#endif