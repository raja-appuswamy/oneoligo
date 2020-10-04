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
		if(c!=NOISE){
			clusters[c].emplace_back(string_idx);
		}
		string_idx++;
	}
	std::cout<<clusters.size()<<std::endl;

	vector<uint8_t> counter(256,0);
	counter['A']=0;
	counter['C']=0;
	counter['T']=0;
	counter['G']=0;
	counter['N']=0;

	for(auto&c:clusters){
//		std::cout<<"Cluster id: "<<c.first<<": "<<c.second.size()<<std::endl;
		string true_string="";
		for(int digit=0; digit<max_string_len; digit++){
			for(auto &string_idx:c.second){
				char ch=input_dataset[string_idx][digit];
				counter[(uint8_t)ch]++;
			}
			auto max_ch=max_element(counter.begin(), counter.end());
			char true_ch=std::distance(counter.begin(), max_ch);
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

	int j=0;

	for(int i=0; i<max_index_str; i++){
			indexes[i].emplace_back(i);
	}

	for(int i=0; i<similarity_results.size(); i++){
		indexes[get<0>(similarity_results[i])].emplace_back(get<1>(similarity_results[i]));
		indexes[get<1>(similarity_results[i])].emplace_back(get<0>(similarity_results[i]));
	}
//	std::for_each(indexes.begin(),indexes.end(),[](auto p){
//		cout<<p.first<<": [";
//
//		for(int i=0; i<p.second.size(); i++){
//			cout<<p.second[i]<<" ";
//		}
//		cout<<"]"<<std::endl;
//	});
	cout<<"Size indexes: "<<indexes.size()<<std::endl;
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
	cout<<"Number of cluster: "<<cluster<<" min_points: "<<min_points<<std::endl;
	return label;
}

void oneDBSCAN(vector<string> &input_data, size_t batch_size, size_t n_batches, int device, uint32_t new_samplingrange, uint32_t new_countfilter, Time &timer, int nPts, string dataset_name){

	vector<idpair> similarity_results;
	unordered_map<int,vector<int>> indexes;
	vector<string> output_dataset;
	vector<int> labels;
	size_t len_input=91;
	int min_points=nPts;

	cout<<"Computing edit distance."<<std::endl;
	similarity_results=onejoin(input_data,batch_size,n_batches,device,new_samplingrange,new_countfilter,timer,"GEN320ks");

	cout<<"\tSize of db: "<<input_data.size()<<std::endl;
	cout<<"\tSize of results: "<<similarity_results.size()<<std::endl;

	cout<<"Sorting results."<<std::endl;

	tbb::parallel_sort(similarity_results.begin(), similarity_results.end(), [](idpair e1, idpair e2){
		return get<0>(e1)<get<0>(e2);
	});

	cout<<"Creating indexes."<<std::endl;
	int max_index_str=input_data.size();
	get_indexes(similarity_results,indexes,max_index_str);
	cout<<"Run DBSCAN."<<std::endl;

	labels=DBSCAN(indexes,min_points,input_data.size());

	get_consensus(input_data, labels, len_input, output_dataset);

	ofstream out_file("consensus_results_min_points_"+to_string(min_points));
	for(auto&s:output_dataset){
		out_file<<s<<std::endl;
	}
}

#endif
