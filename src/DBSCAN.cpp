#ifndef DBSCAN_H
#define DBSCAN_H

#include "embedjoin.hpp"
#include <unordered_map>

using namespace std;

constexpr int UNDEFINED=-1;
constexpr int NOISE=-2;


void get_consensus(){

	return;

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

void DBSCAN(vector<string> &input_data, size_t batch_size, size_t n_batches, int device, uint32_t new_samplingrange, uint32_t new_countfilter, Time &timer, int nPts, string dataset_name){

	vector<idpair> similarity_results;
	unordered_map<int,vector<int>> indexes;
	int min_points=nPts;
	int cluster=0;

	similarity_results=onejoin(input_data,batch_size,n_batches,device,new_samplingrange,new_countfilter,timer,"GEN320ks");

	cout<<"Size of db: "<<input_data.size()<<std::endl;
	cout<<"Size of results: "<<similarity_results.size()<<std::endl;



	tbb::parallel_sort(similarity_results.begin(), similarity_results.end(), [](idpair e1, idpair e2){
		return get<0>(e1)<get<0>(e2);
	});

	cout<<"Run DBSCAN..."<<std::endl;
	int max_index_str=input_data.size();
	get_indexes(similarity_results,indexes,max_index_str);

	for(int nP=0; nP<1000; nP+=50){
		vector<int> label_str(input_data.size(), UNDEFINED);
		min_points=nP;
		cluster=0;

		int str=0;
		for(str=0; str<max_index_str; str++){

			if(label_str[str]!=UNDEFINED){
				continue;
			}
			vector<int> &neigh=indexes.at(str);
			if(neigh.size()<min_points){
				label_str[str]=NOISE;
				continue;
			}
			cluster+=1;
			label_str[str]=cluster;

			set<int> seed_set(indexes.at(str).begin()+1,indexes.at(str).end()); // +1 to exclude the point "str"
			for( auto  itr=seed_set.begin(); itr!= seed_set.end(); ++itr ){

				int q=*itr;
				if(label_str[q]==NOISE){
					label_str[q]=cluster;
				}
				if(label_str[q]!=UNDEFINED){
					continue;
				}
				label_str[q]=cluster;
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
		cout<<"Number of cluster: "<<cluster<<std::endl;
	}
}

#endif
