#ifndef DBSCAN_H
#define DBSCAN_H

#include "embedjoin.hpp"
#include <unordered_map>

using namespace std;

#define UNDEFINED -1
#define NOISE -2


std::vector<int> getNeighbors(std::vector<tuple<int,int>> &similarity_results, int str, unordered_map<int,vector<int>> &indexes){

	std::vector<int> neigh;
//	neigh.emplace_back(str); //Including str

	neigh.emplace_back(str);

//	return indexes[str];
//	if (indexes.find(str) != indexes.end()){
//		return indexes[str];

		neigh.insert(neigh.end(),indexes[str].begin(),indexes[str].end());

//		int start=indexes[str];
//
//		for(int i=start; i<similarity_results.size(); i++){
//
//			if(str!=std::get<0>(similarity_results[i])){
//				break;
//			}
//			neigh.emplace_back( std::get<1>(similarity_results[i]) );
//
//		}
//	}
	return neigh;

}


void get_indexes(vector<tuple<int,int>> &similarity_results, unordered_map<int,vector<int>> &indexes, int max_index_str){

	int j=0;

//	for(int i=0; i<max_index_str; i++){
//		indexes[i].emplace_back(i);
//	}

	if(similarity_results.size()>0){
		indexes.emplace(get<0>(similarity_results[0]),vector<int>{});
	}


	for(int i=0; i<similarity_results.size(); i++){

		indexes[get<0>(similarity_results[i])].emplace_back(get<1>(similarity_results[i]));
		indexes[get<1>(similarity_results[i])].emplace_back(get<0>(similarity_results[i]));

//		if( (get<0>(similarity_results[i])!=get<0>(similarity_results[i+1]))){
//			j++;
//			indexes.emplace(get<0>(similarity_results[i+1]),i+1);
//		}

	}





//	for(auto &t:similarity_results){
////		cout<<get<0>(t)<<" "<<get<1>(t)<<std::endl;
//
//	}

	std::for_each(indexes.begin(),indexes.end(),[](auto p){
		cout<<p.first<<": [";

		for(int i=0; i<p.second.size(); i++){
			cout<<p.second[i]<<" ";
		}
		cout<<"]"<<std::endl;
	});
//	indexes.clear();

	cout<<"Size indexes: "<<indexes.size()<<std::endl;

}

void DBSCAN(string filename, int device, int samplingrange, int countfilter, size_t batch_size, size_t n_batches, int minPts){

	vector<tuple<int,int>> similarity_results;

	vector<string> db;

	unordered_map<int,vector<int>> indexes;


	int min_points=minPts;
	int cluster=0;

	embed_join(filename, device, samplingrange, countfilter, batch_size, n_batches, true, similarity_results, db);

	cout<<"Size of db: "<<db.size()<<std::endl;
	cout<<"Size of results: "<<similarity_results.size()<<std::endl;


	vector<int> label_str(db.size(), UNDEFINED);

	tbb::parallel_sort(similarity_results.begin(), similarity_results.end(), [](tuple<int,int> e1, tuple<int,int> e2){
		return get<0>(e1)<get<0>(e2);
	});


	cout<<"Run DBSCAN..."<<std::endl;
	int max_index_str=db.size();

	get_indexes(similarity_results,indexes,max_index_str);



	int str=0;


	for(str=0; str<max_index_str; str++){

		if(label_str[str]!=UNDEFINED){
			continue;
		}

		vector<int> neigh=getNeighbors(similarity_results,str,indexes);


		if(neigh.size()<min_points){
			label_str[str]=NOISE; //NOISE
			continue;
		}

		cluster+=1;

		label_str[str]=cluster;

		set<int> seed_set(neigh.begin()+1,neigh.end()); // +1 to exclude the point "str"



		for( auto  itr=seed_set.begin(); itr!= seed_set.end(); ++itr ){

			int q=*itr;

			if(label_str[q]==NOISE){
				label_str[q]=cluster;
			}

			if(label_str[q]!=UNDEFINED){
				continue;
			}

			label_str[q]=cluster;

			vector<int> neigh_2=getNeighbors(similarity_results,q,indexes);

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

#endif
