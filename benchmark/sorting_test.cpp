#include<CL/sycl.hpp>
#include<oneapi/dpl/execution>
#include<oneapi/dpl/algorithm>
#include<oneapi/dpl/iterator>
#include "tbb/parallel_sort.h"
using namespace sycl;
using namespace oneapi::dpl::execution;
using namespace std;

struct MyStruct{
    uint32_t idx_rand_str;
	uint32_t idx_hash_func;
	uint32_t hash_id;
	uint32_t idx_str;
	uint32_t idx_rep;
	MyStruct():idx_rand_str(0), idx_hash_func(0), hash_id(0), idx_str(0), idx_rep(0){}
	MyStruct(uint32_t id_rand_string, uint32_t id_hash_func, uint32_t hash_id, uint32_t id_string, uint32_t id_rep ):idx_rand_str(id_rand_string), idx_hash_func(id_hash_func),
		hash_id(hash_id), idx_str(id_string), idx_rep(id_string){}
	bool operator<(const MyStruct& rhs) const {return ( (idx_rand_str < rhs.idx_rand_str)
		|| (idx_rand_str == rhs.idx_rand_str && idx_hash_func < rhs.idx_hash_func)
		|| (idx_rand_str == rhs.idx_rand_str && idx_hash_func == rhs.idx_hash_func && hash_id < rhs.hash_id)
		|| (idx_rand_str == rhs.idx_rand_str && idx_hash_func == rhs.idx_hash_func && hash_id == rhs.hash_id && idx_str < rhs.idx_str)
		|| (idx_rand_str == rhs.idx_rand_str && idx_hash_func == rhs.idx_hash_func && hash_id == rhs.hash_id && idx_str == rhs.idx_str && idx_rep < rhs.idx_rep) ); }
};

void verification(vector<MyStruct> &results, vector<MyStruct> &original_values){

    std::sort(original_values.begin(),original_values.end());

    bool verified=true;
    for(int i=0; i<original_values.size(); i++){
        if(original_values[i].idx_rand_str!=results[i].idx_rand_str){
            verified=false;
            break;
        }
    }

    if(verified){
        cerr<<"PASSED"<<std::endl<<std::endl;
    }
    else{
        cerr<<"FAILED"<<std::endl<<std::endl;
    }
}

void verification_tuples(vector<tuple<uint32_t,uint32_t,uint32_t,uint32_t,uint32_t>> &results, vector<tuple<uint32_t,uint32_t,uint32_t,uint32_t,uint32_t>> &original_values){

    std::sort(original_values.begin(),original_values.end());

    bool verified=true;

    for(int i=0; i<original_values.size(); i++){
        if(get<0>(original_values[i])!=get<0>(results[i])){
            verified=false;
            break;
        }
    }

    if(verified){
        cerr<<"PASSED"<<std::endl<<std::endl;
    }
    else{
        cerr<<"FAILED"<<std::endl<<std::endl;
    }
}

void sort_dpl(queue &q, size_t max_size){
    std::vector<MyStruct> v(max_size,{0,0,0,0,0});
    int n=max_size;

    for(auto &i:v){
        i.idx_rand_str=rand()%max_size;
        i.idx_hash_func=rand()%max_size;
        i.hash_id=rand()%max_size;
        i.idx_str=rand()%max_size;
        i.idx_rep=rand()%max_size;
    }

    vector<MyStruct> original_values(v.begin(),v.end());

    auto start=std::chrono::system_clock::now();

     {
        buffer<MyStruct> buf{v.data(), v.size()};

        std::sort(make_device_policy(q), oneapi::dpl::begin(buf), oneapi::dpl::end(buf));
     }

    auto end=std::chrono::system_clock::now();
    string device_name=q.get_device().get_info<info::device::name>();

    cout<<"Time DPL SORT with STRUCT on "<<device_name<<": "<<(float)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec - ";

    verification(v, original_values);
}

void sort_tbb(size_t max_size){
    std::vector<MyStruct> v(max_size,{0,0,0,0,0});
    int n=max_size;

     for(auto &i:v){
        i.idx_rand_str=rand()%max_size;
        i.idx_hash_func=rand()%max_size;
        i.hash_id=rand()%max_size;
        i.idx_str=rand()%max_size;
        i.idx_rep=rand()%max_size;
    }

    vector<MyStruct> original_values(v.begin(),v.end());


    auto start=std::chrono::system_clock::now();

     {
        tbb::parallel_sort(v.begin(), v.end());
     }
    auto end=std::chrono::system_clock::now();

    cout<<"Time TBB SORT with STRUCT: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec - ";

    verification(v, original_values);


}


void sort_tbb_tuples(size_t max_size){
    std::vector<tuple<uint32_t,uint32_t,uint32_t,uint32_t,uint32_t>> v(max_size,{0,0,0,0,0});
    int n=max_size;

    for(auto &i:v){
        get<0>(i)=rand()%max_size;
        get<1>(i)=rand()%max_size;
        get<2>(i)=rand()%max_size;
        get<3>(i)=rand()%max_size;
        get<4>(i)=rand()%max_size;

    }

    vector<tuple<uint32_t,uint32_t,uint32_t,uint32_t,uint32_t>> original_values(v.begin(),v.end());

    auto start=std::chrono::system_clock::now();

     {
        tbb::parallel_sort(v.begin(), v.end());
     }
    auto end=std::chrono::system_clock::now();

    cout<<"Time TBB SORT with TUPLES: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec - ";

    verification_tuples(v, original_values);
}

void sort_dpl_tuples(queue &q, size_t max_size){
    std::vector<tuple<uint32_t,uint32_t,uint32_t,uint32_t,uint32_t>> v(max_size,{0,0,0,0,0});
    int n=max_size;

    for(auto &i:v){
        get<0>(i)=rand()%max_size;
        get<1>(i)=rand()%max_size;
        get<2>(i)=rand()%max_size;
        get<3>(i)=rand()%max_size;
        get<4>(i)=rand()%max_size;
    }

    vector<tuple<uint32_t,uint32_t,uint32_t,uint32_t,uint32_t>> original_values(v.begin(),v.end());

    auto start=std::chrono::system_clock::now();

     {
        buffer<tuple<uint32_t,uint32_t,uint32_t,uint32_t,uint32_t>> buf{v.data(), v.size()};

        std::sort(make_device_policy(q), oneapi::dpl::begin(buf), oneapi::dpl::end(buf), [](tuple<uint32_t,uint32_t,uint32_t,uint32_t,uint32_t> t1, tuple<uint32_t,uint32_t,uint32_t,uint32_t,uint32_t> t2){
            return get<0>(t1)<get<0>(t2);
        });
     }

    auto end=std::chrono::system_clock::now();

    string device_name=q.get_device().get_info<info::device::name>();

    cout<<"Time DPL SORT with TUPLES: "<<device_name<<": "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec - ";

    verification_tuples(v, original_values);
}




int main() {

    std::cout << "SORTING TEST\n"<<std::endl;
    queue q1(gpu_selector{});
    queue q2(cpu_selector{});
    size_t max_size=158085312;
    std::cout<<"Number of elements: "<<max_size<<std::endl;

    sort_dpl(q1, max_size);

    sort_dpl(q2, max_size);

    sort_tbb(max_size);

    sort_tbb_tuples(max_size);

    sort_dpl_tuples(q1, max_size);

    sort_dpl_tuples(q2, max_size);

    return 0;
}


