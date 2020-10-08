#include<CL/sycl.hpp>
#include<oneapi/dpl/execution>
#include<oneapi/dpl/algorithm>
#include<oneapi/dpl/iterator>
#include "tbb/parallel_sort.h"
using namespace sycl;
using namespace oneapi::dpl::execution;
using namespace std;

struct MyStruct{
    size_t key;
    size_t val1;
    size_t val2;
    MyStruct(size_t k, size_t v1, size_t v2): key(k), val1(v1), val2(v2){}
};

void verification(vector<MyStruct> &results, vector<MyStruct> &original_values){

    std::sort(original_values.begin(),original_values.end(),[](auto b1, auto b2){
        return b1.key<b2.key;
    });

    bool verified=true;
    for(int i=0; i<original_values.size(); i++){
        if(original_values[i].key!=results[i].key){
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

void verification_tuples(vector<tuple<size_t,size_t,size_t>> &results, vector<tuple<size_t,size_t,size_t>> &original_values){

    std::sort(original_values.begin(),original_values.end(),[](auto b1, auto b2){
        return get<0>(b1)<get<0>(b2);
    });

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
    std::vector<MyStruct> v(max_size,{0,0,0});
    int n=max_size;

    for(auto &i:v){
        i.key=n;
        i.val1=10;
        i.val2=n;
        n--;
    }
    vector<MyStruct> original_values(v.begin(),v.end());

    auto start=std::chrono::system_clock::now();

     {
        buffer<MyStruct> buf{v.data(), v.size()};

        std::sort(make_device_policy(q), oneapi::dpl::begin(buf), oneapi::dpl::end(buf), [](MyStruct t1, MyStruct t2){
            return t1.key<t2.key;
        });
     }

    auto end=std::chrono::system_clock::now();
    string device_name=q.get_device().get_info<info::device::name>();

    cout<<"Time DPL SORT with STRUCT on "<<device_name<<": "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec - ";

    verification(v, original_values);
}

void sort_tbb(size_t max_size){
    std::vector<MyStruct> v(max_size,{0,0,0});
    int n=max_size;

    for(auto &i:v){
        i.key=n;
        i.val1=10;
        i.val2=n;
        n--;
    }

    vector<MyStruct> original_values(v.begin(),v.end());


    auto start=std::chrono::system_clock::now();

     {
        tbb::parallel_sort(v.begin(), v.end(), [](MyStruct t1, MyStruct t2){
            return t1.key<t2.key;
        });
     }
    auto end=std::chrono::system_clock::now();

    cout<<"Time TBB SORT with STRUCT: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec - ";

    verification(v, original_values);


}


void sort_tbb_tuples(size_t max_size){
    std::vector<tuple<size_t,size_t,size_t>> v(max_size,{0,0,0});
    int n=max_size;

    for(auto &i:v){
        get<0>(i)=n;
        get<1>(i)=10;
        get<2>(i)=n;
        n--;
    }

    vector<tuple<size_t,size_t,size_t>> original_values(v.begin(),v.end());

    auto start=std::chrono::system_clock::now();

     {
        tbb::parallel_sort(v.begin(), v.end(), [](auto t1, auto t2){
            return get<0>(t1)<get<0>(t2);
        });
     }
    auto end=std::chrono::system_clock::now();

    cout<<"Time TBB SORT with TUPLES: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000<<"sec - ";

    verification_tuples(v, original_values);
}

void sort_dpl_tuples(queue &q, size_t max_size){
    std::vector<tuple<size_t,size_t,size_t>> v(max_size,{0,0,0});
    int n=max_size;

    for(auto &i:v){
        get<0>(i)=n;
        get<1>(i)=10;
        get<2>(i)=n;
        n--;
    }

    vector<tuple<size_t,size_t,size_t>> original_values(v.begin(),v.end());

    auto start=std::chrono::system_clock::now();

     {
        buffer<tuple<size_t,size_t,size_t>> buf{v.data(), v.size()};

        std::sort(make_device_policy(q), oneapi::dpl::begin(buf), oneapi::dpl::end(buf), [](tuple<size_t,size_t,size_t> t1, tuple<size_t,size_t,size_t> t2){
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
    size_t max_size=100000000;

    sort_dpl(q1, max_size);

    sort_dpl(q2, max_size);

    sort_tbb(max_size);

    sort_tbb_tuples(max_size);

    sort_dpl_tuples(q1, max_size);

    sort_dpl_tuples(q2, max_size);

    return 0;
}

