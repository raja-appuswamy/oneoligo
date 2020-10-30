#include "constants.hpp"


    
    #ifndef DEF_NUM_STR
    #define DEF_NUM_STR 7 // r: number of CGK-embedding for each input string
    #endif

    #ifndef DEF_NUM_HASH
        #define DEF_NUM_HASH 16  //z: number of hash functions for each embedded string
    #endif

    #ifndef DEF_NUM_BITS
        #define DEF_NUM_BITS 12// m: number of bits in each hash function
    #endif

    #ifndef DEF_NUM_CHAR
        #define DEF_NUM_CHAR 4 //dictsize: alpha beta size of input strings, could be 4 for DNA dataset (ACGT); 26 for UNIREF dataset (A~Z); 37 for TREC dataset (A~Z,0~9,' ')
    #endif

    #ifndef DEF_ALLOUTPUTRESULT
        #define DEF_ALLOUTPUTRESULT 0
    #endif

    #ifndef DEF_SHIFT
        #define DEF_SHIFT 50
    #endif

    #ifndef DEF_HASH_SZ
        #define DEF_HASH_SZ 1000003 //size of hash table;
    #endif

    #ifndef DEF_K_INPUT
        #define DEF_K_INPUT 150 // edit distance threshold
    #endif

namespace constants{
    const size_t NUM_STR=DEF_NUM_STR;
    const size_t NUM_HASH=DEF_NUM_HASH;
    const size_t NUM_BITS=DEF_NUM_BITS;
    const uint8_t NUM_CHAR=DEF_NUM_CHAR;
    const bool ALLOUTPUTRESULT=DEF_ALLOUTPUTRESULT;
    const size_t SHIFT=DEF_SHIFT;
    const size_t HASH_SZ=DEF_HASH_SZ;
    const size_t K_INPUT=DEF_K_INPUT;

    const size_t NUM_REP = static_cast<size_t>((K_INPUT+SHIFT-1)/SHIFT);

}
