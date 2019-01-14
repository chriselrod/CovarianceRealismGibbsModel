// #include <iostream>
#include "class_definitions.hpp"
#include "initialize_rngs.hpp"
#include "process_inputs.hpp"
#include "conditional_groups.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef MAXNUMGROUPS
#define MAXNUMGROUPS 16
#endif
#ifndef MAXIMUMSAMPLESIZE
#define MAXIMUMSAMPLESIZE 8192
#endif
#define MAXGROUPSTIMESSAMPLESIZE (MAXNUMGROUPS * MAXIMUMSAMPLESIZE)



Groups<6,MAXIMUMSAMPLESIZE> groups_6;
#pragma omp threadprivate(groups_6)
InverseWisharts<6> inverse_wisharts_6;
#pragma omp threadprivate(inverse_wisharts_6)
float individual_probs[MAXGROUPSTIMESSAMPLESIZE];
#pragma omp threadprivate(individual_probs)

InverseWishart rank1covs[MAXIMUMSAMPLESIZE];

template <std::size_t NUMGROUPS, size_t MAXN>
void run_sample(
    float* probs, float* revcholwisharts, float* cholinvwisharts, // what we're sampling
    InverseWisharts<NUMGROUPS>& iw, Groups<NUMGROUPS, MAXN>& groups, float* individual_probs, // work space
    float* X, InverseWishart* rank1covs, float* base_p,// data
    std::size_t warmup, std::size_t iter // sample info
){
    initialize_groups(groups, base_p);
    calc_wisharts(revcholwisharts, cholinvwisharts, iw, rank1covs, groups, iter);
    // p is working probs
    for (std::size_t i = 0; i < warmup; i++){
        update_probabilities(probs, iw);
        update_groups(groups, reinterpret_cast<float*>(individual_probs), probs, revcholwisharts, X, iter);
        calc_wisharts(revcholwisharts, cholinvwisharts, iw, rank1covs, groups, iter);
    }
    update_probabilities(probs, iw);
    for (std::size_t i = 1; i < iter; i++){
        update_groups(groups, individual_probs, probs, revcholwisharts, X, iter);
        probs += NUMGROUPS; revcholwisharts += NUMGROUPS; cholinvwisharts += NUMGROUPS;
        calc_wisharts(revcholwisharts, cholinvwisharts, iw, rank1covs, groups, iter);
        update_probabilities(probs, iw);
    }
}

extern "C" {

void sample_6groups(
    float* probs, float* revcholwisharts, float* cholinvwisharts, float* X,
    float* BPP, float* base_p, std::size_t N, std::size_t warmup, std::size_t iter
){
    groups_6.resize(N);
    processBPP(X, BPP, N);
    rank1covariances(reinterpret_cast<InverseWishart*>(&rank1covs), X, N);
    run_sample(
        probs, revcholwisharts, cholinvwisharts, // what we're sampling
        inverse_wisharts_6, groups_6, reinterpret_cast<float*>(&individual_probs), // work space
        X, reinterpret_cast<InverseWishart*>(&rank1covs), base_p,// data
        warmup, iter // sample info
    );
}

void processBPP_e(float* X, float* BPP, std::size_t N){
    processBPP(X, BPP, N);
}
void rank1covs_e(InverseWishart* rank1covs, float* X, std::size_t N){
    rank1covariances(rank1covs, X, N);
}
void initialize_groups_e6(int8_t* groups_ptr, float* base_p, std::size_t N){
    groups_6.resize(N);
    initialize_groups(groups_6, base_p);
    for (std::size_t n = 0; n < N; n++){
        groups_ptr[n] = groups_6[n];
    }
}
void calc_wisharts_e6(float* revcholwisharts, float* cholinvwisharts, float* iw_ptr, InverseWishart* rank1covs, int8_t* groups_ptr, std::size_t iter, std::size_t N){
    groups_6.resize(N);
    for (std::size_t n = 0; n < N; n++){
        groups_6[n] = groups_ptr[n];
    }
    calc_wisharts(revcholwisharts, cholinvwisharts, inverse_wisharts_6, rank1covs, groups_6, iter);
    float* iw6_ptr = reinterpret_cast<float*>(&(inverse_wisharts_6.inverse_wisharts));
    for (std::size_t g = 0; g < 36; g += 6){
        _mm256_storeu_ps(iw_ptr+ g, _mm256_loadu_ps(iw6_ptr + g));
    }
}
void update_probabilities_e6(float* probs, float* iw_ptr){
    // for (std::size_t g = 0; g < 6; g++){
    //     inverse_wisharts_6[g] = iw_ptr[g];
    // }
    float* iw6_ptr = reinterpret_cast<float*>(&(inverse_wisharts_6.inverse_wisharts));
    for (std::size_t g = 0; g < 36; g += 6){
        _mm256_storeu_ps(iw6_ptr+ g, _mm256_loadu_ps(iw_ptr + g));
    }
    update_probabilities(probs, inverse_wisharts_6);
}
void update_groups_e6(int8_t* groups_ptr, float* probs, float* individual_probs, float* revcholwisharts, float* X, std::size_t iter, std::size_t N){
    groups_6.resize(N);
    update_groups(groups_6, individual_probs, probs, revcholwisharts, X, iter);
    for (std::size_t n = 0; n < N; n++){
        groups_ptr[n] = groups_6[n];
    }
}

__m512 vsrandunif(){
    return srandunif_vector();
}

}

// void parallel_sample_6groups(){

// }


