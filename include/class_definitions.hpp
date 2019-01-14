#ifndef GIBBSCLASSDEFS
#define GIBBSCLASSDEFS

// #include <iostream>
#include <immintrin.h>
#include <math.h>



template <size_t NUMGROUPS, size_t MAXN>
class Groups{
public:
    int8_t groups[MAXN];
    size_t N;
    
    template <class T>
    inline int8_t& operator[] (T i){
        return groups[i];
    }
    std::size_t size(){
        return N;
    }
    void resize(std::size_t N_new){
        this -> N = N_new;
    }
};

class InverseWishart{
    __m256 data;
public: 
    InverseWishart(){
        data = _mm256_set1_ps(0.0f);
    };
    InverseWishart(__m256 d){
        data = d;
    };
    inline InverseWishart operator+ (const InverseWishart& win){
        return _mm256_add_ps(this -> data, win.data);
    }
    inline InverseWishart& operator+= (const InverseWishart& win){
        this -> data = _mm256_add_ps(this -> data, win.data);
        return *this;
    }
    template <class T>
    inline float operator[] (T i){
        return (this -> data)[i];
        // return data[i];
    }
};

template <std::size_t NUMGROUPS>
class InverseWisharts{
public:
    InverseWishart inverse_wisharts[NUMGROUPS];

    // InverseWisharts(){
    //     for (std::size_t i = 0; i < NUMGROUPS; i++){
    //         this -> inverse_wisharts[i] = _mm256_set_ps(powf(1.5, NUMGROUPS - i), 3.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f);
    //     }
    //     return this;
    // }

    template <class T>
    inline InverseWishart& operator[] (T i){
        return inverse_wisharts[i];
    }

    void reset_priors(){
        for (std::size_t i = 0; i < NUMGROUPS; i++){
            this -> inverse_wisharts[i] = _mm256_set_ps(powf(1.5, NUMGROUPS - i), 3.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f);
        }
    }

    template <size_t MAXN>
    void update(InverseWishart* rank1covs, Groups<NUMGROUPS, MAXN>& groups){
        size_t N = groups.size();
        for (std::size_t n = 0; n < N; n++){
            this -> inverse_wisharts[groups[n]] += rank1covs[n];
        }

    }

    void factor_and_invert(float* __restrict revcholwisharts, float* __restrict cholinvwisharts, std::size_t iter){
        float L11, L21, L31, L22, L32, L33, R11, R21, R31, R22, R32, R33;
        InverseWishart iw;
        std::size_t stride = iter * NUMGROUPS;
        for (std::size_t g = 0; g < NUMGROUPS; g++){
            iw = this -> inverse_wisharts[g];
            L11 = sqrtf(iw[0]);
            R11 = 1 / L11;
            L21 = R11 * iw[1];
            L31 = R11 * iw[2];
            L22 = sqrtf(iw[3] - L21 * L21);
            R22 = 1 / L22;
            L32 = R22 * (iw[4] - L21 * L31);
            L33 = sqrtf(iw[5] - L31 * L31 - L32 * L32);
            R33 = 1 / L33;

            R21 = -R22 * L21 * R11;
            R31 = -R33 * (L31 * R11 + L32 * R21);
            R32 = -R33 * L32 * R22;

            cholinvwisharts[g           ] = L11;
            cholinvwisharts[g +   stride] = L21;
            cholinvwisharts[g + 2*stride] = L31;
            cholinvwisharts[g + 3*stride] = L22;
            cholinvwisharts[g + 4*stride] = L32;
            cholinvwisharts[g + 5*stride] = L33;
            cholinvwisharts[g + 6*stride] = iw[6];
            cholinvwisharts[g + 7*stride] = iw[7];
            
            revcholwisharts[g           ] = R11;
            revcholwisharts[g +   stride] = R21;
            revcholwisharts[g + 2*stride] = R31;
            revcholwisharts[g + 3*stride] = R22;
            revcholwisharts[g + 4*stride] = R32;
            revcholwisharts[g + 5*stride] = R33;
            revcholwisharts[g + 6*stride] = iw[6];
            revcholwisharts[g + 7*stride] = iw[7];
        }
    }
};

template<size_t NUMGROUPS, size_t MAXN>
void calc_wisharts(float* revcholwisharts, float* cholinvwisharts, InverseWisharts<NUMGROUPS>& iw,
                    InverseWishart* rank1covs, Groups<NUMGROUPS, MAXN>& groups, std::size_t iter){

    // std::cout << "Resetting priors." << std::endl;
    iw.reset_priors();
    // std::cout << "Updating InverseWisharts based on rank1covs and groups." << std::endl;
    iw.update(rank1covs, groups);
    // std::cout << "Factoring the InverseWisharts and inverting the factors." << std::endl;
    iw.factor_and_invert(revcholwisharts, cholinvwisharts, iter);
    // std::cout << "Succesfully factored. Leaving calc_wisharts()." << std::endl;
}

#endif