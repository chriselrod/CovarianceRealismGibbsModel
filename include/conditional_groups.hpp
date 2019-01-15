#include<immintrin.h>
#include<sleef.h>
#include "Ziggurat.hpp"
#include "class_definitions.hpp"
#include "vectorized_pcg_rng.hpp"
// #include <iostream>
#include<cmath>

float srandunif();
__m512 srandunif_vector();
__m512_2 srandunif_2vectors();

// #ifndef NUMGROUPS
// #define NUMGROUPS 6
// #endif


#define VECTORWIDTH 16
// #define LOG2VECTOR 4
#define VECTOR __m512
#define GVECTOR __m128i
#define VBROADCAST(x) _mm512_set1_ps(x)
#define VLOAD(ptr) _mm512_loadu_ps(ptr)
#define VSTORE(ptr, vec) _mm512_storeu_ps(ptr, vec)
#define VADD(a,b) _mm512_add_ps(a,b)
#define VMUL(a,b) _mm512_mul_ps(a,b)
#define VFMADD(a,b,c) _mm512_fmadd_ps(a,b,c)
#define VEXP(x) Sleef_expf16_u10avx512f(x)
#define VLOG1P(x) Sleef_log1pf16_u10avx512f(x)
#define VLOG(x) Sleef_logf16_u35avx512f(x)
// #define VLOG(x) Sleef_logf16_u10avx512f(x)
#define MASK __mmask16
#define VMASKLOAD(mask, ptr) _mm512_maskz_loadu_ps(mask, ptr)
#define VMASKSTORE(ptr, mask, vec) _mm512_mask_storeu_ps(ptr, mask, vec)
#define CREATE_GROUP_VECTOR(g) _mm_set1_epi8(g)
#define VIFELSE(mask, a, b) _mm_mask_blend_epi8(mask, a, b)
#define VLESS(a, b) _mm512_cmple_ps_mask(a, b)
// #define GROUPSTORE(ptr, vec) _mm_storeu_epi32(reinterpret_cast<int*>(ptr), vec)
#define GROUPSTORE(ptr, vec) _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), vec)


inline std::size_t sub2ind(std::size_t nrow, std::size_t i, std::size_t j){
    return nrow * j + i;
}

// template<std::size_t NUMGROUPS, size_t MAXN, size_t MAXMAX>
template<std::size_t NUMGROUPS, size_t MAXN>
void update_groups(Groups<NUMGROUPS, MAXN>& groups, float* __restrict p, float* __restrict group_p, float* __restrict revcholwisharts, float* __restrict x, std::size_t iter){

    // std::size_t N = groups.size();
    // std::size_t stride = NUMGROUPS * iter;
    // std::size_t iter_rem = N & (VECTORWIDTH - 1);
    // std::size_t remainder_lb = N - iter_rem ; // ((N / VECTORWIDTH) * VECTORWIDTH);
    // // std::size_t mask_off = VECTORWIDTH - iter_rem;
    // MASK mask = MASK( ((uint16_t)((1 << iter_rem) - 1)) ); // << mask_off );

    // VECTOR vone = VBROADCAST(1.0f);
    // // _mm512_set1_ps
    // for (std::size_t g = 0; g < NUMGROUPS; g++){
    //     float Li11 = revcholwisharts[g           ];
    //     float Li21 = revcholwisharts[g +   stride];
    //     float Li31 = revcholwisharts[g + 2*stride];
    //     float Li22 = revcholwisharts[g + 3*stride];
    //     float Li32 = revcholwisharts[g + 4*stride];
    //     float Li33 = revcholwisharts[g + 5*stride];
    //     float nu   = revcholwisharts[g + 6*stride];
    //     float exponent = -1.5f-0.5f*nu;
    //     float base = logf(group_p[g]) + logf(Li11) + logf(Li22) + logf(Li33) +
    //                     lgammaf(-exponent) - lgammaf(0.5f*nu) - 1.5f*logf(nu);

    //     VECTOR vLi11 = VBROADCAST(Li11);
    //     VECTOR vLi21 = VBROADCAST(Li21);
    //     VECTOR vLi31 = VBROADCAST(Li31);
    //     VECTOR vLi22 = VBROADCAST(Li22);
    //     VECTOR vLi32 = VBROADCAST(Li32);
    //     VECTOR vLi33 = VBROADCAST(Li33);
    //     VECTOR vexponent = VBROADCAST(exponent);
    //     VECTOR vbase = VBROADCAST(base);

    //     for ( std::size_t i = 0; i < N - VECTORWIDTH; i += VECTORWIDTH){
    //         VECTOR x1 = VLOAD(x + sub2ind(N, i, 0));
    //         VECTOR x2 = VLOAD(x + sub2ind(N, i, 1));
    //         VECTOR x3 = VLOAD(x + sub2ind(N, i, 2));

    //         VECTOR vlx1 =   VMUL(vLi11, x1);
    //         VECTOR vlx2 = VFMADD(vLi21, x1,   VMUL(vLi22, x2));
    //         VECTOR vlx3 = VFMADD(vLi31, x1, VFMADD(vLi32, x2, VMUL(vLi33, x3)));

    //         VSTORE(p + sub2ind(N, i, g),
    //             // VEXP(VFMADD(VLOG1P(VFMADD(vlx3, vlx3, VFMADD(vlx2, vlx2, VMUL(vlx1, vlx1)))), vexponent, vbase))
    //             VEXP(VFMADD(VLOG(VFMADD(vlx3, vlx3, VFMADD(vlx2, vlx2, VFMADD(vlx1, vlx1, vone)))), vexponent, vbase))
    //         );

    //     }

    //     if (remainder_lb <= N){
    //         VECTOR x1 = VMASKLOAD(mask, x + sub2ind(N, remainder_lb, 0));
    //         VECTOR x2 = VMASKLOAD(mask, x + sub2ind(N, remainder_lb, 1));
    //         VECTOR x3 = VMASKLOAD(mask, x + sub2ind(N, remainder_lb, 2));

    //         VECTOR vlx1 =   VMUL(vLi11, x1);
    //         VECTOR vlx2 = VFMADD(vLi21, x1,   VMUL(vLi22, x2));
    //         VECTOR vlx3 = VFMADD(vLi31, x1, VFMADD(vLi32, x2, VMUL(vLi33, x3)));

    //         VMASKSTORE(p + sub2ind(N, remainder_lb, g), mask,
    //             // VEXP(VFMADD(VLOG1P(VFMADD(vlx3, vlx3, VFMADD(vlx2, vlx2, VMUL(vlx1, vlx1)))), vexponent, vbase))
    //             VEXP(VFMADD(VLOG(VFMADD(vlx3, vlx3, VFMADD(vlx2, vlx2, VFMADD(vlx1, vlx1, vone)))), vexponent, vbase))
    //         );

    //     }

        
    // }

    std::size_t N = groups.size();
    std::size_t stride = NUMGROUPS * iter;
    std::size_t iter_rem = N & (VECTORWIDTH - 1);
    std::size_t remainder_lb = N - iter_rem ; // ((N / VECTORWIDTH) * VECTORWIDTH);
    // std::size_t mask_off = VECTORWIDTH - iter_rem;
    MASK mask = MASK( ((uint16_t)((1 << iter_rem) - 1)) ); // << mask_off );

    // _mm512_set1_ps
    for (std::size_t g = 0; g < NUMGROUPS; g++){
        float Li11 = revcholwisharts[g           ];
        float Li21 = revcholwisharts[g +   stride];
        float Li31 = revcholwisharts[g + 2*stride];
        float Li22 = revcholwisharts[g + 3*stride];
        float Li32 = revcholwisharts[g + 4*stride];
        float Li33 = revcholwisharts[g + 5*stride];
        float nu   = revcholwisharts[g + 6*stride];
        float exponent = -1.5f-0.5f*nu;
        float base = logf(group_p[g]) + logf(Li11) + logf(Li22) + logf(Li33) +
                        lgammaf(-exponent) - lgammaf(0.5f*nu) - 1.5f*logf(nu);

        VECTOR vLi11 = VBROADCAST(Li11);
        VECTOR vLi21 = VBROADCAST(Li21);
        VECTOR vLi31 = VBROADCAST(Li31);
        VECTOR vLi22 = VBROADCAST(Li22);
        VECTOR vLi32 = VBROADCAST(Li32);
        VECTOR vLi33 = VBROADCAST(Li33);

        VECTOR vone = VBROADCAST(1.0f);
        for ( std::size_t i = 0; i < N; i += VECTORWIDTH){
            VECTOR x1 = VLOAD(x + sub2ind(N, i, 0));
            VECTOR x2 = VLOAD(x + sub2ind(N, i, 1));
            VECTOR x3 = VLOAD(x + sub2ind(N, i, 2));

            VECTOR vlx1 =   VMUL(vLi11, x1);
            VECTOR vlx2 = VFMADD(vLi21, x1,   VMUL(vLi22, x2));
            VECTOR vlx3 = VFMADD(vLi31, x1, VFMADD(vLi32, x2, VMUL(vLi33, x3)));

            VSTORE(p + sub2ind(N, i, g),
                VFMADD(vlx3, vlx3, VFMADD(vlx2, vlx2, VFMADD(vlx1, vlx1, vone)))
            );

        }

        if (remainder_lb <= N){
            VECTOR x1 = VMASKLOAD(mask, x + sub2ind(N, remainder_lb, 0));
            VECTOR x2 = VMASKLOAD(mask, x + sub2ind(N, remainder_lb, 1));
            VECTOR x3 = VMASKLOAD(mask, x + sub2ind(N, remainder_lb, 2));

            VECTOR vlx1 =   VMUL(vLi11, x1);
            VECTOR vlx2 = VFMADD(vLi21, x1,   VMUL(vLi22, x2));
            VECTOR vlx3 = VFMADD(vLi31, x1, VFMADD(vLi32, x2, VMUL(vLi33, x3)));

            VSTORE(p + sub2ind(N, remainder_lb, g),
                VFMADD(vlx3, vlx3, VFMADD(vlx2, vlx2, VFMADD(vlx1, vlx1, vone)))
            );

        }

        for ( std::size_t i = 0; i < N; i += VECTORWIDTH){

            VSTORE(p + sub2ind(N, i, g), VLOG( VLOAD(p + sub2ind(N, i, g)) ) );

        }

        VECTOR vexponent = VBROADCAST(exponent);
        VECTOR vbase = VBROADCAST(base);
        for ( std::size_t i = 0; i < N; i += VECTORWIDTH){

            VSTORE(p + sub2ind(N, i, g),
                VFMADD(VLOAD(p + sub2ind(N, i, g)), vexponent, vbase)
            );

        }

        for ( std::size_t i = 0; i < N; i += VECTORWIDTH){

            VSTORE(p + sub2ind(N, i, g),
                // VEXP(VFMADD(VLOG1P(VFMADD(vlx3, vlx3, VFMADD(vlx2, vlx2, VMUL(vlx1, vlx1)))), vexponent, vbase))
                VEXP(VLOAD(p + sub2ind(N, i, g)))
            );

        }

    }

    int8_t* group_ptr = reinterpret_cast<int8_t*>(&(groups.groups));
    
    GVECTOR vgs[NUMGROUPS];
    for (int8_t g = 0; g < (int8_t) NUMGROUPS; g++){
        vgs[g] = CREATE_GROUP_VECTOR(NUMGROUPS - 1 - g);
    }
    
    for (std::size_t i = 0; i < N - VECTORWIDTH; i += VECTORWIDTH){

        __m512_2 unifs = srandunif_2vectors();

        VECTOR vps[NUMGROUPS-1];
        VECTOR last_vp = VLOAD(p + i);
        for (std::size_t g = 0; g < NUMGROUPS-1; g++){
            vps[g] = last_vp;
            last_vp = VADD(last_vp, VLOAD(p + sub2ind(N, i, g+1)));
        }
        VECTOR vu = VMUL(unifs.x, last_vp);
        GVECTOR vg = vgs[0];
        for (std::size_t g = 1; g < NUMGROUPS; g++){
            vg = VIFELSE( VLESS(vu, vps[NUMGROUPS-1-g]), vg, vgs[g] );
        }
        GROUPSTORE(group_ptr + i, vg);

        i += VECTORWIDTH;

        // VECTOR vps[NUMGROUPS-1];
        last_vp = VLOAD(p + i);
        for (std::size_t g = 0; g < NUMGROUPS-1; g++){
            vps[g] = last_vp;
            last_vp = VADD(last_vp, VLOAD(p + sub2ind(N, i, g+1)));
        }
        vu = VMUL(unifs.y, last_vp);
        vg = vgs[0];
        for (std::size_t g = 1; g < NUMGROUPS; g++){
            vg = VIFELSE( VLESS(vu, vps[NUMGROUPS-1-g]), vg, vgs[g] );
        }
        GROUPSTORE(group_ptr + i, vg);

    }

    size_t NpW = N + VECTORWIDTH - 1;
    for (std::size_t i = NpW - (NpW & (VECTORWIDTH - 1)); i < N; i += VECTORWIDTH){
        VECTOR vps[NUMGROUPS-1];
        VECTOR last_vp = VLOAD(p + i);
        for (std::size_t g = 0; g < NUMGROUPS-1; g++){
            vps[g] = last_vp;
            last_vp = VADD(last_vp, VLOAD(p + sub2ind(N, i, g+1)));
        }
        VECTOR vu = VMUL(srandunif_vector(), last_vp);
        GVECTOR vg = vgs[0];
        for (std::size_t g = 1; g < NUMGROUPS; g++){
            vg = VIFELSE( VLESS(vu, vps[NUMGROUPS-1-g]), vg, vgs[g] );
        }
        GROUPSTORE(group_ptr + i, vg);
        // i += VECTORWIDTH;
    }

    // std::size_t j = N - VECTORWIDTH;
    // if (i < j){
    //     VECTOR vps[NUMGROUPS-1];
    //     VECTOR last_vp = VLOAD(p + j);
    //     for (std::size_t g = 0; g < NUMGROUPS-1; g++){
    //         vps[g] = last_vp;
    //         last_vp = VADD(last_vp, VLOAD(p + sub2ind(N, j, g+1)));
    //     }
    //     VECTOR vu = VMUL(srandunif_vector(), last_vp);
    //     GVECTOR vg = vgs[0];
    //     for (std::size_t g = 1; g < NUMGROUPS; g++){
    //         vg = VIFELSE( VLESS(vu, vps[NUMGROUPS-1-g]), vg, vgs[g] );
    //     }
    //     GROUPSTORE(group_ptr + j, vg);
    // }

}

template<std::size_t NUMGROUPS, size_t MAXN>
void initialize_groups(Groups<NUMGROUPS, MAXN>& groups, float* base_p){
    std::size_t N = groups.size();
    // std::cout << "N: " << N << std::endl;
    // std::cout << "base_p: " << base_p[0];// << N << std::endl;
    // for (int i = 1; i < 6; i++){
    //     std::cout << ", " << base_p[i];
    // }
    // std::cout << std::endl;
    // std::size_t remainder_lb = ((N / VECTORWIDTH) * VECTORWIDTH);

    int8_t* group_ptr = reinterpret_cast<int8_t*>(&(groups.groups));

    GVECTOR vgs[NUMGROUPS];
    for (int8_t g = 0; g < (int8_t) NUMGROUPS; g++){
        vgs[g] = CREATE_GROUP_VECTOR(NUMGROUPS - 1 - g);
    }
    
    // float ps[NUMGROUPS-1];
    float last_p = base_p[0];
    VECTOR vps[NUMGROUPS-1];
    for (std::size_t g = 0; g < NUMGROUPS-1; g++){
        // ps[g] = last_p;
        vps[g] = VBROADCAST(last_p);
        last_p += base_p[g+1];
    }
    VECTOR last_vp = VBROADCAST(last_p);
    for (std::size_t i = 0; i < N - VECTORWIDTH; i += VECTORWIDTH){

        __m512_2 unifs = srandunif_2vectors();

        VECTOR vu = VMUL(unifs.x, last_vp);
        GVECTOR vg = vgs[0];
        for (std::size_t g = 1; g < NUMGROUPS; g++){
            vg = VIFELSE( VLESS(vu, vps[NUMGROUPS-1-g]), vg, vgs[g] );
        }
        GROUPSTORE(group_ptr + i, vg);

        i += VECTORWIDTH;

        vu = VMUL(unifs.y, last_vp);
        vg = vgs[0];
        for (std::size_t g = 1; g < NUMGROUPS; g++){
            vg = VIFELSE( VLESS(vu, vps[NUMGROUPS-1-g]), vg, vgs[g] );
        }
        GROUPSTORE(group_ptr + i, vg);

    }
    

    size_t NpW = N + VECTORWIDTH - 1;
    for (std::size_t i = NpW - (NpW & (VECTORWIDTH - 1)); i < N; i += VECTORWIDTH){
        // std::cout << "i: " << i << std::endl;
        VECTOR vu = VMUL(srandunif_vector(), last_vp);
        GVECTOR vg = vgs[0];
        for (std::size_t g = 1; g < NUMGROUPS; g++){
            vg = VIFELSE( VLESS(vu, vps[NUMGROUPS-1-g]), vg, vgs[g] );
        }
        // std::cout << "vg[0]: " << ((int) vg[0]) << std::endl;
        GROUPSTORE(group_ptr + i, vg);
    }

    // std::cout << "First groups: " << (int) group_ptr[0];
    // for (int j = 1; j < VECTORWIDTH; j++){
    //     std::cout << ", " << (int) group_ptr[j];
    // }
    // std::cout << std::endl;
}

