#include <math.h>
#include <immintrin.h>
#include "class_definitions.hpp"

#ifndef VECTORWIDTH
#define VECTORWIDTH 16
#endif

inline void fpdbacksolve(float* __restrict X, float* __restrict BPP, std::size_t n, std::size_t N){

    float x1 = BPP[n];
    float x2 = BPP[n + N];
    float x3 = BPP[n + 2*N];
    float S11 = BPP[n + 4*N];
    float S12 = BPP[n + 5*N];
    float S22 = BPP[n + 6*N];
    float S13 = BPP[n + 7*N];
    float S23 = BPP[n + 8*N];
    float S33 = BPP[n + 9*N];

    float U33 = sqrtf(S33);

    float Ui33 = 1 / U33;
    float U13 = S13 * Ui33;
    float U23 = S23 * Ui33;
    float U22 = sqrtf(S22 - U23*U23);
    float Ui22 = 1 / U22;
    float U12 = (S12 - U13*U23) * Ui22;
    float U11 = sqrtf(S11 - U12*U12 - U13*U13);

    float Ui11 = 1 / U11;
    float Ui12 = - U12 * Ui11 * Ui22;

    float Ui33x3 = Ui33 * x3;

    X[n      ] = Ui11*x1 + Ui12*x2 - (U13 * Ui11 + U23 * Ui12) * Ui33x3;
    X[n +   N] = Ui22*x2 - U23 * Ui22 * Ui33x3;
    X[n + 2*N] = Ui33x3;

}

void processBPP(float* __restrict X, float* __restrict BPP, std::size_t N){
    for (std::size_t n = 0; n < N; n++){
        fpdbacksolve(X, BPP, n, N);
    }
}


// optionally, vectorize via
// __m512 _mm512_permutex2var_ps(__m512 a, __m512i idx, __m512 b)

// void rank1covariances(InverseWishart* rank1covs, float* X, std::size_t N){
//     // float* rank1covs_ptr = reinterpret_cast<float*>(rank1covs.data());
//     float* rank1covs_ptr = reinterpret_cast<float*>(rank1covs);
//     // std::size_t N_iter = N >> 4;
//     std::size_t n;

//     // __m512 ones = _mm512_set1_ps(1.0f);
//     // __m512i select_mix2_0 = _mm512_set_epi32(
//     //     0, 16, 1, 17, 4, 20, 5, 21,  8, 24,  9, 25, 12, 28, 13, 29
//     // );
//     // __m512i select_mix2_1 = _mm512_set_epi32(
//     //     2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31
//     // );
//     // __m512 vx, vy, vz, vxx, vxy, vxz, vyy, vyz, vzz;

//     const u_int64_t two_f32_ones = 0x3f8000003f800000;
//     __m512i inds = _mm512_set_epi32(
//         // 0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120
//         120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0
//     );

//     for (n = 0; n <= N - VECTORWIDTH; n += VECTORWIDTH){
//         __m512 vx = _mm512_loadu_ps(X + n      );
//         __m512 vy = _mm512_loadu_ps(X + n +   N);
//         __m512 vz = _mm512_loadu_ps(X + n + 2*N);
//         __m512 xx = _mm512_mul_ps(vx, vx);
//         __m512 xy = _mm512_mul_ps(vx, vy);
//         __m512 xz = _mm512_mul_ps(vx, vz);
//         __m512 yy = _mm512_mul_ps(vy, vy);
//         __m512 yz = _mm512_mul_ps(vy, vz);
//         __m512 zz = _mm512_mul_ps(vz, vz);

// // #if defined(__clang__) || !defined(__GNUC__)
//         _mm512_i32scatter_ps(rank1covs_ptr    , inds, xx, 4);
//         _mm512_i32scatter_ps(rank1covs_ptr + 1, inds, xy, 4);
//         _mm512_i32scatter_ps(rank1covs_ptr + 2, inds, xz, 4);
//         _mm512_i32scatter_ps(rank1covs_ptr + 3, inds, yy, 4);
//         _mm512_i32scatter_ps(rank1covs_ptr + 4, inds, yz, 4);
//         _mm512_i32scatter_ps(rank1covs_ptr + 5, inds, zz, 4);
// // #else
// //         _mm512_i32scatter_ps(rank1covs_ptr     , inds, xx, 4);
// //         _mm512_i32scatter_ps(rank1covs_ptr +  4, inds, xy, 4);
// //         _mm512_i32scatter_ps(rank1covs_ptr +  8, inds, xz, 4);
// //         _mm512_i32scatter_ps(rank1covs_ptr + 12, inds, yy, 4);
// //         _mm512_i32scatter_ps(rank1covs_ptr + 16, inds, yz, 4);
// //         _mm512_i32scatter_ps(rank1covs_ptr + 20, inds, zz, 4);
// // #endif

//         u_int64_t* uint_ptr = reinterpret_cast<u_int64_t*>(rank1covs_ptr) + 3;
//         for (std::size_t i = 0; i < 16; i++){
//             uint_ptr[0] = two_f32_ones;
//             uint_ptr += 4;
//         }
        
//         rank1covs_ptr += 8*VECTORWIDTH;

//         // __m512 xx_xy_0 = _mm512_permutex2var_ps(xx, select_mix2_0, xy);
//         // __m512 xx_xy_1 = _mm512_permutex2var_ps(xx, select_mix2_1, xy);

//         // __m512 xz_yy_0 = _mm512_permutex2var_ps(xz, select_mix2_0, yy);
//         // __m512 xz_yy_1 = _mm512_permutex2var_ps(xz, select_mix2_1, yy);

//         // __m512 yz_zz_0 = _mm512_permutex2var_ps(yz, select_mix2_0, zz);
//         // __m512 yz_zz_1 = _mm512_permutex2var_ps(yz, select_mix2_1, zz);

//         // __m512 xx_xy_xz_yy_0 =  _mm512_permutex2var_ps(xx_xy_0, select_mix4_0, xz_yy_0);
//         // __m512 xx_xy_xz_yy_1 =  _mm512_permutex2var_ps(xx_xy_1, select_mix4_1, xz_yy_1);
//         // __m512 xx_xy_xz_yy_2 =  _mm512_permutex2var_ps(xx_xy_0, select_mix4_2, xz_yy_0);
//         // __m512 xx_xy_xz_yy_3 =  _mm512_permutex2var_ps(xx_xy_1, select_mix4_3, xz_yy_1);

//         // __m512 yz_zz_ones_0 = _mm512_permutex2var_ps(yz_zz_0, select_mix4_0, ones);
//         // __m512 yz_zz_ones_1 = _mm512_permutex2var_ps(yz_zz_1, select_mix4_1, ones);
//         // __m512 yz_zz_ones_2 = _mm512_permutex2var_ps(yz_zz_0, select_mix4_2, ones);
//         // __m512 yz_zz_ones_3 = _mm512_permutex2var_ps(yz_zz_1, select_mix4_3, ones);



//     }
//     float x, y, z;
//     for (; n < N; n++){
//         x = X[n      ];
//         y = X[n +   N];
//         z = X[n + 2*N];
//         rank1covs[n] = _mm256_set_ps(
//             1.0f, 1.0f, z*z, y*z, y*y, x*z, x*y, x*x
//         );
//     }
// }

void rank1covariances(InverseWishart* __restrict rank1covs, float* __restrict X, std::size_t N){
    float x, y, z;
    for (std::size_t n = 0; n < N; n++){
        x = X[n      ];
        y = X[n +   N];
        z = X[n + 2*N];
        // rank1covs[n] = _mm256_set_ps(
        //     1.0f, 1.0f, z*z, y*z, y*y, x*z, x*y, x*x
        // );
        _mm256_storeu_ps(
            reinterpret_cast<float*>(rank1covs + n),
            _mm256_set_ps(
                1.0f, 1.0f, z*z, y*z, y*y, x*z, x*y, x*x
            )
        );
    }
}