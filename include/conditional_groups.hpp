#include "xsimd/xsimd.hpp"
#include "Ziggurat.hpp"
#include "class_definitions.hpp"
#include "vectorized_pcg_rng.hpp"
#include <iostream>
#include <cmath>

#ifndef VECTORWIDTHDEFINED
constexpr std::size_t VECTORWIDTH = xsimd::simd_type<float>::size;
#define VECTORWIDTHDEFINED
#endif

typedef xsimd::batch<u_int32_t, VECTORWIDTH> GVECTOR;

// #define LOG2VECTOR 4
// #define VECTOR __m512
// #define GVECTOR __m128i
// #define VBROADCAST(x) _mm512_set1_ps(x)
// #define VLOAD(ptr) _mm512_loadu_ps(ptr)
// #define VSTORE(ptr, vec) _mm512_storeu_ps(ptr, vec)
// #define VADD(a,b) _mm512_add_ps(a,b)
// #define VMUL(a,b) _mm512_mul_ps(a,b)
// #define VFMADD(a,b,c) _mm512_fmadd_ps(a,b,c)
// #define VEXP(x) Sleef_expf16_u10avx512f(x)
// #define VLOG1P(x) Sleef_log1pf16_u10avx512f(x)
// #define VLOG(x) Sleef_logf16_u35avx512f(x)
// // #define VLOG(x) Sleef_logf16_u10avx512f(x)
// #define MASK __mmask16
// #define VMASKLOAD(mask, ptr) _mm512_maskz_loadu_ps(mask, ptr)
// #define VMASKSTORE(ptr, mask, vec) _mm512_mask_storeu_ps(ptr, mask, vec)
// #define CREATE_GROUP_VECTOR(g) _mm_set1_epi8(g)
// #define VIFELSE(mask, a, b) _mm_mask_blend_epi8(mask, a, b)
// #define VLESS(a, b) _mm512_cmple_ps_mask(a, b)
// // #define GROUPSTORE(ptr, vec) _mm_storeu_epi32(reinterpret_cast<int*>(ptr), vec)
// #define GROUPSTORE(ptr, vec) _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), vec)


inline std::size_t sub2ind(std::size_t nrow, std::size_t i, std::size_t j){
    return nrow * j + i;
}

// template<std::size_t NUMGROUPS, size_t MAXN, size_t MAXMAX>
template<std::size_t NUMGROUPS, size_t MAXN>
void update_groups(Groups<NUMGROUPS, MAXN>& groups, float* __restrict p, float* __restrict group_p, float* __restrict revcholwisharts, float* __restrict x, std::size_t iter){

    std::size_t N = groups.size();
    std::size_t stride = NUMGROUPS * iter;
    std::size_t iter_rem = N & (VECTORWIDTH - 1);
    std::size_t remainder_lb = N - iter_rem ; // ((N / VECTORWIDTH) * VECTORWIDTH);
    // std::size_t mask_off = VECTORWIDTH - iter_rem;
    // MASK mask = MASK( ((uint16_t)((1 << iter_rem) - 1)) ); // << mask_off );

    // float exponent[NUMGROUPS];
    // float base[NUMGROUPS];

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

        auto vLi11 = xsimd::set_simd(Li11);
        auto vLi21 = xsimd::set_simd(Li21);
        auto vLi31 = xsimd::set_simd(Li31);
        auto vLi22 = xsimd::set_simd(Li22);
        auto vLi32 = xsimd::set_simd(Li32);
        auto vLi33 = xsimd::set_simd(Li33);

        auto vone = xsimd::set_simd(1.0f);
        for ( std::size_t i = 0; i < remainder_lb; i += VECTORWIDTH){
            auto x1 = xsimd::load_unaligned(x + sub2ind(N, i, 0));
            auto x2 = xsimd::load_unaligned(x + sub2ind(N, i, 1));
            auto x3 = xsimd::load_unaligned(x + sub2ind(N, i, 2));

            auto vlx1 =   vLi11 * x1;
            auto vlx2 = xsimd::fma(vLi22, x2, vLi21 * x1);
            auto vlx3 = xsimd::fma(vLi33, x3, xsimd::fma(vLi32, x2, vLi31 * x1));

            auto xtLitLixp1 = xsimd::fma(vlx3, vlx3, xsimd::fma(vlx2, vlx2, xsimd::fma(vlx1, vlx1, vone)));

            xtLitLixp1.store_unaligned(p + sub2ind(N, i, g));
        }

        for ( std::size_t i = remainder_lb; i < N; i++ ){
            float x1 = x[sub2ind(N, i, 0)];
            float x2 = x[sub2ind(N, i, 1)];
            float x3 = x[sub2ind(N, i, 2)];

            float lx1 =   Li11 * x1;
            float lx2 = fma(Li22, x2, Li21 * x1);
            float lx3 = fma(Li33, x3, fma(Li32, x2, Li31 * x1));

            float xtLitLixp1 = fma(lx3, lx3, fma(lx2, lx2, fma(lx1, lx1, 1.0f)));

            p[sub2ind(N, i, g)] = xtLitLixp1;
        }

        for ( std::size_t i = 0; i < N; i += VECTORWIDTH){
            log(xsimd::load_unaligned(p + sub2ind(N, i, g))).store_unaligned(p + sub2ind(N, i, g));
        }

        auto vexponent = xsimd::set_simd(exponent);
        auto vbase = xsimd::set_simd(base);
        for ( std::size_t i = 0; i < N; i += VECTORWIDTH){

            auto loglike = exp(xsimd::fma(xsimd::load_unaligned(p + sub2ind(N, i, g)), vexponent, vbase));
            loglike.store_unaligned(p + sub2ind(N, i, g));

        }
    }
    // for (std::size_t g = 0; g < NUMGROUPS; g += 2){
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
    //
    //     auto vLi11 = xsimd::set_simd(Li11);
    //     auto vLi21 = xsimd::set_simd(Li21);
    //     auto vLi31 = xsimd::set_simd(Li31);
    //     auto vLi22 = xsimd::set_simd(Li22);
    //     auto vLi32 = xsimd::set_simd(Li32);
    //     auto vLi33 = xsimd::set_simd(Li33);
    //
    //     float Li11_2 = revcholwisharts[g+1           ];
    //     float Li21_2 = revcholwisharts[g+1 +   stride];
    //     float Li31_2 = revcholwisharts[g+1 + 2*stride];
    //     float Li22_2 = revcholwisharts[g+1 + 3*stride];
    //     float Li32_2 = revcholwisharts[g+1 + 4*stride];
    //     float Li33_2 = revcholwisharts[g+1 + 5*stride];
    //     float nu_2   = revcholwisharts[g+1 + 6*stride];
    //     float exponent_2 = -1.5f-0.5f*nu;
    //     float base_2 = logf(group_p[g+1]) + logf(Li11_2) + logf(Li22_2) + logf(Li33_2) +
    //                     lgammaf(-exponent_2) - lgammaf(0.5f*nu_2) - 1.5f*logf(nu_2);
    //
    //     auto vLi11_2 = xsimd::set_simd(Li11_2);
    //     auto vLi21_2 = xsimd::set_simd(Li21_2);
    //     auto vLi31_2 = xsimd::set_simd(Li31_2);
    //     auto vLi22_2 = xsimd::set_simd(Li22_2);
    //     auto vLi32_2 = xsimd::set_simd(Li32_2);
    //     auto vLi33_2 = xsimd::set_simd(Li33_2);
    //
    //     auto vone = xsimd::set_simd(1.0f);
    //     for ( std::size_t i = 0; i < remainder_lb; i += VECTORWIDTH){
    //         auto x1 = xsimd::load_unaligned(x + sub2ind(N, i, 0));
    //         auto x2 = xsimd::load_unaligned(x + sub2ind(N, i, 1));
    //         auto x3 = xsimd::load_unaligned(x + sub2ind(N, i, 2));
    //
    //         auto vlx1 =   vLi11 * x1;
    //         auto vlx2 = xsimd::fma(vLi22, x2, vLi21 * x1);
    //         auto vlx3 = xsimd::fma(vLi33, x3, xsimd::fma(vLi32, x2, vLi31 * x1));
    //
    //         auto xtLitLixp1 = xsimd::fma(vlx3, vlx3, xsimd::fma(vlx2, vlx2, xsimd::fma(vlx1, vlx1, vone)));
    //
    //         xtLitLixp1.store_unaligned(p + sub2ind(N, i, g));
    //
    //         auto vlx1_2 =   vLi11_2 * x1;
    //         auto vlx2_2 = xsimd::fma(vLi22_2, x2, vLi21_2 * x1);
    //         auto vlx3_2 = xsimd::fma(vLi33_2, x3, xsimd::fma(vLi32_2, x2, vLi31_2 * x1));
    //
    //         auto xtLitLixp1_2 = xsimd::fma(vlx3_2, vlx3_2, xsimd::fma(vlx2_2, vlx2_2, xsimd::fma(vlx1_2, vlx1_2, vone)));
    //
    //         xtLitLixp1_2.store_unaligned(p + sub2ind(N, i, g+1));
    //     }
    //
    //     for ( std::size_t i = remainder_lb; i < N; i++ ){
    //         float x1 = x[sub2ind(N, i, 0)];
    //         float x2 = x[sub2ind(N, i, 1)];
    //         float x3 = x[sub2ind(N, i, 2)];
    //
    //         float lx1 =   Li11 * x1;
    //         float lx2 = fma(Li22, x2, Li21 * x1);
    //         float lx3 = fma(Li33, x3, fma(Li32, x2, Li31 * x1));
    //
    //         float xtLitLixp1 = fma(lx3, lx3, fma(lx2, lx2, fma(lx1, lx1, 1.0f)));
    //
    //         p[sub2ind(N, i, g)] = xtLitLixp1;
    //
    //         float lx1_2 =   Li11_2 * x1;
    //         float lx2_2 = fma(Li22_2, x2, Li21_2 * x1);
    //         float lx3_2 = fma(Li33_2, x3, fma(Li32_2, x2, Li31_2 * x1));
    //
    //         float xtLitLixp1_2 = fma(lx3_2, lx3_2, fma(lx2_2, lx2_2, fma(lx1_2, lx1_2, 1.0f)));
    //
    //         p[sub2ind(N, i, g+1)] = xtLitLixp1_2;
    //     }
    //
    //     for ( std::size_t i = 0; i < N; i += VECTORWIDTH){
    //         log(xsimd::load_unaligned(p + sub2ind(N, i, g))).store_unaligned(p + sub2ind(N, i, g));
    //         log(xsimd::load_unaligned(p + sub2ind(N, i, g+1))).store_unaligned(p + sub2ind(N, i, g+1));
    //     }
    //
    //     auto vexponent = xsimd::set_simd(exponent);
    //     auto vbase = xsimd::set_simd(base);
    //     auto vexponent_2 = xsimd::set_simd(exponent_2);
    //     auto vbase_2 = xsimd::set_simd(base_2);
    //     for ( std::size_t i = 0; i < N; i += VECTORWIDTH){
    //
    //         auto loglike = exp(xsimd::fma(xsimd::load_unaligned(p + sub2ind(N, i, g)), vexponent, vbase));
    //         loglike.store_unaligned(p + sub2ind(N, i, g));
    //
    //         auto loglike_2 = exp(xsimd::fma(xsimd::load_unaligned(p + sub2ind(N, i, g+1)), vexponent_2, vbase_2));
    //         loglike_2.store_unaligned(p + sub2ind(N, i, g+1));
    //
    //     }
    // }

        // for ( std::size_t i = 0; i < N; i += VECTORWIDTH){
        //
        //
        //     exp(xsimd::load_unaligned(p + sub2ind(N, i, g))).store_unaligned(p + sub2ind(N, i, g));
        //     // VSTORE(p + sub2ind(N, i, g),
        //     //     // VEXP(VFMADD(VLOG1P(VFMADD(vlx3, vlx3, VFMADD(vlx2, vlx2, VMUL(vlx1, vlx1)))), vexponent, vbase))
        //     //     VEXP(VLOAD(p + sub2ind(N, i, g)))
        //     // );
        //
        // }

    // }


// #elseif
//
//     std::size_t prob_length = NUMGROUPS * N;
//     float exponent[NUMGROUPS];
//     float base[NUMGROUPS];
//     for (std::size_t g = 0; g < NUMGROUPS; g++){
//         float Li11 = revcholwisharts[g           ];
//         float Li21 = revcholwisharts[g +   stride];
//         float Li31 = revcholwisharts[g + 2*stride];
//         float Li22 = revcholwisharts[g + 3*stride];
//         float Li32 = revcholwisharts[g + 4*stride];
//         float Li33 = revcholwisharts[g + 5*stride];
//         float nu   = revcholwisharts[g + 6*stride];
//         exponent[g] = -1.5f-0.5f*nu;
//         base[g] = logf(group_p[g]) + logf(Li11) + logf(Li22) + logf(Li33) +
//                         lgammaf(-exponent[g]) - lgammaf(0.5f*nu) - 1.5f*logf(nu);
//
//         for ( std::size_t i = 0; i < N; i++){
//             float x1 = x[sub2ind(N, i, 0)];
//             float x2 = x[sub2ind(N, i, 1)];
//             float x3 = x[sub2ind(N, i, 2)];
//
//             float vlx1 = Li11 * x1;
//             float vlx2 = Li21 * x1 + Li22 * x2;
//             float vlx3 = Li31 * x1 + Li32 * x2 + Li33 * x3;
//
//             p[sub2ind(N, i, g)] = vlx3 * vlx3 + vlx2 * vlx2 + vlx1 * vlx1 + 1.0f;
//             // p[sub2ind(N, i, g)] = logf(vlx3 * vlx3 + vlx2 * vlx2 + vlx1 * vlx1 + 1.0f);
//
//         }
//     }
//     for (std::size_t i = 0; i < prob_length; i++){
//         p[i] = logf(p[i]);
//     }
//     for (std::size_t g = 0; g < NUMGROUPS; g++){
//         for ( std::size_t i = 0; i < N; i++){
//             p[sub2ind(N, i, g)] = p[sub2ind(N, i, g)] * exponent[g] + base[g];
//         }
//     }
//     for (std::size_t i = 0; i < prob_length; i++){
//         p[i] = expf(p[i]);
//     }
// #endif

    u_int32_t* group_ptr = reinterpret_cast<u_int32_t*>(&(groups.groups));

    // GVECTOR vgs[NUMGROUPS];
    u_int32_t vgs_sized[NUMGROUPS*VECTORWIDTH];
    u_int32_t* vgs = reinterpret_cast<u_int32_t*>(&vgs_sized);
    for (u_int32_t g = 0; g < (u_int32_t) NUMGROUPS; g++){
        xsimd::set_simd((u_int32_t) (NUMGROUPS - 1 - g)).store_unaligned(vgs + g*VECTORWIDTH);
        // vgs[g] = ((GVECTOR) (NUMGROUPS - 1 - g));
    }
    std::size_t i = 0;
    float cumulative_p[(NUMGROUPS-1)*VECTORWIDTH];
    float* cp_ptr = reinterpret_cast<float*>(&cumulative_p);
    for (; i < N - VECTORWIDTH; i += VECTORWIDTH){

        vfloat_2 unifs = srandunif_2vectors();

        // float vps[(NUMGROUPS-1)*VECTORWIDTH];
        // vfloat vps[NUMGROUPS-1];
        vfloat last_vp = xsimd::load_unaligned(p + i);
        for (std::size_t g = 0; g < NUMGROUPS-1; g++){
            last_vp.store_unaligned(cp_ptr + g*VECTORWIDTH);
            // vps[g] = last_vp;
            last_vp = last_vp + xsimd::load_unaligned(p + sub2ind(N, i, g+1));
        }
        vfloat vu = unifs.x * last_vp;
        // GVECTOR vg = vgs[0];
        vint32 vg = xsimd::load_unaligned(vgs);
        for (std::size_t g = 1; g < NUMGROUPS; g++){
            auto increment_group = vu > xsimd::load_unaligned(cp_ptr + VECTORWIDTH*(NUMGROUPS-1-g));
            // auto increment_group = vu > vps[NUMGROUPS-1-g];
            vg = xsimd::select(
                *reinterpret_cast<xsimd::batch_bool<u_int32_t,VECTORWIDTH>*>(&increment_group),
                vg, xsimd::load_unaligned(vgs + g*VECTORWIDTH)
            );
            // vg = xsimd::select( vu < vps[NUMGROUPS-1-g], vg, vgs[g] );
        }
        vg.store_unaligned(group_ptr + i);

        i += VECTORWIDTH;

        // VECTOR vps[NUMGROUPS-1];
        last_vp = xsimd::load_unaligned(p + i);
        for (std::size_t g = 0; g < NUMGROUPS-1; g++){
            last_vp.store_unaligned(cp_ptr + g*VECTORWIDTH);
            // vps[g] = last_vp;
            last_vp = last_vp + xsimd::load_unaligned(p + sub2ind(N, i, g+1));
        }
        vu = unifs.y * last_vp;
        // vg = vgs[0];
        vg = xsimd::load_unaligned(vgs);
        for (std::size_t g = 1; g < NUMGROUPS; g++){
            auto increment_group = vu > xsimd::load_unaligned(cp_ptr + VECTORWIDTH*(NUMGROUPS-1-g));
            // auto increment_group = vu > vps[NUMGROUPS-1-g];
            vg = xsimd::select(
                *reinterpret_cast<xsimd::batch_bool<u_int32_t,VECTORWIDTH>*>(&increment_group),
                vg, xsimd::load_unaligned(vgs + g*VECTORWIDTH)
            );
            // vg = xsimd::select( vu < vps[NUMGROUPS-1-g], vg, vgs[g] );
        }
        vg.store_unaligned(group_ptr + i);

    }

    // size_t NpW = N + VECTORWIDTH - 1;
    // for (std::size_t i = NpW - (NpW & (VECTORWIDTH - 1)); i < N; i += VECTORWIDTH){
    if (i < N){
        // std::size_t i = N - VECTORWIDTH;
        // vfloat vps[NUMGROUPS-1];
        vfloat last_vp = xsimd::load_unaligned(p + i);
        for (std::size_t g = 0; g < NUMGROUPS-1; g++){
            last_vp.store_unaligned(cp_ptr + g*VECTORWIDTH);
            // vps[g] = last_vp;
            last_vp = last_vp + xsimd::load_unaligned(p + sub2ind(N, i, g+1));
        }
        vfloat vu = srandunif_vector() * last_vp;
        // GVECTOR vg = vgs[0];
        vint32 vg = xsimd::load_unaligned(vgs);
        for (std::size_t g = 1; g < NUMGROUPS; g++){
            auto increment_group = vu > xsimd::load_unaligned(cp_ptr + VECTORWIDTH*(NUMGROUPS-1-g));
            // auto increment_group = vu > vps[NUMGROUPS-1-g];
            vg = xsimd::select(
                *reinterpret_cast<xsimd::batch_bool<u_int32_t,VECTORWIDTH>*>(&increment_group),
                vg, xsimd::load_unaligned(vgs + g*VECTORWIDTH)
            );
            // vg = xsimd::select( vu < vps[NUMGROUPS-1-g], vg, vgs[g] );
        }
        vg.store_unaligned(group_ptr + i);
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

    u_int32_t* group_ptr = reinterpret_cast<u_int32_t*>(&(groups.groups));

    // GVECTOR vgs[NUMGROUPS];
    // u_int32_t* vgs_ptr = reinterpret_cast<u_int32_t*>(&vgs);
    u_int32_t vgs_sized[NUMGROUPS*VECTORWIDTH];
    u_int32_t* vgs = reinterpret_cast<u_int32_t*>(&vgs_sized);
    for (u_int32_t g = 0; g < (u_int32_t) NUMGROUPS; g++){
        xsimd::set_simd((u_int32_t) (NUMGROUPS - 1 - g)).store_unaligned(vgs + g*VECTORWIDTH);
        // vgs[g] = ((GVECTOR) (NUMGROUPS - 1 - g));
    }

    // float ps[NUMGROUPS-1];
    float last_p = base_p[0];
    // vfloat vps[NUMGROUPS-1];
    float cumulative_p[(NUMGROUPS-1)*VECTORWIDTH];
    float* cp_ptr = reinterpret_cast<float*>(&cumulative_p);
    for (std::size_t g = 0; g < NUMGROUPS-1; g++){
        // ps[g] = last_p;
        xsimd::set_simd(last_p).store_unaligned(cp_ptr + g * VECTORWIDTH);
        // vps[g] = xsimd::set_simd(last_p);
        last_p += base_p[g+1];
    }
    vfloat last_vp = xsimd::set_simd(last_p);
    std::size_t i = 0;
    for (; i < N - VECTORWIDTH; i += VECTORWIDTH){

        vfloat_2 unifs = srandunif_2vectors();

        vfloat vu = unifs.x * last_vp;
        // GVECTOR vg = vgs[0];
        vint32 vg = xsimd::load_unaligned(vgs);
        for (std::size_t g = 1; g < NUMGROUPS; g++){
            auto increment_group = vu > xsimd::load_unaligned(cp_ptr + VECTORWIDTH*(NUMGROUPS-1-g));
            // auto increment_group = vu > vps[NUMGROUPS-1-g];
            vg = xsimd::select(
                *reinterpret_cast<xsimd::batch_bool<u_int32_t,VECTORWIDTH>*>(&increment_group),
                vg, xsimd::load_unaligned(vgs + g*VECTORWIDTH)
            );
            // vg = xsimd::select( vu < vps[NUMGROUPS-1-g], vg, vgs[g] );
        }
        vg.store_unaligned(group_ptr + i);

        i += VECTORWIDTH;

        vu = unifs.y * last_vp;
        // vg = vgs[0];
        vg = xsimd::load_unaligned(vgs);
        for (std::size_t g = 1; g < NUMGROUPS; g++){
            auto increment_group = vu > xsimd::load_unaligned(cp_ptr + VECTORWIDTH*(NUMGROUPS-1-g));
            // auto increment_group = vu > vps[NUMGROUPS-1-g];
            vg = xsimd::select(
                *reinterpret_cast<xsimd::batch_bool<u_int32_t,VECTORWIDTH>*>(&increment_group),
                vg, xsimd::load_unaligned(vgs + g*VECTORWIDTH)
            );
            // vg = xsimd::select( vu < vps[NUMGROUPS-1-g], vg, vgs[g] );
        }
        vg.store_unaligned(group_ptr + i);

    }

    if (i < N){
        // std::size_t i = N - VECTORWIDTH;
        vfloat vu = srandunif_vector() * last_vp;
        // GVECTOR vg = vgs[0];
        vint32 vg = xsimd::load_unaligned(vgs);
        for (std::size_t g = 1; g < NUMGROUPS; g++){
            auto increment_group = vu > xsimd::load_unaligned(cp_ptr + VECTORWIDTH*(NUMGROUPS-1-g));
            // auto increment_group = vu > vps[NUMGROUPS-1-g];
            vg = xsimd::select(
                *reinterpret_cast<xsimd::batch_bool<u_int32_t,VECTORWIDTH>*>(&increment_group),
                vg, xsimd::load_unaligned(vgs + g*VECTORWIDTH)
            );
            // vg = xsimd::select( vu < vps[NUMGROUPS-1-g], vg, vgs[g] );
        }
        // std::cout << "vg[0]: " << ((int) vg[0]) << std::endl;
        vg.store_unaligned(group_ptr + i);
    }
    // size_t NpW = N + VECTORWIDTH - 1;
    // for (std::size_t i = NpW - (NpW & (VECTORWIDTH - 1)); i < N; i += VECTORWIDTH){
    //     // std::cout << "i: " << i << std::endl;
    //     vfloat vu = srandunif_vector() * last_vp;
    //     // GVECTOR vg = vgs[0];
    //     vint32 vg = xsimd::load_unaligned(vgs);
    //     for (std::size_t g = 1; g < NUMGROUPS; g++){
    //         auto increment_group = vu > vps[NUMGROUPS-1-g];
    //         vg = xsimd::select(
    //             *reinterpret_cast<xsimd::batch_bool<u_int32_t,VECTORWIDTH>*>(&increment_group),
    //             vg, xsimd::load_unaligned(vgs + g*VECTORWIDTH)
    //         );
    //         // vg = xsimd::select( vu < vps[NUMGROUPS-1-g], vg, vgs[g] );
    //     }
    //     // std::cout << "vg[0]: " << ((int) vg[0]) << std::endl;
    //     vg.store_unaligned(group_ptr + i);
    // }

    // std::cout << "First groups: " << (int) group_ptr[0];
    // for (int j = 1; j < VECTORWIDTH; j++){
    //     std::cout << ", " << (int) group_ptr[j];
    // }
    // std::cout << std::endl;
}
