#ifndef PCGHEADERS
#define PCGHEADERS

#include "xsimd/xsimd.hpp"
// #include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef VECTORWIDTHDEFINED
constexpr std::size_t VECTORWIDTH = xsimd::simd_type<float>::size;
#define VECTORWIDTHDEFINED
#endif

constexpr std::size_t VECTORWIDTH64 = xsimd::simd_type<u_int64_t>::size;

typedef xsimd::batch<u_int64_t, VECTORWIDTH64> vint64;
typedef xsimd::batch<u_int32_t, VECTORWIDTH> vint32;
typedef xsimd::batch<float, VECTORWIDTH> vfloat;

struct vint64_2{
    vint64 x, y;
};
struct vfloat_2{
    vfloat x, y;
};


    // 4294967297, converted to two uint_32s, are two 1s. 1, converted to two uint_32s is a 0 and a 1.
vint64 one_one = xsimd::set_simd((u_int64_t) 4294967297);
vint64 zero_one = xsimd::set_simd((u_int64_t) 1);
auto selector = (*reinterpret_cast<vint32*>(&one_one)) == (*reinterpret_cast<vint32*>(&zero_one));

struct pcg{
    vint64_2 state;
    const vint64_2 lcg_multiplier;
    // const u_int64_t increment;
    // const u_int64_t transformation_multiplier;
    // const vint increment;
    // const vint transformation_multiplier;
};

// pcg initialize_pcg(u_int64_t);

// union int_to_float{
//     __m512i i;
//     __m512i f;
// };

inline vint32 rotate(vint32 x, vint32 r) {
    return (x >> r) | (x << (32 - r));
}

inline vint64 pcg_xsh_rr(pcg *rng) {
    vint64 oldstate0 = rng->state.x;
    vint64 oldstate1 = rng->state.y;

    rng->state.x = (rng->lcg_multiplier.x * oldstate0) + 1; // xsimd::set_simd(rng->increment);
    rng->state.y = (rng->lcg_multiplier.y * oldstate1) + 1; // xsimd::set_simd(rng->increment);

    vint64 xorshifted0 = ((oldstate0 >> 18) ^ oldstate0) >> 27;
    vint64 rot0 = oldstate0 >> 59;
    vint64 xorshifted1 = ((oldstate1 >> 18) ^ oldstate1) << 5;
    vint64 rot1 = oldstate1 >> 27;



    vint32 xorshifted = xsimd::select(selector, (*reinterpret_cast<vint32*>(&xorshifted0)), (*reinterpret_cast<vint32*>(&xorshifted1)));
    vint32 rot = xsimd::select(selector, (*reinterpret_cast<vint32*>(&rot0)), (*reinterpret_cast<vint32*>(&rot1)));
    vint32 rotated = rotate(xorshifted, rot);
    return *reinterpret_cast<vint64*>(&rotated);
}

inline vint64_2 pcg_rxs_m_xs(pcg *rng) {
    vint64 oldstate0 = rng->state.x;
    vint64 oldstate1 = rng->state.y;

    vint64 count0 = (oldstate0 >> 59) + 5;
    vint64 count1 = (oldstate1 >> 59) + 5;

    rng->state.x = (rng->lcg_multiplier.x * oldstate0) + 1; // xsimd::set_simd(rng->increment);
    rng->state.y = (rng->lcg_multiplier.y * oldstate1) + 1; // xsimd::set_simd(rng->increment);

    vint64 xorshifted0 = ((oldstate0 >> count0) ^ oldstate0) * 0xaef17502108ef2d9; // rng -> transformation_multiplier;
    vint64 xorshifted1 = ((oldstate1 >> count1) ^ oldstate1) * 0xaef17502108ef2d9; // rng -> transformation_multiplier;

    return {
        ((xorshifted0 >> 43) ^ xorshifted0),
        ((xorshifted1 >> 43) ^ xorshifted1)
    } ;
}

inline vfloat int64_to_float_1_to_2(vint64 x){
    vint32 masked = ((*reinterpret_cast<vint32*>(&x)) & xsimd::set_simd((u_int32_t) 8388607)) | xsimd::set_simd((u_int32_t) 1065353216);
    return *reinterpret_cast<vfloat*>(&masked);
}

inline vfloat vrandunif(pcg *rng){
    return 2.0f - int64_to_float_1_to_2(pcg_xsh_rr(rng));
}


inline vfloat_2 vrandunif_2(pcg *rng){
    vint64_2 randints = pcg_rxs_m_xs(rng);

    return {
        2.0f - int64_to_float_1_to_2(randints.x),
        2.0f - int64_to_float_1_to_2(randints.y)
    };
}

#endif
