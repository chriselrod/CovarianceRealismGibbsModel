#ifndef PCGHEADERS
#define PCGHEADERS

#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif



struct __m512i_2{
    __m512i x, y;
};
struct __m512_2{
    __m512 x, y;
};

struct pcg{
    __m512i_2 state;
    const __m512i_2 lcg_multiplier;
    const __m512i increment;
    const __m512i transformation_multiplier;
};

pcg initialize_pcg(u_int64_t);

// union int_to_float{
//     __m512i i;
//     __m512i f;
// };

inline __m512i pcg_xsh_rr(pcg *rng) {
    __m512i oldstate0 = rng->state.x;
    __m512i oldstate1 = rng->state.y;

    rng->state.x = _mm512_add_epi64(
      _mm512_mullo_epi64(rng->lcg_multiplier.x, oldstate0), rng->increment);
    rng->state.y = _mm512_add_epi64(
      _mm512_mullo_epi64(rng->lcg_multiplier.y, oldstate1), rng->increment);

    __m512i xorshifted0 = _mm512_srli_epi64(
      _mm512_xor_epi64(_mm512_srli_epi64(oldstate0, 18), oldstate0), 27);
    __m512i rot0 = _mm512_srli_epi64(oldstate0, 59);
    __m512i xorshifted1 = _mm512_srli_epi64(
      _mm512_xor_epi64(_mm512_srli_epi64(oldstate1, 18), oldstate1), 27);
    __m512i rot1 = _mm512_srli_epi64(oldstate1, 59);
    return _mm512_inserti32x8(
      _mm512_castsi256_si512(
          _mm512_cvtepi64_epi32(_mm512_rorv_epi32(xorshifted0, rot0))),
      _mm512_cvtepi64_epi32(_mm512_rorv_epi32(xorshifted1, rot1)), 1);
}

inline __m512i_2 pcg_rxs_m_xs(pcg *rng) {
    __m512i oldstate0 = rng->state.x;
    __m512i oldstate1 = rng->state.y;

    __m512i five = _mm512_set1_epi64(5);
    __m512i count0 = _mm512_add_epi64(_mm512_srli_epi64(oldstate0, 59), five);
    __m512i count1 = _mm512_add_epi64(_mm512_srli_epi64(oldstate1, 59), five);

    rng->state.x = _mm512_add_epi64(
    _mm512_mullo_epi64(rng->lcg_multiplier.x, oldstate0), rng->increment);
    rng->state.y = _mm512_add_epi64(
    _mm512_mullo_epi64(rng->lcg_multiplier.y, oldstate1), rng->increment);

    __m512i xorshifted0 = _mm512_mullo_epi64(
        _mm512_xor_epi64(_mm512_srlv_epi64(oldstate0, count0), oldstate0),
        rng -> transformation_multiplier
    );

    __m512i xorshifted1 = _mm512_mullo_epi64(
        _mm512_xor_epi64(_mm512_srlv_epi64(oldstate1, count1), oldstate1),
        rng -> transformation_multiplier
    );

    return {
        _mm512_xor_epi64(_mm512_srli_epi64(xorshifted0, 43), xorshifted0),
        _mm512_xor_epi64(_mm512_srli_epi64(xorshifted1, 43), xorshifted1)
    } ;
}


inline __m512 vrandunif(pcg *rng){
    __m512i randint = pcg_xsh_rr(rng);
    __m512i masked_int = _mm512_or_si512(
        _mm512_and_si512(randint, _mm512_set1_epi32(8388607)), _mm512_set1_epi32(1065353216));
    return _mm512_sub_ps(_mm512_set1_ps(2.0), _mm512_castsi512_ps(masked_int));
}


inline __m512_2 vrandunif_2(pcg *rng){
    __m512i_2 randints = pcg_rxs_m_xs(rng);
    __m512i andmask =  _mm512_set1_epi32(8388607);
    __m512i ormask = _mm512_set1_epi32(1065353216);
    __m512i masked_int0 = _mm512_or_si512(
        _mm512_and_si512(randints.x, andmask), ormask
    );
    __m512i masked_int1 = _mm512_or_si512(
        _mm512_and_si512(randints.y, andmask), ormask
    );

    __m512 two = _mm512_set1_ps(2.0);

    return {
        _mm512_sub_ps(two, _mm512_castsi512_ps(masked_int0)),
        _mm512_sub_ps(two, _mm512_castsi512_ps(masked_int1))
    };
}

#endif
