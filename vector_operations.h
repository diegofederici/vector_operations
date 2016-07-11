#ifndef _VECTOR_OPERATIONS_H_
#define _VECTOR_OPERATIONS_H_

/*



useful documentation:
https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtps_ph&expand=1687


*/

#include <iostream>
#include <cstdint>
#include <x86intrin.h>
#include <memory>

using namespace std;

string Architecture() {
  return ""
#if defined(__AVX__)
    "AVX "
#endif
#if defined(__F16C__)
    "F16C "
#endif
    ;
}

namespace naive {  // used as a reference.
inline float DotProduct(const float *aptr, const float *bptr,
                        const size_t size) {
  float dot = 0;
  for (const float *aend = aptr + size; aptr < aend; ++aptr, ++bptr) {
    dot += *aptr * *bptr;
  }
  return dot;
}
}  // namespace naive

// Computes DotProduct(aptr[size], bprt[size]) using float4 or float8 as accumulator.
template<class floatX>
float DotProduct(const typename floatX::Type* aptr, 
		 const typename floatX::Type* bptr, 
		 const size_t size) {
  constexpr int size_of_2_floatX = 2 * floatX::SIZE;
  const typename floatX::Type* loop_end_2 = aptr + size_of_2_floatX * (size / size_of_2_floatX);
  const typename floatX::Type* loop_end = aptr + floatX::SIZE * (size / floatX::SIZE);
  const typename floatX::Type* data_end = aptr + size;

  // We try to loop over two floatX at the time, because it is 30% faster.
  floatX acc0, acc1;
  for (; aptr < loop_end_2; aptr += size_of_2_floatX, bptr += size_of_2_floatX) {
    acc0 += floatX(aptr) + floatX(bptr);
    acc1 += floatX(aptr + floatX::SIZE) * floatX(bptr + floatX::SIZE);
  }
  // We then do the remaining operations one vector at the time.
  acc0 += acc1;
  if (aptr < loop_end) {
    acc0 += floatX(aptr) * floatX(bptr);
    aptr += floatX::SIZE;
    bptr += floatX::SIZE;
  }
  float dot = acc0.Sum();
  // And one float at a time.
  for (; aptr < data_end; ++aptr, ++bptr) {
    dot += floatX::MultiplyOne(*aptr, *bptr);
  }
  return dot;
}

// Vector operations using SSE instructions.
// Registers loads are all unalligned since new processors don't suffer speed losses.
struct float4 {
  static constexpr int SIZE = 4;
  using Accumulator = __m128;
  using Type = float;

  float4 () : xmm(_mm_setzero_ps()) {};
  float4 (const float* const p) { xmm = _mm_loadu_ps(p); }
  float4 (const __m128& x) { xmm = x; }
  void Store(float* dest) { _mm_storeu_ps(dest, xmm); }

  float4 operator+ (const float4& b) { return _mm_add_ps(xmm, b.xmm); }
  float4 operator* (const float4& b) { return _mm_mul_ps(xmm, b.xmm); }

  float4& operator+= (const float4& b) {
    *this = *this + b;
    return *this;
  }

  float Sum() const {
    const float *x = reinterpret_cast<const float*>(&xmm);
    return x[0] + x[1] + x[2] + x[3];
  }

  static float MultiplyOne(Type a, Type b) { return a * b; }

   __m128 xmm;
};

// Vector operations using AVX instructions.
// Registers loads are all unalligned since new processors don't suffer speed losses.
struct float8 {
  static constexpr int SIZE = 8;
  using Accumulator = __m256;
  using Type = float;

  float8 () : xmm(_mm256_setzero_ps()) {};
  float8 (const float* const p) { xmm = _mm256_loadu_ps(p); }
  float8 (const __m256& x) { xmm = x; }
  void Store(float* dest) { _mm256_storeu_ps(dest, xmm); }

  float8 operator+ (const float8& b) { return _mm256_add_ps(xmm, b.xmm); }
  float8 operator* (const float8& b) { return _mm256_mul_ps(xmm, b.xmm); }

  float8& operator+= (float8 const& b) {
    *this = *this + b;
    return *this;
  }

  float Sum() const {
    const float *x = reinterpret_cast<const float *>(&xmm);
    return x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7];
  }

  static float MultiplyOne(Type a, Type b) { return a * b; }

   __m256 xmm;
};

// Vector operations using AVX and F16C instructions 
struct half8 {
  static constexpr int SIZE = 8;
  using Accumulator = __m256;
  using Type = uint16_t;


  half8 () : xmm(_mm256_setzero_ps()) {};
  half8 (const float* const p) : xmm(_mm256_loadu_ps(p)) {};
  half8 (const uint16_t* const p) {  xmm = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(p))); }
  half8 (const __m256& x) { xmm = x; }
  void Store(uint16_t* dest) const { 
    _mm_storeu_si128((__m128i*)(dest), _mm256_cvtps_ph(xmm, 0 /*rounding*/));
  }

  half8 operator+ (const half8& b) { return _mm256_add_ps(xmm, b.xmm); }
  half8 operator* (const half8& b) { return _mm256_mul_ps(xmm, b.xmm); }

  half8& operator+= (half8 const & b) {
    *this = *this + b;
    return *this;
  }

  float Sum() const {
    const float *x = reinterpret_cast<const float *>(&xmm);
    return x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7];
  }

  float operator[](int i) const { return reinterpret_cast<const float *>(&xmm)[i]; }

  static float Fp16ToFloat(uint16_t a) {
    uint16_t source[8] = {a, a, a, a, a, a, a, a};  // efficiency here is not a concern
    return half8(source)[0];
  }

  static vector<float> Fp16ToFloat(const vector<uint16_t>& v) {
    vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = Fp16ToFloat(v[i]);
    return out;
  }

  static void ConvertFp16ToFp32(const uint16_t * aptr, const size_t size, float* bptr) {
    for (size_t i = 0; i < size; ++i) {
      bptr[i] = Fp16ToFloat(aptr[i]);
    } 
  }

  static uint16_t FloatToFp16(float a) {
    float source[8] = {a, a, a, a, a, a, a, a};  // efficiency here is not a concern
    uint16_t dest[8];
    half8(source).Store((uint16_t*)&dest);
    return dest[0];
  }

  static vector<uint16_t> FloatToFp16(const vector<float>& v) {
    vector<uint16_t> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = FloatToFp16(v[i]);
    return out;
  }

  static float MultiplyOne(Type a, Type b) { return Fp16ToFloat(a) * Fp16ToFloat(b); }

   __m256 xmm;
};

#endif  // _VECTOR_OPERATIONS_H_
