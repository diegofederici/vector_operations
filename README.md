# Vector Operations

Implementing dot products using SSE, AVX, and F16C (FP16) intrinsics in C++.

## About

SSE and AVX allow to do 4/8 32 bit float vector operations.
GCC and CLANG implement vector extensions 
[link](https://gcc.gnu.org/onlinedocs/gcc/Vector-Extensions.html), 
which simplify implementations a lot *but* AFAIK require memory alignement.

This is done, I believe, for historical reasons, which do not seem to apply anymore on 
new-ish Intel processors. 

For example see:
 [here](http://stackoverflow.com/questions/20259694/is-the-sse-unaligned-load-intrinsic-any-slower-than-the-aligned-load-intrinsic-o), 
or search [google](https://www.google.com/#q=aligned%20vs%20unaligned%20intel).

Additionally we now have F16C instructions that allow to increase memory bandwidth and cache usage.
See [here](https://software.intel.com/en-us/articles/performance-benefits-of-half-precision-floats)
for some information on half precision floats. 

## Objective

I wanted to compare SSE, AVX and AVX with half-precision floats (FP16).
I wrote some code using instruction intrinsics and compared the setups in dot-product a large-ish 
vectors.

## Results

After compiling the code with 
`clang++ -mavx2 -mfma -mf16c -O3 -DNDEBUG -std=c++11 -Wall main.cc && ./a.out`

on a `Intel(R) Core(TM) i7-4770 CPU @ 3.40GHz` I got:

- Testing vector with maximum size of 25000
  - TestDotProd naive      : 184.664808ms
  - TestDotProd SSE        : 31.090338ms
  - TestDotProd AVX        : 19.883545ms
  - TestDotProd AVX F16C   : 27.071737ms
- Testing vector with maximum size of 50000
  - TestDotProd naive      : 724.800277ms
  - TestDotProd SSE        : 122.662481ms
  - TestDotProd AVX        : 81.48282ms
  - TestDotProd AVX F16C   : 107.057727ms
- Testing vector with maximum size of 75000
  - TestDotProd naive      : 1669.5751ms
  - TestDotProd SSE        : 303.518798ms
  - TestDotProd AVX        : 189.698784ms
  - TestDotProd AVX F16C   : 243.225732ms
- Testing vector with maximum size of 100000
  - TestDotProd naive      : 2901.73819ms
  - TestDotProd SSE        : 494.993702ms
  - TestDotProd AVX        : 340.185629ms
  - TestDotProd AVX F16C   : 425.169806ms
- Testing vector with maximum size of 125000
  - TestDotProd naive      : 4593.64872ms
  - TestDotProd SSE        : 772.930664ms
  - TestDotProd AVX        : 544.155754ms
  - TestDotProd AVX F16C   : 664.107381ms
- Testing vector with maximum size of 150000
  - TestDotProd naive      : 6513.90722ms
  - TestDotProd SSE        : 1114.06982ms
  - TestDotProd AVX        : 784.996856ms
  - TestDotProd AVX F16C   : 956.420871ms
- Testing vector with maximum size of 175000
  - TestDotProd naive      : 8887.28345ms
  - TestDotProd SSE        : 1522.54309ms
  - TestDotProd AVX        : 1074.99112ms
  - TestDotProd AVX F16C   : 1302.38023ms

-- diego
