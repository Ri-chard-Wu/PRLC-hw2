#include <iostream>

#ifdef __SSE2__
  #include <emmintrin.h>
#else
  #warning SSE2 support is not available. Code will not compile
#endif

int main(int argc, char **argv)
{
    double input1 = 4.0, input2 = 3.0;

    __m128d a = _mm_set_pd(input1, input2);
    __m128d b = _mm_set_pd(6.0, 5.0);

    __m128d add = _mm_add_pd(a, b);
    __m128d mul = _mm_mul_pd(a, b);
    __m128d div = _mm_div_pd(a, b);
    __m128d sqrt = _mm_sqrt_pd(a);


    double d[2];

    _mm_storeu_pd(d, add);
    std::cout << "add: " << d[0] << "," << d[1] << std::endl;

    _mm_storeu_pd(d, mul);
    std::cout << "mul: " << d[0] << "," << d[1] << std::endl;

    _mm_storeu_pd(d, div);
    std::cout << "div: " << d[0] << "," << d[1] << std::endl;

    _mm_storeu_pd(d, sqrt);
    std::cout << "sqrt: " << d[0] << "," << d[1] << std::endl;

    return 0;
}