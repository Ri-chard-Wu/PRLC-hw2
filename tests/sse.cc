#include <iostream>

#ifdef __SSE2__
  #include <emmintrin.h>
#else
  #warning SSE2 support is not available. Code will not compile
#endif

using namespace std;

int main(int argc, char **argv)
{

    // __m128d a = _mm_set_pd(4.0, 3.0);
    // __m128d b = _mm_set_pd(6.0, 5.0);

    // __m128d add = _mm_add_pd(a, b);
    // __m128d mul = _mm_mul_pd(a, b);
    // __m128d div = _mm_div_pd(a, b);
    // __m128d sqrt = _mm_sqrt_pd(a);

    // double d[2];

    // _mm_storeu_pd(d, add);
    // std::cout << "add: " << d[0] << "," << d[1] << std::endl;

    // _mm_storeu_pd(d, mul);
    // std::cout << "mul: " << d[0] << "," << d[1] << std::endl;

    // _mm_storeu_pd(d, div);
    // std::cout << "div: " << d[0] << "," << d[1] << std::endl;

    // _mm_storeu_pd(d, sqrt);
    // std::cout << "sqrt: " << d[0] << "," << d[1] << std::endl;
    




    // __m128d min = _mm_min_pd(a, b);
    // __m128d cmpeq = _mm_cmpeq_pd(a, b);
    // __m128d cmpge = _mm_cmpge_pd(a, b);
    // __m128d cmpgt = _mm_cmpgt_pd(a, b);
    // __m128d cmple = _mm_cmple_pd(a, b);
    // __m128d cmplt = _mm_cmplt_pd(a, b);

    double buf_pd[2];
    __m128d a;
    __m128d b;

    
    // a = _mm_set_pd(4.0, 3.0);
    // b = _mm_set_pd(6.0, 3.0);

    // _mm_storeu_pd(buf_pd, _mm_cmpeq_pd(a, b));
    // if(buf_pd[0]){
    //     std::cout << "0's are equal"<< endl;
    // }
    // else{
    //     std::cout << "0's are not equal"<< endl;
    // }
    
    // if(buf_pd[1]){
    //     std::cout << "1's are equal"<< endl;
    // }
    // else{
    //     std::cout << "1's are not equal"<< endl;
    // }



    
    // a = _mm_set_pd(4.0, 60.0);
    // b = _mm_set_pd(6.0, 3.0);

    // _mm_storeu_pd(buf_pd, _mm_cmpgt_pd(a, b));
    // if(buf_pd[0]){
    //     std::cout << "a[0] > b[0]"<< endl;
    // }
    // else{
    //     std::cout << "a[0] <= b[0]"<< endl;
    // }

    // if(buf_pd[1]){
    //     std::cout << "a[1] > b[1]"<< endl;
    // }
    // else{
    //     std::cout << "a[1] <= b[1]"<< endl;
    // }




    a = _mm_set_pd(4.0, 60.0);
    b = _mm_set_pd(6.0, 300.0);

    _mm_storeu_pd(buf_pd, _mm_min_pd(a, b));
 
    cout << "min[0]" << buf_pd[0] << endl;
    cout << "min[1]" << buf_pd[1] << endl;
 


    return 0;
}