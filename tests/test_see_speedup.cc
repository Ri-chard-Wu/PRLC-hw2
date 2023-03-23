#include <chrono>
#include <iostream>

#ifdef __SSE2__
  #include <emmintrin.h>
#else
  #warning SSE2 support is not available. Code will not compile
#endif

using namespace std::chrono;
using namespace std;







int main()
{
    int sz = 100000;
    
    double a0[sz], b0[sz], c0[sz];
    double a1[sz], b1[sz], c1[sz];

    for(int i=0;i<sz;i++){
        a0[i] = 1.*i;
        b0[i] = 1.*sz - i;
        a1[i] = 1.*i*2;
        b1[i] = 1.*(sz - i)*2;        
    }

    auto start = high_resolution_clock::now();

    for(int i=0;i<sz;i++){
        c0[i] = a0[i] + b0[i];
        c1[i] = a1[i] + b1[i];    
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "dt: "<<duration.count()<<" us"<<endl;

    for(int i=1002;i<1002+5;i++){
        cout << c0[i] << ", " << c1[i] <<endl;
    }

                




    start = high_resolution_clock::now();

    __m128d a2, b2, c2;
    double buf_pd[2];
    for(int i=0;i<sz;i++){
        a2 = _mm_set_pd(a0[i], a1[i]);
        b2 = _mm_set_pd(b0[i], b1[i]);
        c2 = _mm_add_pd(a2, b2);
        _mm_storeu_pd(buf_pd, c2);
        c0[i] = buf_pd[1];
        c1[i] = buf_pd[0];
    }

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "dt: "<<duration.count()<<" us"<<endl;


    for(int i=1002;i<1002+5;i++){
        cout << c0[i] << ", " << c1[i] <<endl;
    }


    return 0;
}









// int main()
// {
//     int sz = 100000;
    
//     double a0[sz], b0[sz], c0[sz];
//     double a1[sz], b1[sz], c1[sz];

//     for(int i=0;i<sz;i++){
//         a0[i] = 1.*i;
//         b0[i] = 1.*sz - i;
//         a1[i] = 1.*i*2;
//         b1[i] = 1.*(sz - i)*2;        
//     }

//     auto start = high_resolution_clock::now();

//     for(int i=0;i<sz;i++){
//         c0[i] = a0[i] * b0[i];
//         c1[i] = a1[i] * b1[i];    
//     }

//     auto stop = high_resolution_clock::now();
//     auto duration = duration_cast<microseconds>(stop - start);
//     cout << "dt: "<<duration.count()<<" us"<<endl;

//     for(int i=1002;i<1002+5;i++){
//         cout << c0[i] << ", " << c1[i] <<endl;
//     }

                




//     start = high_resolution_clock::now();

//     __m128d a2, b2, c2;
//     double buf_pd[2];
//     for(int i=0;i<sz;i++){
//         a2 = _mm_set_pd(a0[i], a1[i]);
//         b2 = _mm_set_pd(b0[i], b1[i]);
//         c2 = _mm_mul_pd(a2, b2);
//         _mm_storeu_pd(buf_pd, c2);
//         c0[i] = buf_pd[1];
//         c1[i] = buf_pd[0];
//     }

//     stop = high_resolution_clock::now();
//     duration = duration_cast<microseconds>(stop - start);
//     cout << "dt: "<<duration.count()<<" us"<<endl;


//     for(int i=1002;i<1002+5;i++){
//         cout << c0[i] << ", " << c1[i] <<endl;
//     }


//     return 0;
// }