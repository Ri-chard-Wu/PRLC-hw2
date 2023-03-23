#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <tbb/parallel_for.h>
using namespace std::chrono;
using namespace std;




int main(int argc, char **argv)
{

    auto start = high_resolution_clock::now();


    auto values = std::vector<double>(10000);
    
    tbb::parallel_for( tbb::blocked_range<int>(0, 1000),
                       [&](tbb::blocked_range<int> r)
    {
        for (int i=r.begin(); i<r.end(); ++i)
        {
            values[i] = std::sin(i * 0.001);
        }
    });


    double total = 0;

    for (double value : values)
    {
        total += value;
    }

    std::cout << total << std::endl;

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "parallel dt: "<<duration.count()<<" us"<<endl;






    start = high_resolution_clock::now();
    
    for (int i=0; i<values.size(); ++i)
    {
        values[i] = std::sin(i * 0.001);
    }

    total = 0;

    for (double value : values)
    {
        total += value;
    }

    std::cout << total << std::endl;

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "sequential dt: "<<duration.count()<<" us"<<endl;



    return 0;
}