



#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <iostream>
using namespace std;







int main(int argc, char **argv) {

    int a, c;
    int b = 234;
    a = b << 8;
    c = a >> 8;

    printf("%d\n", c);

	return 0;
}


