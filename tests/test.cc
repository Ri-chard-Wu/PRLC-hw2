



#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <iostream>
using namespace std;

struct JCB{
    int i;
    int j;
};

int main(int argc, char **argv) {


    JCB jcb;
    int* ptr;
    int a = 4;

    ptr = &(jcb.i);
    // ptr = &a;
    *ptr = 45;
    printf("%d\n", jcb.i);

	return 0;
}


