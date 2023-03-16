



#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <iostream>
using namespace std;


struct PCB{
    int nJobs;
};

int main(int argc, char **argv) {

    PCB* procCtrltble = new PCB[3];

    procCtrltble[0].nJobs = 100;

    printf("%d\n", procCtrltble[0].nJobs);

	return 0;
}


