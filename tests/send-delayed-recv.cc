

#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <iostream>
using namespace std;

#define SEND_INT 0
int main(int argc, char **argv) {


	int rank, world_size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int sz = 10;
	int data[sz];


	if (rank == 1){
		for(int i=0;i<sz;i++){
			data[i] = i * 3;
			printf("[proc %d] send: %d\n", rank, data[i]);
		}	
		
		MPI_Send(data, 100, MPI_INT, 0, SEND_INT, MPI_COMM_WORLD);
	}
	else if (rank == 0){
		double a = 1.0, b = 1.00000000001;
		for(unsigned int i = 0; i < 100000000; i++){
			for(unsigned int j = 0; j < 10; j++){
				a = a * b;
			}
		}
		printf("a: %f\n", a);

		MPI_Recv(data, 100, MPI_INT, 1, SEND_INT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		for(int i=0;i<sz;i++){
			printf("[proc %d] receive: %d\n", rank, data[i]);
		}	

	}
		


	return 0;
}