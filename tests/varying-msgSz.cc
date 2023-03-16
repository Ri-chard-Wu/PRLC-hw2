



#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <iostream>
using namespace std;

#define SEND_INT 0
int main(int argc, char **argv) {


	int rank, world_size;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int sz = 10, recvSz;
	int n_msg = 5;
	int data[sz] = {25, 36, 144, 5, 1001, 78, 9999, 658, 101010, 12};
	int sendSz, sendSzs[n_msg] = {2, 5, 4, 8, 6};
	int flag;

	double a = 1.0, b = 1.00000000001;

	if (rank == 0){
		

		while(1){
			printf("[proc %d] doing other work...\n", rank);
			for(unsigned int i = 0; i < 100000000; i++){
				for(unsigned int j = 0; j < 10; j++){
					a = a * b;
				}
			}
			printf("[proc %d] finish other work: a: %f. Begin to recv...\n", rank, a);


			
			for(int i=1; i<world_size; i++){
				
				MPI_Iprobe(i, SEND_INT, MPI_COMM_WORLD, &flag, &status);
				if(!flag){
					printf("[proc %d] no msg from proc %d.\n", rank, i);
					continue;
				}

				MPI_Get_count(&status, MPI_INT, &recvSz);
				MPI_Recv(data, recvSz, MPI_INT, i, SEND_INT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				printf("[proc %d] recv from proc %d: ", rank, i);
				for(int j = 0; j < recvSz; j++){
					printf("%d, ", data[j]);
				}	
				printf("\n");			


			}
		}

	}
	else{
		
		for(int j = 0; j < n_msg; j++){

			sendSz = sendSzs[j];

			printf("[proc %d] %d'th send: ", rank, j);

			for(int i = 0; i < sendSz; i++){
				printf("%d, ", data[i]);
			}	
			printf("\n");
			MPI_Send(data, sendSz, MPI_INT, 0, SEND_INT, MPI_COMM_WORLD);
		}


		printf("[proc %d] do other work...\n", rank);

	}

	return 0;
}






