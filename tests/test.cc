



#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <iostream>
using namespace std;

#define SEND_INT 0
#define STATUS_IDLE 1

void receiver(int rank, int world_size){

	int flag, recvSz;
    MPI_Status status;
    double a = 1.0, b = 1.00000000001;
    
    int data[1000000];
    int status_idle = 0, all_idle_mask = ((1 << (world_size - 1)) - 1) << 1;
   
    printf("[proc %d] doing other work...\n", rank);
    for(unsigned int i = 0; i < 100000000; i++){
        for(unsigned int j = 0; j < 10; j++){
            a = a * b;
        }
    }
    printf("[proc %d] finish other work: a: %f. Begin to recv...\n", rank, a);



    while(1){
        
        for(int i=1; i<world_size; i++){
            
            MPI_Iprobe(i, SEND_INT, MPI_COMM_WORLD, &flag, &status);
            if(!flag){
                continue;
            }


            MPI_Get_count(&status, MPI_INT, &recvSz);
            MPI_Recv(data, recvSz, MPI_INT, i, SEND_INT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
           
            printf("[proc %d] recv from proc %d. recvSz: %d\n", rank, i, recvSz);
      		
        }
    }
}



void sender(int rank){
	int sz = 1000000;
	int data[sz];

    for(int i = 0; i < sz; i++){
        data[i] = i*2;        
    }	
    printf("[proc %d] send data. size: %d\n", rank, sz);
    MPI_Send(data, sz, MPI_INT, 0, SEND_INT, MPI_COMM_WORLD);
}






int main(int argc, char **argv) {


	int rank, world_size;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	

	if (rank == 0){
		
        receiver(rank, world_size);

	}
	else{
		
        sender(rank);

	}

	return 0;
}


