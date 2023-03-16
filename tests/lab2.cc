



#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <iostream>
using namespace std;

#define SEND_INT 0
#define STATUS_IDLE 1



int main(int argc, char **argv) {


	int rank, world_size;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int sz = 10, recvSz;
	int n_msg = 5;
	int data[sz] = {25, 36, 144, 5, 1001, 78, 9999, 658, 101010, 12};
    int status_idle = 0, all_idle_mask = ((1 << (world_size - 1)) - 1) << 1;
	int sendSz, sendSzs[n_msg] = {2, 5, 4, 8, 6};
	int flag;

	double a = 1.0, b = 1.00000000001;

	if (rank == 0){
		

			int b;
	
			MPI_Recv(&b, 1, MPI_INT, 1, SEND_INT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			printf("[proc %d] recv: %d", rank, b);


	
	



	}
	else{
		
		for(int j = 0; j < n_msg; j++){
			
			int a = 4;

			MPI_Send(&a, 1, MPI_INT, 0, SEND_INT, MPI_COMM_WORLD);
		}



	}

	return 0;
}









// int main(int argc, char** argv) {
// 	if (argc != 3) {
// 		fprintf(stderr, "must provide exactly 2 arguments!\n");
// 		return 1;
// 	}

// 	unsigned long long r = atoll(argv[1]);
// 	unsigned long long k = atoll(argv[2]);
// 	unsigned long long global_sum = 0, local_sum = 0;
// 	unsigned long long y, r2;
// 	unsigned long long workSize, startIdx, stopIdx;

// 	int rank, world_size;
// 	MPI_Init(&argc, &argv);
// 	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
// 	MPI_Comm_size(MPI_COMM_WORLD, &world_size);


// 	workSize =  r / world_size; 
// 	startIdx = workSize * rank;
// 	if(rank < world_size - 1){stopIdx = workSize * (rank + 1) - 1;}
// 	else if(rank == world_size - 1){stopIdx = r - 1;}

// 	r2 = r*r;
// 	for (unsigned long long x = startIdx; x <= stopIdx; x++) {
// 		y = ceil(sqrtl(r2 - x*x));
// 		local_sum += y;
// 		local_sum %= k;
// 	}

// 	MPI_Reduce(&local_sum, &global_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

// 	global_sum %= k;
// 	if (rank == 0) {printf("%llu\n", (4 * global_sum) % k);}
// }





// #include <assert.h>
// #include <stdio.h>
// #include <math.h>
// #include <mpi.h>
// #include <iostream>
// using namespace std;

// int main(int argc, char** argv) {
// 	if (argc != 3) {
// 		fprintf(stderr, "must provide exactly 2 arguments!\n");
// 		return 1;
// 	}

// 	unsigned long long r = atoll(argv[1]);
// 	unsigned long long k = atoll(argv[2]);
// 	unsigned long long global_sum = 0, local_sum = 0;
// 	unsigned long long y, r2;
// 	unsigned long long workSize, startIdx, stopIdx;

// 	int rank, world_size;
// 	MPI_Init(&argc, &argv);
// 	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
// 	MPI_Comm_size(MPI_COMM_WORLD, &world_size);


// 	workSize =  r / world_size; 
// 	startIdx = workSize * rank;
// 	if(rank < world_size - 1){stopIdx = workSize * (rank + 1) - 1;}
// 	else if(rank == world_size - 1){stopIdx = r - 1;}

// 	r2 = r*r;
// 	for (unsigned long long x = startIdx; x <= stopIdx; x++) {
// 		y = ceil(sqrtl(r2 - x*x));
// 		local_sum += y;
// 		local_sum %= k;
// 	}

// 	MPI_Reduce(&local_sum, &global_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

// 	global_sum %= k;
// 	if (rank == 0) {printf("%llu\n", (4 * global_sum) % k);}
// }
