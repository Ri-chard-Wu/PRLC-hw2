#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <iostream>
using namespace std;

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}

	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long y;

	int rank, world_size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);



	float local_sum = ((float)rank)*10.0;
	printf("process %d: %f\n", rank, local_sum);

	float global_sum;
	MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 1, MPI_COMM_WORLD);

	if (rank == 1) {
		printf("Total sum = %f\n", global_sum);
	}



	// for (unsigned long long x = 0; x < r; x++) {
	// 	y = ceil(sqrtl(r*r - x*x));
	// 	pixels += y;
	// 	pixels %= k;
	// }
	// printf("%llu\n", (4 * pixels) % k);
}
