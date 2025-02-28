#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
#define T(i, j) (T[(i)*(n_cells+2) + (j)])
#define T_new(i, j) (T_new[(i)*(n_cells+2) + (j)])

double MAX_RESIDUAL = 1.e-8;

void kernel(double *T, int n_cells, int max_iterations);
void write_to_csv(double *T, int n_cells, const char *filename);

int main(int argc, char *argv[]){

	int n_cells = 1000;
	int max_iterations = 100;

	int SIZE = (n_cells + 2) * (n_cells + 2);

	double *T = (double *)malloc(SIZE * sizeof(double));
	
	unsigned int Nthrds = 0;

	sscanf(argv[1], "%d", &Nthrds);
	
	omp_set_dynamic(0);
	omp_set_num_threads(Nthrds);
	
	// initialize grid and boundary conditions
	for (unsigned i = 0; i <= n_cells + 1; i++) {
		for (unsigned j = 0; j <= n_cells + 1; j++) {
			if((j == 0) || (j == (n_cells + 1))){
				T(i, j) = 1.0;
			}
			else {
				T(i, j) = 0.0;
			}
		}
	}

	kernel (T, n_cells, max_iterations);

	char filename[50];
	
	snprintf(filename, sizeof(filename), "P04_Serial_%d_threads.csv", Nthrds);

	write_to_csv(T, n_cells, filename);

	free(T);

	return 0;

}


void kernel (double *T, int n_cells, int max_iterations) {

	int iteration = 0;
	double residual = 1.e6;

	int SIZE = (n_cells + 2) * (n_cells + 2);

	double *T_new = (double *)malloc(SIZE * sizeof(double));
	
	float start_time = omp_get_wtime();
	
	#pragma omp target data map(tofrom: T[0:SIZE]) map(alloc: T_new[0:SIZE])
	{
	
		while (residual > MAX_RESIDUAL) {
			residual = 0.0;
			
			#pragma omp target teams distribute parallel for simd collapse(2)
				for (unsigned i = 1; i <= n_cells; i++) {
					for (unsigned j = 1; j <= n_cells; j++) {
						T_new(i, j) = 0.25 * (T(i + 1, j) + T(i - 1, j) + T(i, j + 1) + T(i, j - 1));
					}
				}
			

			#pragma omp target teams distribute parallel for simd reduction(max: residual) collapse(2)
				for (unsigned int i = 1; i <= n_cells; i++) {
					for (unsigned int j = 1; j <= n_cells; j++) {
						residual = MAX(fabs(T_new(i, j) - T(i, j)), residual);
						T(i, j) = T_new(i, j);
					}
				}

			iteration++;

		}
	}	

	
	float run_time = omp_get_wtime() - start_time;
	
	printf("Run time: %f \n \n", run_time);

	printf("residual = %.9e\n", residual);
	
	free(T_new);
}


void write_to_csv(double *T, int n_cells, const char *filename) {
	FILE *file = fopen(filename, "w");
	if (!file) {
		fprintf(stderr, "Error: could not open file %s for writing.\n", filename);
		return;
	}

	for (unsigned i = 0; i <= n_cells + 1; i++){
		for (unsigned j = 0; j <= n_cells + 1; j++) {
			fprintf(file, "%.6f", T(i,j));
			if (j < n_cells + 1){
				fprintf(file, ",");
			}
		}
		fprintf(file, "\n");
	}

	fclose(file);
	printf("Matrix saved to %s\n", filename);



}


