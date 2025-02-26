#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
#define T(i, j) (T[(i)*(n_cells+2) + (j)])
#define T_new(i, j) (T_new[(i)*(n_cells+2) + (j)])

double MAX_RESIDUAL = 1.e-15;

int main(int argc, char *argv[]){


	// initialize grid and boundary conditions
	for (unsigned i = 0; i <= n_cells + 1; i++)
		for (unsigned j = 0; j <= n_cells + 1; j++)
			if((j == 0) || (j == (n_cells + 1)))
				T(i, j) = 1.0;
    			else
      				T(i, j) = 0.0;


}


void kernel(double *T, int max_iterations) {

  int iteration = 0;
  double residual = 1.e6;
  double *T_new = (double *)malloc(SIZE * sizeof(double));
  while (residual > MAX_RESIDUAL && iteration < max_iterations) {
    for (unsigned i = 1; i <= N; i++)
      for (unsigned j = 1; j <= N; j++)
        T_new(i, j) =
            0.25 * (T(i + 1, j) + T(i - 1, j) + T(i, j + 1) + T(i, j - 1));
    residual = 0.0;
    for (unsigned int i = 1; i <= N; i++) {
      for (unsigned int j = 1; j <= N; j++) {
        residual = MAX(fabs(T_new(i, j) - T(i, j)), residual);
        T(i, j) = T_new(i, j);
      }
    }
    iteration++;
  }
  printf("residual = %.9e\n", residual);
  free(T_new);
}
