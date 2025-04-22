#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Inicializa una matriz con valores aleatorios en [0,1) */
void init_matrix(double *M, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            M[i * cols + j] = (double)rand() / (double)RAND_MAX;
}

/* Pone en cero una matriz */
void zero_matrix(double *M, int rows, int cols) {
    memset(M, 0, rows * cols * sizeof(double));
}

/* Imprime una matriz (sólo si es pequeña) */
void print_matrix(double *M, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.3f ", M[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

/* Multiplicación de matrices: C = A * B */
void mult(double *A, double *B, double *C, int m, int k, int n) {
    #pragma omp for schedule(dynamic) collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum=0;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/* Suma de matrices: C = A + B */
void sum_matrix(double *A, double *B, double *C, int rows, int cols) {
    #pragma omp for schedule(dynamic) collapse(2) 
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            C[i * cols + j] = A[i * cols + j] + B[i * cols + j];
}

void scatter_blockcyclic(double *M, double *local_M, int dim, int size, int rank) {
    if (rank == 0) {
        for (int i = 0; i < dim; i++) {
            int dest = i % size;
            if (dest == 0) {
                memcpy(local_M + (i / size) * dim, M + i * dim, dim * sizeof(double));
            } else {
                MPI_Send(M + i * dim, dim, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        for (int i = 0; i < dim / size; i++) {
            MPI_Recv(local_M + i * dim, dim, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

void gather_blockcyclic(double *local_M, double *M, int dim, int size, int rank) {
    if (rank == 0) {
        for (int i = 0; i < dim; i++) {
            int src = i % size;
            if (src == 0) {
                memcpy(M + i * dim, local_M + (i / size) * dim, dim * sizeof(double));
            } else {
                MPI_Recv(M + i * dim, dim, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        for (int i = 0; i < dim / size; i++) {
            MPI_Send(local_M + i * dim, dim, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char **argv) {
    int rank, size, dim, n_threads;

    dim = atoi(argv[1]);
    n_threads = atoi(argv[2]);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    /* Difunde la dimensión a todos los procesos */
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Comprobamos que la dimensión sea divisible por el número de procesos */
    if (dim % size != 0) {
        if (rank == 0)
            printf("La dimensión (%d) debe ser divisible por el número de procesos (%d).\n", dim, size);
        MPI_Finalize();
        return 1;
    }
    int rows_per_proc = dim / size;

    /* Cada proceso reserva memoria para su bloque de filas */
    double *local_A = (double *)malloc(rows_per_proc * dim * sizeof(double));
    double *local_C = (double *)malloc(rows_per_proc * dim * sizeof(double));
    double *local_D = (double *)malloc(rows_per_proc * dim * sizeof(double));
    double *local_E = (double *)malloc(rows_per_proc * dim * sizeof(double));

    /* Cada proceso necesita la matriz B completa */
    double *B = (double *)malloc(dim * dim * sizeof(double));

    /* El proceso 0 reserva e inicializa la matriz A completa y la matriz B */
    double *A = NULL;
    double *C = NULL, *D = NULL, *E = NULL;

    if (rank == 0) {
        A = (double *)malloc(dim * dim * sizeof(double));
        C = (double *)malloc(dim * dim * sizeof(double));
        D = (double *)malloc(dim * dim * sizeof(double));
        E = (double *)malloc(dim * dim * sizeof(double));
        init_matrix(A, dim, dim);
        init_matrix(B, dim, dim);
        // Inicializar C, D, E a cero
        zero_matrix(C, dim, dim);
        zero_matrix(D, dim, dim);
        zero_matrix(E, dim, dim);
    }

    /* Distribuye las filas de A entre todos los procesos */
    // MPI_Scatter(A, rows_per_proc * dim, MPI_DOUBLE, local_A, rows_per_proc * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    scatter_blockcyclic(A, local_A, dim, size, rank);
    /* Difunde la matriz B a todos los procesos */
    MPI_Bcast(B, dim * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Inicializa en cero las matrices locales */
    zero_matrix(local_C, rows_per_proc, dim);
    zero_matrix(local_D, rows_per_proc, dim);
    zero_matrix(local_E, rows_per_proc, dim);

    double t_start, t_end, time_mult1, time_mult2, time_sum;
    double max_time_mult1, max_time_mult2, max_time_sum;

    #pragma omp parallel num_threads(n_threads) private(t_start,t_end) reduction(max: time_mult1,time_mult2,time_sum)
    {
        //Operación 1: C = A * B
        t_start=omp_get_wtime();
        mult(local_A,B,local_C,rows_per_proc,dim,dim);
        t_end=omp_get_wtime();
        time_mult1= t_end - t_start;

        //Operación 2: D = C * B
        t_start=omp_get_wtime();
        mult(local_C,B,local_D,rows_per_proc,dim,dim);
        t_end=omp_get_wtime();
        time_mult2= t_end - t_start;

        //Operación 3: E = D + C
        t_start=omp_get_wtime();
        sum_matrix(local_D,local_C,local_E,rows_per_proc,dim);
        t_end=omp_get_wtime();
        time_sum= t_end - t_start;
    }

    MPI_Reduce(&time_mult1, &max_time_mult1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&time_mult2, &max_time_mult2, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&time_sum, &max_time_sum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Reunir las partes de E en el proceso 0
    //MPI_Gather(local_E, rows_per_proc * dim, MPI_DOUBLE, E, rows_per_proc * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    gather_blockcyclic(local_E, E, dim, size, rank);

    if (dim <= 10) {
        // Reunir los valores locales de C y D para poder imprimir por pantalla la matriz C y D completa.
        //MPI_Gather(local_C, rows_per_proc * dim, MPI_DOUBLE, C, rows_per_proc * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        gather_blockcyclic(local_C, C, dim, size, rank);
        //MPI_Gather(local_D, rows_per_proc * dim, MPI_DOUBLE, D, rows_per_proc * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        gather_blockcyclic(local_D, D, dim, size, rank);
    }

    if (rank == 0) {
        printf("Tiempo de ejecucion de C = A * B: %f segundos\n", max_time_mult1);
        printf("Tiempo de ejecucion de D = C * B: %f segundos\n", max_time_mult2);
        printf("Tiempo de ejecucion de E = D + C: %f segundos\n", max_time_sum);
        printf("Tiempo de ejecucion total: %f segundos\n", max_time_mult1+max_time_mult2+max_time_sum);
    }

    /* Opcional: reunir resultados para imprimir las matrices si la dimensión es pequeña */
    if (rank == 0 && dim <= 10) {
        printf("\nMatrix A:\n");
        print_matrix(A, dim, dim);
        printf("Matrix B:\n");
        print_matrix(B, dim, dim);
        printf("Matrix C = A * B:\n");
        print_matrix(C, dim, dim);
        printf("Matrix D = C * B:\n");
        print_matrix(D, dim, dim);
        printf("Matrix E = D + C:\n");
        print_matrix(E, dim, dim);
    }

    /* Liberar memoria */
    if (rank == 0) {
        free(A);
        free(C);
        free(D);
        free(E);
    }
    free(B);
    free(local_A);
    free(local_C);
    free(local_D);
    free(local_E);

    MPI_Finalize();
    return 0;
}