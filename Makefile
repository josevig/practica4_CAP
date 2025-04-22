CC=mpicc

prueba: matrices_MPI_OpenMP.c 
	$(CC) -fopenmp  matrices_MPI_OpenMP.c -o  matrices_MPI_OpenMP.exe

clean: 
	rm *.exe
