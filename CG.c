#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ITER_TIMES 200

double MatrixDotProduct(double** A, double** B, int rows, int cols, int shift);
void MatrixAdd(double** A, double** B, double a, double b, double** C, int rows, int cols, int shiftA, int shiftB, int shiftC);
void exchangeBoundaryValues(double** d, int numRows, int numCols, int* neighborProcs);


int main(int argc, char** argv) {
    /* Initialize MPI */
    int myid, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    int n = atoi(argv[1]);
    double h = 1.0 / (n + 1);

    // Distribute the mesh points among processes
    int gridDim = sqrt(numprocs);  // Assume numprocs is a perfect square
    int row = myid / gridDim;      // row index
    int col = myid % gridDim;      // column index
    int neighborProcs[4];          // up, down, left, right
    neighborProcs[0] = myid - gridDim >= 0 ? myid - gridDim : -1;
    neighborProcs[1] = myid + gridDim < numprocs ? myid + gridDim : -1;
    neighborProcs[2] = myid % gridDim != 0 ? myid - 1 : -1;
    neighborProcs[3] = myid % gridDim != gridDim - 1 ? myid + 1 : -1;

    // Calculate start and end index of the mesh points' block
    int blockSize = n / gridDim;
    int residual = n % gridDim;
    int numRows = blockSize, numCols = blockSize;
    int I_START, I_END, J_START, J_END;
    if(row < residual) {
        numRows++;
        I_START = row * numRows + 1;
    }else{
        I_START = row * numRows + residual + 1; 
    }
    I_END = I_START + numRows - 1;

    if(col < residual) {
        numCols++;
        J_START = col * numCols + 1;
    }else{
        J_START = col * numCols + residual + 1;
    }
    J_END = J_START + numCols - 1;

    // Allocate memory for b, u, g, q
    double** b = (double**)malloc(numRows * sizeof(double*));
    double** u = (double**)malloc(numRows * sizeof(double*));
    double** g = (double**)malloc(numRows * sizeof(double*));
    double** q = (double**)malloc(numRows * sizeof(double*));
    for (int i = 0; i < numRows; i++) {
        b[i] = (double*)malloc(numCols * sizeof(double));
        u[i] = (double*)malloc(numCols * sizeof(double));
        g[i] = (double*)malloc(numCols * sizeof(double));
        q[i] = (double*)malloc(numCols * sizeof(double));
    }

    // Initialize b
    double x, y;
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            x = (I_START + i) * h;
            y = (J_START + j) * h;
            b[i][j] = 2 * h * h * (x * (1-x) + y * (1-y));
        }
    }


    // Start time measurement
    double startTime, endTime;
    startTime = MPI_Wtime();
    // 1.1 u = 0
    // 1.2 g = -b
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            u[i][j] = 0;
            g[i][j] = -b[i][j];
        }
    }
    // 1.3 d = b
    // need to extend the size of the matrix by 1
    int extendedRowSize = numRows + 2;
    int extendedColSize = numCols + 2;
    double** d = (double**)malloc(extendedRowSize * sizeof(double*));
    for (int i = 0; i < extendedRowSize; i++) {
        d[i] = (double*)malloc(extendedColSize * sizeof(double));
    }
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            d[i + 1][j + 1] = b[i][j];
        }
    }
    // initialize the boundary values
    for(int i = 0; i < extendedRowSize; i++) {
        for(int j = 0; j < extendedColSize; j++) {
            if(i == 0 || i == extendedRowSize - 1 || j == 0 || j == extendedColSize - 1) {
                d[i][j] = 0;
            }
        }
    }

    // 1.4 q0 = g^T * g
    double q0 = MatrixDotProduct(g, g, numRows, numCols, 0);
    MPI_Allreduce(MPI_IN_PLACE, &q0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // 2. CG iteration
    int iter = 0;
    double tau, q1, beta, dotProduct;
    while (iter < ITER_TIMES) {        
        // 2.1 q = Ad
        exchangeBoundaryValues(d, extendedRowSize, extendedColSize, neighborProcs);
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                q[i][j] = -1 * (d[i + 1][j] + d[i][j + 1] + d[i + 1][j + 2] + d[i + 2][j + 1]) + 4 * d[i + 1][j + 1];

            }
        }

        // 2.2 tau = q0 / d^T * q
        dotProduct = MatrixDotProduct(d, q, numRows, numCols, 1);
        MPI_Allreduce(MPI_IN_PLACE, &dotProduct, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        tau = q0 / dotProduct;

        // 2.3 u = u + tau * d
        MatrixAdd(u, d, 1.0, tau, u, numRows, numCols, 0, 1, 0);

        // 2.4 g = g + tau * q
        MatrixAdd(g, q, 1.0, tau, g, numRows, numCols, 0, 0, 0);

        // 2.5 q1 = g^T * g
        q1 = MatrixDotProduct(g, g, numRows, numCols, 0);
        MPI_Allreduce(MPI_IN_PLACE, &q1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
 
        // 2.6 beta = q1 / q0;
        beta = q1 / q0;

        // 2.6 d = -g + beta * d
        MatrixAdd(g, d, -1.0, beta, d, numRows, numCols, 0, 1, 1);
        
        // 2.7 q0 = q1
        q0 = q1;

        iter++;
    }

    // 3. Calculate the norm of g
    double norm;
    dotProduct = MatrixDotProduct(g, g, numRows, numCols, 0);
    MPI_Reduce(&dotProduct, &norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    endTime = MPI_Wtime();
    double exe_time = endTime - startTime;
    double max_exe_time;
    MPI_Reduce(&exe_time, &max_exe_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(myid == 0) {
        // printf("%f\n", max_exe_time);
        printf("The norm of the vector g is %.10f.\n", sqrt(norm));
    }

    // Free memory
    for (int i = 0; i < extendedRowSize; i++) {
        free(d[i]);
    }
    free(d);

    for (int i = 0; i < numRows; i++) {
        free(b[i]);
        free(u[i]);
        free(g[i]);
        free(q[i]);
    }
    free(b);
    free(u);
    free(g);
    free(q);

    MPI_Finalize();
    return 0;
}

double MatrixDotProduct(double** A, double** B, int rows, int cols, int shift) {
    double sum = 0.0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sum += A[i + shift][j + shift] * B[i][j];
        }
    }
    return sum;
}

void MatrixAdd(double** A, double** B, double a, double b, double** C, int rows, int cols, int shiftA, int shiftB, int shiftC) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            C[i+shiftC][j+shiftC] = a * A[i+shiftA][j+shiftA] + b * B[i + shiftB][j + shiftB];
        }
    }
}

void exchangeBoundaryValues(double** d, int numRows, int numCols, int* neighborProcs) {
    if (neighborProcs[0] != -1) {  // up
        MPI_Sendrecv(&d[1][0], numCols, MPI_DOUBLE, neighborProcs[0], 0,
                     &d[0][0], numCols, MPI_DOUBLE, neighborProcs[0], 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (neighborProcs[1] != -1) {  // down
        MPI_Sendrecv(&d[numRows - 2][0], numCols, MPI_DOUBLE, neighborProcs[1], 0,
                     &d[numRows - 1][0], numCols, MPI_DOUBLE, neighborProcs[1], 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (neighborProcs[2] != -1) {  // left
        double leftSend[numRows], leftRecv[numRows];
        for (int i = 0; i < numRows; i++) {
            leftSend[i] = d[i][1];
        }
        MPI_Sendrecv(leftSend, numRows, MPI_DOUBLE, neighborProcs[2], 0,
                     leftRecv, numRows, MPI_DOUBLE, neighborProcs[2], 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < numRows; i++) {
            d[i][0] = leftRecv[i];
        }
    }
    if (neighborProcs[3] != -1) {  // right
        double rightSend[numRows], rightRecv[numRows];
        for (int i = 0; i < numRows; i++) {
            rightSend[i] = d[i][numCols-2];
        }
        MPI_Sendrecv(rightSend, numRows, MPI_DOUBLE, neighborProcs[3], 0,
                     rightRecv, numRows, MPI_DOUBLE, neighborProcs[3], 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < numRows; i++) {
            d[i][numCols - 1] = rightRecv[i];
        }
    }
}