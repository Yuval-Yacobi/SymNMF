#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "symnmf.h"

/**
Initializes a new vector node for the linked list.
Return: Pointer to vector or NULL if malloc fails.
*/
vector* create_vector_node() {
    vector *v = malloc(sizeof(vector));
    if (v== NULL){
        printf("An Error Has Occurred\n");
        return NULL;
    }
    v->next = NULL;
    v->cords = NULL;
    return v;
}

/* Reads a single line of coordinates from a file into a linked list.
Parameter fp: File pointer to read from.
Parameter d: Updated with the vector dimension.
Parameter N: Used to identify the first row.
Parameter last_c: Stores the last char read.
Return: Head of coordinate list or NULL on error. */  
static cord* read_coords(FILE *fp, int *d, int N, char *last_c) {
    double val;
    cord *head = NULL;
    cord *curr = NULL;
    cord *new_node;
    while (fscanf(fp, "%lf", &val) == 1) {
        new_node = malloc(sizeof(cord));
        if (!new_node) {
            printf("An Error Has Occurred\n");
            return NULL;
        }
        new_node->value = val;
        new_node->next = NULL;
        if (!head) {
            head = new_node;
        } else {
            curr->next = new_node;
        }
        curr = new_node;
        if (N == 0) {
            (*d)++;
        }
        *last_c = fgetc(fp);
        if (*last_c == '\n' || *last_c == EOF) {
            break;
        }
    }
    return head;
}

/* Reads the entire file and builds a linked list of vectors.
Parameter filename: String path to the data file.
Parameter N: Updated with total number of points.
Parameter d: Updated with the point dimension.
Return: Head of the vector linked list, or NULL on failure. */
vector* file_to_list(char *filename, int *N, int *d) {
    FILE *fp;
    vector *head = NULL, *curr_v = NULL, *new_v;
    cord *row_cords;
    char last_c = ' ';
    fp = fopen(filename, "r");
    if (!fp) {
        printf("An Error Has Occurred\n");
        return NULL;}
    *N = 0;
    *d = 0;
    while (last_c != EOF) {
        row_cords = read_coords(fp, d, *N, &last_c);
        if (!row_cords) {
            break;}
        new_v = create_vector_node();
        if (!new_v) {
            free_list(head);
            fclose(fp);
            return NULL;}
        new_v->cords = row_cords;
        if (!head) {
            head = new_v;
        } else {
            curr_v->next = new_v;}
        curr_v = new_v;
        (*N)++;
    }
    fclose(fp);
    return head;
}

/* Frees all memory allocated for the list of vectors.
Parameter head: Pointer to the list head.
Iterates through each vector, first freeing its internal coordinates list, 
and then freeing the vector node itself.
 */
void free_list(vector *head) {
    vector *curr_v, *tmp_v;
    cord *curr_c, *tmp_c;
    
    curr_v = head;
    while (curr_v != NULL) {
        curr_c = curr_v->cords;
        while (curr_c != NULL) {
            tmp_c = curr_c;
            curr_c = curr_c->next;
            free(tmp_c);
        }
        tmp_v = curr_v;
        curr_v = curr_v->next;
        free(tmp_v);
    }
}

/* Copies values from the linked list to a 2D array.
 * Iterates through each vector and its coordinates to populate the matrix rows.
 * Parameter head: Pointer to the start of the vector linked list.
 * Parameter mat: Pre-allocated 2D array to store the coordinate values.
 * Parameter N, d: Number of points and dimensions of each point. */
void list_to_matrix(vector *head, double **mat, int N, int d) {
    int i, j;
    vector *curr_v = head;
    cord *curr_c;
    
    for (i = 0; i < N; i++) {
        curr_c = curr_v->cords;
        for (j = 0; j < d; j++) {
            mat[i][j] = curr_c->value;
            curr_c = curr_c->next;
        }
        curr_v = curr_v->next;
    }
}


/**
 * Allocates a 2D matrix in a contiguous memory block.
 * rows_n: Number of rows.
 * cols_d: Number of columns.
 * returns: Pointer to the matrix (array of row pointers), or NULL on failure.
 */
double** allocate_matrix(int rows_n, int cols_d){
    int i;
    double *data, **matrix;
    matrix = (double **)malloc(rows_n*sizeof(double *));
    if (matrix == NULL){
        printf("An Error Has Occurred\n");
        return NULL;
    }
    data = (double *)calloc(rows_n* cols_d, sizeof(double));
    if (data == NULL) {
        free(matrix); 
        printf("An Error Has Occurred\n");
        return NULL;
    }
    for (i = 0; i < rows_n; i++) {
        matrix[i] = data + (i * cols_d);
    }
    return matrix;
}



/*
Calculates the squared Euclidean distance between vectors.
Parameter point1: Pointer to the first vector.
Parameter point2: Pointer to the second vector.
Parameter cols_d: The dimension of the vectors.
Return: Sum of squared differences.
 */
double euclidean_dist_sq(double *point1, double *point2 ,int cols_d) {
    double sum = 0, diff;
    int i;
    for(i=0;i<cols_d;i++) {
        diff = point1[i] - point2[i];
        sum += diff * diff;
    }
    return sum;
}

/*
Computes the similarity matrix A from the input points.
Parameter points: 2D array of input data points.
Parameter N: Number of points.
Parameter d: Dimension of each point.
A[i][j] = exp(-||x_i - x_j||^2 / 2) for i != j, and 0 for i == j.
Return: N x N similarity matrix A or NULL.
 */
double** sym(double** points, int N, int d){
    int i,j;
    double dist, val;
    double **A = allocate_matrix(N,N);
    if (A == NULL){
        return NULL;
    }
    for(i=0; i<N;i++){
        A[i][i]=0;
        for(j=i+1; j<N; j++){
            dist = euclidean_dist_sq(points[i],points[j], d);
            val = exp(-dist/2.0);
            A[i][j] = val;
            A[j][i] = val;
        }
    }
    return A;
}

/**
High-level wrapper to compute the Diagonal Degree Matrix from input points.
Internally generates A, computes D, and manages memory for A.
Parameter points: 2D array of input data.
Parameter N: Number of points.
Parameter d: Dimension of points.
Return: N x N diagonal matrix or NULL.
 */
double** ddg(double** points, int N, int d){
    double **A, **diagonal;
    A = sym(points, N, d);
    if (A == NULL) {
        return NULL;
    }
    diagonal = calculate_ddg(A, N);
    free_matrix(A);
    return diagonal;
}

/**
Calculates the Diagonal Degree Matrix from an existing similarity matrix A.
Sums each row of A to populate the diagonal elements of D.
Parameter A: An N x N similarity matrix.
Parameter N: Number of points.
Return: N x N diagonal matrix or NULL.
 */
double** calculate_ddg(double** A, int N){
    double **diagonal;
    int i, j;
    double rowSum;
    diagonal = allocate_matrix(N,N);
    if (diagonal== NULL){
        return NULL;
    }
    for (i = 0; i < N; i++) {
        rowSum = 0;
        for (j = 0; j < N; j++) {
            rowSum += A[i][j];
        }
        diagonal[i][i] = rowSum;
    }
    return diagonal;
}

/** Computes the Normalized Similarity Matrix W = D^-1/2 * A * D^-1/2.
Parameter points: 2D array of input data, Parameter N: Number of points. Parameter d: Dimension of points.
Return: N x N normalized matrix W or NULL. */
double** norm(double** points, int N, int d){
    int i,j;
    double *D_sqrt, **W, **D, **A;
    A = sym(points, N, d);
    if (A == NULL) {
        return NULL;}
    D = calculate_ddg(A, N);
    if (D == NULL ){
        free_matrix(A);
        return NULL;}
    D_sqrt = (double*)malloc(N*sizeof(double));
    if (D_sqrt== NULL ){
        printf("An Error Has Occurred\n");
        free_matrix(A);
        free_matrix(D);
        return NULL;}
    W = allocate_matrix(N,N);
    if(W == NULL){
        free_matrix(A);
        free_matrix(D);
        free(D_sqrt);
        return NULL;}
    for (i=0;i<N; i++){ /* calculate D^-1/2 for all diagonal elements */
        D_sqrt[i] = (D[i][i] != 0) ? 1.0 / sqrt(D[i][i]) : 0;
    }
    for(i=0;i<N;i++){
        for(j=0;j<N;j++){
            W[i][j] = D_sqrt[i] * A[i][j] * D_sqrt[j];
        }
    }
    free_matrix(A);
    free_matrix(D);
    free(D_sqrt);
    return W;
}


/*
Performs standard matrix multiplication: matrix1 * matrix2.
The number of columns in matrix1 must equal the number of rows in matrix2.
Parameter matrix1: First input matrix.
Parameter rows1: Number of rows in matrix1.
Parameter cols1: Number of columns in matrix1.
Parameter matrix2: Second input matrix.
Parameter rows2: Number of rows in matrix2.
Parameter cols2: Number of columns in matrix2.
Return: Result matrix of size rows1 x cols2.
 */
double** matrix_multiplication(double** matrix1, int rows1, int cols1, double** matrix2, int rows2, int cols2){
    int i,j,z;
    double** mul_matrix;
    if (cols1 != rows2){
        printf("An Error Has Occurred\n");
        return NULL;
    }
    mul_matrix = allocate_matrix(rows1, cols2);
    if(mul_matrix==NULL){
        return NULL;
    }
    for(i=0;i<rows1;i++){
        for(j=0;j<cols2;j++){
            for(z=0;z<cols1;z++){
                mul_matrix[i][j] += matrix1[i][z] * matrix2[z][j];
            }
        }
    }
    return mul_matrix;
}

/*
Creates a new matrix that is the transpose of the input matrix.
The rows of the input become the columns of the output and vice versa.
Parameter matrix: Input matrix to transpose.
Parameter rows: Number of rows in the input.
Parameter cols: Number of columns in the input.
Return: New transposed matrix or NULL.
 */
double** transpose(double** matrix,int rows,int cols){
    int i,j;
    double **result;
    result = allocate_matrix(cols,rows);
    if (result == NULL) {
        return NULL;
    }
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}



/** Applies the multiplicative update rule for matrix H.
Parameter H: Current clustering matrix N x k.
Parameter W: Normalized similarity matrix.
Parameter N: Number of points.
Parameter k: Number of clusters.
Return: Updated H matrix of size N x k or NULL.*/
double** update_H(double** H, double** W, int N, int k){
    int i,j;
    double **new_H, **numerator, **denominator, ratio;
    numerator = matrix_multiplication(W, N, N, H, N, k);
    if (numerator == NULL){
        return NULL;
    }
    denominator = calculate_denominator(H, N, k);
    if (denominator == NULL){
        free_matrix(numerator);
        return NULL;
    }
    new_H = allocate_matrix(N,k);
    if (new_H != NULL){
        for(i=0; i<N;i++){
            for(j=0;j<k;j++){
                denominator[i][j] += 1e-6;
                ratio = numerator[i][j]/denominator[i][j];
                new_H[i][j] = H[i][j] * (0.5 + 0.5*ratio);
            }
        }
    }
    free_matrix(numerator);
    free_matrix(denominator);
    if (new_H == NULL){
        return NULL;
    }
    return new_H;
}

/*
Computes the denominator matrix (H * HT * H).
Parameter H: Current clustering matrix.
Parameter N: Number of points.
Parameter k: Number of clusters.
Return: Resulting N x k matrix or NULL.
 */
double** calculate_denominator(double** H, int N, int k) {
    double **H_transpose, **mul_transpose, **denominator;
    H_transpose = transpose(H, N, k);
    if (H_transpose == NULL){
        return NULL;
    }
    mul_transpose = matrix_multiplication(H, N, k, H_transpose, k, N);
    if (mul_transpose == NULL) {
        free_matrix(H_transpose);
        return NULL;
    }
    denominator = matrix_multiplication(mul_transpose, N, N, H, N, k);
    free_matrix(H_transpose);
    free_matrix(mul_transpose);
    return denominator;
}

/*
Calculates the squared F-norm of mat1 - mat2.
This is used to check convergence between iterations of matrix H.
Parameter mat1: The first matrix.
Parameter mat2: The second matrix.
Parameter rows: Number of rows in the matrices.
Parameter cols: Number of columns in the matrices.
Return: Sum of squared differences.
 */
double fnorm_squared(double** mat1, double** mat2, int rows, int cols) {
    double sum = 0.0;
    int i;
    for (i = 0; i < rows; i++) {
        sum += euclidean_dist_sq(mat1[i], mat2[i], cols);
    }
    return sum;
}

/*
Prints an rows x cols matrix to the standard output.
Each value is formatted to 4 decimal places, separated by commas.
Each row of the matrix is printed on a new line.
Parameter matrix: The matrix to be printed.
Parameter rows: Number of rows in the matrix.
Parameter cols: Number of columns in the matrix.
 */
void print_matrix(double **matrix, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            printf("%.4f", matrix[i][j]);
            if (j < cols - 1) {
                printf(",");
            }
        }
        printf("\n");
    }
}

/*
Frees the memory allocated for a matrix.
Parameter matrix: The matrix to be freed.
 */
void free_matrix(double **matrix) {
    if (matrix != NULL) {
        /* Free the contiguous data block  */
        if (matrix[0] != NULL) {
            free(matrix[0]);
        }
        /* Free the array of row pointers */
        free(matrix);
    }
}


/* 
Directs execution based on the goal string.
Parameter points: Input data matrix.
Parameter goal: Command ("sym", "ddg", "norm").
Parameter N: Row dimension.
Parameter d: Column dimension.
Return: Result matrix or NULL on failure..*/
double** execute_goal(double **points, char *goal, int N, int d) {
    if (strcmp(goal, "sym") == 0) {
        return sym(points, N, d);
    } 
    else if (strcmp(goal, "ddg") == 0) {
        return ddg(points, N, d);
    }
    else if (strcmp(goal, "norm") == 0) {
        return norm(points, N, d);
    } 

    printf("An Error Has Occurred\n"); 
    return NULL;
}

/**
Runs the SymNMF algorithm: iteratively updates H until convergence or max_iter.
Parameter H: Initial clustering matrix N x k.
Parameter W: Normalized similarity matrix.
Parameter N: Number of points.
Parameter k: Number of clusters.
Return: Optimized H matrix or NULL.
 */
double** symnmf_algorithm(double** H, double** W, int N, int k) {
    int iter = 0, max_iter = 300;
    double **new_H, epsilon = 1e-4;

    while (iter < max_iter) {
        new_H = update_H(H, W, N, k);
        if (new_H == NULL) {
            return NULL;
        }
        /* Check convergence using squared Frobenius norm */
        if (fnorm_squared(H, new_H, N, k) < epsilon) {
            free_matrix(H);
            H = new_H;
            break;
        }
        free_matrix(H);
        H = new_H;
        iter++;
    }
    return H;
}



/**
Main entry point: validates arguments, loads file into a matrix via a linked list, 
and triggers the execution of the requested mathematical goal.
Parameter argc: Argument count.
Parameter argv: Array containing goal and file path.
Return: 0 on success, 1 on error.
 */
int main(int argc, char **argv){
    int N, d;
    double **points, **res;
    vector *head;
    if (argc != 3) {
        printf("An Error Has Occurred\n");
        return 1; }
    head = file_to_list(argv[2], &N, &d);
    if(head == NULL){
        return 1; }
    points = allocate_matrix(N, d);
    if(points == NULL){
        return 1; }
    list_to_matrix(head, points, N, d);
    free_list(head);
    res = execute_goal(points, argv[1], N, d);
    free_matrix(points);
    if (res == NULL){
        return 1;
    }
    print_matrix(res, N,N);
    free_matrix(res);
    return 0;
}


