#ifndef SYMNMF_H_
#define SYMNMF_H_

/* Represents a single coordinate in a linked list of coordinates */
typedef struct cord {
    double value;
    struct cord *next;
} cord;

/* Represents a vector (point) as a linked list of coordinates */
typedef struct vector { 
    cord *cords; 
    struct vector *next; 
} vector;

/* Allocates a new vector node and sets pointers to NULL. Returns NULL on failure */
vector* create_vector_node();

/* Reads file and builds a linked list of vectors; updates N (rows) and d (cols) */
vector* file_to_list(char *filename, int *N, int *d);

/* Deep-frees the entire linked list structure (vectors and their coordinates) */
void free_list(vector *head);

/* Populates a pre-allocated 2D array with values from the vector linked list */
void list_to_matrix(vector *head, double **mat, int N, int d);

/* Allocates a 2D matrix as a contiguous memory block for efficiency. Returns NULL on failure */
double** allocate_matrix(int rows_n, int cols_d);

/* Computes the squared Euclidean distance between two arrays of doubles */
double euclidean_dist_sq(double *point1, double *point2 ,int cols_d);

/* Computes Similarity Matrix A: A[i][j] = exp(-dist^2/2), A[i][i] = 0 */
double** sym(double** points, int N, int d);

/* High-level wrapper that creates A, calculates D, and frees A */
double** ddg(double** points, int N, int d);

/* Calculates Diagonal Degree Matrix D where D[i][i] is the sum of row i in A */
double** calculate_ddg(double** A, int N);

/* Computes Normalized Similarity Matrix W = D^-1/2 * A * D^-1/2 */
double** norm(double** points, int N, int d);

/* Standard matrix multiplication (matrix1 * matrix2). Requires cols1 == rows2 */
double** matrix_multiplication(double** matrix1, int rows1, int cols1, double** matrix2, int rows2, int cols2);

/* Returns a new matrix that is the transpose of the input matrix */
double** transpose(double** matrix,int rows,int cols);

/* Updates H using the rule: H = H * (0.5 + 0.5 * ((W*H) / (H*H^T*H))) */
double** update_H(double** H, double** W, int N, int k);

/* Calculates the matrix product H * H^T * H (the denominator of the update rule) */
double** calculate_denominator(double** H, int N, int k);

/* Calculates the squared Frobenius norm of the difference (mat1 - mat2) */
double fnorm_squared(double** mat1, double** mat2, int rows, int cols);

/* Formats and prints matrix to stdout: 4 decimal places, rows on new lines, values comma-separated */
void print_matrix(double **matrix, int rows, int cols);

/* Frees a 2D matrix (frees the contiguous data block first, then the row pointers) */
void free_matrix(double **matrix);

/* Verifies if memory allocation was successful, exits on failure */
void check_allocation(void *ptr);

/* Executes sym, ddg, or norm based on the goal string; manages calculation flow */
double** execute_goal(double **points, char *goal, int N, int d);

/* Iteratively optimizes H until convergence (epsilon=1e-4) or max_iter=300 */
double** symnmf_algorithm(double** H, double** W, int N, int k);

#endif