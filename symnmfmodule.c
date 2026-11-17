#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

static double** py_list_to_matrix(PyObject *list_py, int rows, int cols);
static PyObject* matrix_to_py_list(double** matrix, int rows, int cols);
static PyObject* get_matrix_by_goal(PyObject *args, char *goal);
static PyObject* get_sym_py(PyObject *self, PyObject *args);
static PyObject* get_ddg_py(PyObject *self, PyObject *args);
static PyObject* get_norm_py(PyObject *self, PyObject *args);
static PyObject* get_symnmf_py(PyObject *self, PyObject *args);
PyMODINIT_FUNC PyInit_symnmfmodule(void);


/**
 * Array of functions to be registered with the Python interpreter.
 */
static PyMethodDef symnmfMethods[] = {
    {"sym", (PyCFunction)get_sym_py, METH_VARARGS, PyDoc_STR("Calculate Similarity Matrix")},
    {"ddg", (PyCFunction)get_ddg_py, METH_VARARGS, PyDoc_STR("Calculate Diagonal Degree Matrix")},
    {"norm", (PyCFunction)get_norm_py, METH_VARARGS, PyDoc_STR("Calculate Normalized Similarity Matrix")},
    {"symnmf", (PyCFunction)get_symnmf_py, METH_VARARGS, PyDoc_STR("Run SPNMF algorithm")},
    {NULL, NULL, 0, NULL}
};
/**
 * Module definition structure.
 */
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    NULL,
    -1,
    symnmfMethods
};


/**
Converts a Python list of lists into a C double** matrix.
Parameter list_py: PyObject representing the nested Python list.
Parameter rows: Number of rows in the matrix.
Parameter cols: Number of columns in the matrix.
Return: A contiguous double** matrix or NULL if memory allocation or conversion fails.
 */
static double** py_list_to_matrix(PyObject *list_py, int rows, int cols){
    int i, j;
    PyObject *row_py;
    double** matrix = allocate_matrix(rows, cols);
    if (!matrix){
        return NULL;
    }
    for (i = 0; i < rows; i++) {
        /* Retrieve the inner list (row) */
        row_py = PyList_GetItem(list_py, i);
        for (j = 0; j < cols; j++) {
            /* Convert each Python Float object to a C double */
            matrix[i][j] = PyFloat_AsDouble(PyList_GetItem(row_py, j));
            /* Check if the conversion to double failed */
            if (PyErr_Occurred()) { 
                free_matrix(matrix);
                return NULL;
            }
        }
    }
    return matrix;
}

/**
Converts a C double** matrix into a Python list of lists.
Parameter matrix: The C matrix to convert.
Parameter rows: Number of rows.
Parameter cols: Number of columns.
Return: A new Python list object (list of lists) containing the matrix values.
 */
static PyObject* matrix_to_py_list(double** matrix, int rows, int cols) {
    int i, j;
    PyObject *outer_list = PyList_New(rows);
    for (i = 0; i < rows; i++) {
        PyObject *inner_list = PyList_New(cols);
        for (j = 0; j < cols; j++) {
            PyList_SetItem(inner_list, j, PyFloat_FromDouble(matrix[i][j]));
        }
        PyList_SetItem(outer_list, i, inner_list);
    }
    return outer_list;
}

/**
Generic wrapper for sym, ddg, and norm goals.
Parameter args: Python argument tuple containing the input points.
Parameter goal: C-string defining the operation ("sym", "ddg", or "norm").
Return: A Python list of lists containing the result matrix.
 */
static PyObject* get_matrix_by_goal(PyObject *args, char *goal) {
    PyObject *points_list_py, *result;
    int N, d;
    double **points, **res;
    /* Parse arguments from Python */
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &points_list_py)){
        return NULL;
    }
    /* Determine dimensions */
    N = PyList_Size(points_list_py);
    if (N == 0) {
        PyErr_SetString(PyExc_ValueError, "An Error Has Occurred\n");
        return NULL;
    }
    d = PyList_Size(PyList_GetItem(points_list_py, 0));
    /* Convert to C, Compute, Convert back to Python */
    points = py_list_to_matrix(points_list_py, N, d);
    if (points == NULL) {
        return NULL;
    }
    res = execute_goal(points, goal, N, d);
    if (res == NULL) {
        free_matrix(points);
        return NULL;
    }
    result = matrix_to_py_list(res, N, N);
    free_matrix(points);
    free_matrix(res);
    return result;
}

/**
Python interface for the Similarity Matrix (sym) calculation.
This function acts as a wrapper that routes Python arguments to 
the generic matrix computation logic with the "sym" goal.
Parameter self: Pointer to the module object (unused here).
Parameter args: Python tuple containing the input data points (list of lists).
Return: A Python list of lists representing the N x N Similarity Matrix, 
or NULL if an error occurs.
 */
static PyObject* get_sym_py(PyObject *self, PyObject *args){
    return get_matrix_by_goal(args, "sym");
}

/**
Python interface for the Diagonal Degree Matrix (ddg) calculation.
Routes the input data to the generic matrix computation logic using the "ddg" goal.
Parameter args: Python tuple containing the input points list.
Return: A Python list of lists representing the N x N diagonal matrix,
or NULL on failure.
 */
static PyObject* get_ddg_py(PyObject *self, PyObject *args){
    return get_matrix_by_goal(args, "ddg");
}

/**
Python interface for the Normalized Similarity Matrix (norm) calculation.
Routes the input data to the generic matrix computation logic 
using the "norm" goal.
Parameter args: Python tuple containing the input points list.
Return: A Python list of lists representing the N x N normalized matrix W,
or NULL on failure.
 */
static PyObject* get_norm_py(PyObject *self, PyObject *args){
    return get_matrix_by_goal(args, "norm");
}

/**
Python wrapper for the full SymNMF optimization algorithm.
Parameter args: Tuple containing initial H, matrix W, N, and k.
Return: The optimized matrix H as a Python list of lists.
 */
static PyObject* get_symnmf_py(PyObject *self, PyObject *args) {
    PyObject *W_py, *H_py, *result;
    int N, k;
    double **W, **H, **final_H;
    /* Parse arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!ii", &PyList_Type, &H_py, &PyList_Type, &W_py, &N, &k)) {
        return NULL;
    }
    /* Convert Python lists to C matrices */
    W = py_list_to_matrix(W_py, N, N);
    if (W == NULL){
        return NULL;}
    H = py_list_to_matrix(H_py, N, k);
    if (H == NULL){
        free_matrix(W);
        return NULL;}
    /* Call the core algorithm logic (from symnmf.c) */
    final_H = symnmf_algorithm(H, W, N, k);
    /* Convert the result back to Python */
    result = matrix_to_py_list(final_H, N, k);
    /* Free memory */
    free_matrix(W);
    free_matrix(final_H);
    return result;
}


/**
Module initialization function (Entry Point).
This function is automatically called by the Python interpreter 
during the 'import' statement. It initializes the module structure
and registers the defined methods (sym, ddg, norm, symnmf).
Return: A pointer to the newly created module object on success, 
or NULL if an error occurred during initialization.
 */
PyMODINIT_FUNC PyInit_symnmfmodule(void) {
    PyObject *m;
    m = PyModule_Create(&symnmfmodule);
    if (!m){
        return NULL;
    }
    return m;
}
