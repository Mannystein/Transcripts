#include "uthash.h" 
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <Python.h>


void free_memory_2d(double** v_i_list, int rows);
void free_contiguous_int(int** arr);
void free_contiguous_double(double** arr);
void capsule_cleanup(PyObject *capsule);
void argsort(int* indices, double* values, int size);
int is_increasing_sequence(int* arr, int length);
int kendall_distance(int* a, int* b, int p);
int cayley_distance(int* a, int* b, int p);
double joint_entropy_ranks(int* ranks, int num_pairs, int L);


