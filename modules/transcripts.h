#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "helpers.h" //Memory management, indexing and sorting
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>


int factorial(int n);
double D_KL(double* P1, double* P2, int prob_len);
void permutation_from_rank(int rank, int len, int result[]);
int permutation_rank(int arr[], int len);
double* delay_vectors(const double* ts, int L, int tau, int p);
int* delay_vectors_to_symbols(double* dv, int rows, int cols, int F);
double* partitions(const double* ts, int T, int p);
double permutation_entropy(int* symbols, int p, int len);
int* get_transcript2(int* symbol1, int* symbol2, int size);
int* get_transcripts(int* symbols1, int* symbols2, int rows, int cols);
int find_order_class(int* transcript, int length);
int* find_order_class_array(int* transcripts, int n_transcripts, int p);
double calc_JS_C(int* A, int* B, int subarray_len, int num_subarrays);
double* transcript_mutual_info(double* ts1, double* ts2, int T, int Lambda, int tau, int p);
int* find_unique_degrees(double* ts, int T, int tau, int p);
double coupling_complexity(double* time_series, int N, int T, int tau, int p);
double coupling_complexity_symbols(int* symbol_series, int N, int L, int p);
