#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "helpers.h" //Memory management, indexing and sorting
#include "transcripts.h" //Memory management, indexing and sorting
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

// wrapper to create delay vectors from time series
static PyObject* delay_vectors_wrapper(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* input_array = Py_None;
    PyArrayObject* input_array_np;
    int tau, p;

    // Parse the input arguments
    
    static char* kwlist[] = {
        "data", "delay", "dim", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oii", kwlist, &input_array, &tau, &p)) {
        return NULL;
    }

    if (input_array == Py_None) {
        PyErr_SetString(PyExc_ValueError, "Data may not be None");
        Py_DECREF(input_array_np);
        return NULL;
    }

    // Convert the input object to a NumPy array
    input_array_np = (PyArrayObject*)PyArray_FROM_OTF(input_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (input_array_np == NULL) {
        return NULL;
    }

    // Check input array dimensions
    int ndim = PyArray_NDIM(input_array_np);
    if (ndim != 1) {
        PyErr_SetString(PyExc_ValueError, "Input array must be 1-dimensional");
        Py_DECREF(input_array_np);
        return NULL;
    }

    // Set L based on the length of the input array
    int T = PyArray_DIM(input_array_np, 0);

    if (T-tau*(p-1) < 0) {
        PyErr_SetString(PyExc_ValueError, "T-tau*(p-1) < 0. Try choosing smaller delay or dim");
        Py_DECREF(input_array_np);
        return NULL;
    }

    // Get data pointer from the NumPy array
    double* ts = (double*)PyArray_DATA(input_array_np);

    // Call the C function
    //double** result = delay_vectors(ts, L, tau, p);
    double* result = delay_vectors(ts, T, tau, p);

    if (result == NULL) {
        // Handle error in delay_vectors
        Py_DECREF(input_array_np);
        return NULL;
    }

    // Convert the C result to a NumPy array
    npy_intp dims[2] = {T - tau * (p - 1), p};
    PyObject* output_array = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, result);
    //PyArray_ENABLEFLAGS((PyArrayObject*)output_array, NPY_ARRAY_OWNDATA);
    // Attach the free function to the NumPy array's destructor
    PyObject *capsule = PyCapsule_New((void*)result, NULL, capsule_cleanup);
    PyArray_SetBaseObject((PyArrayObject*)output_array, capsule);
    
    // Clean up
    Py_DECREF(input_array_np);
    
    return output_array;
}

// wrapper to find transcripts between two time series
static PyObject* transcripts_wrapper(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* input_array1;
    PyObject* input_array2;
    PyArrayObject* input_array_np1;
    PyArrayObject* input_array_np2;
    int tau, p;

    static char* kwlist[] = {
        "data1", "data2", "delay", "dim", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOii", kwlist, &input_array1, &input_array2, &tau, &p)) {
        return NULL;
    }

    // Convert the input object to a NumPy array
    input_array_np1 = (PyArrayObject*)PyArray_FROM_OTF(input_array1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (input_array_np1 == NULL) {
        return NULL;
    }
    input_array_np2 = (PyArrayObject*)PyArray_FROM_OTF(input_array2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (input_array_np2 == NULL) {
        return NULL;
    }

    // Check input array dimensions
    
    if ((PyArray_NDIM(input_array_np1) != 1) || (PyArray_NDIM(input_array_np2) != 1)) {
        PyErr_SetString(PyExc_ValueError, "Input arrays are not 1-dimensional.");
        Py_DECREF(input_array_np1);
        Py_DECREF(input_array_np2);
        return NULL;
    }

    // Set L based on the length of the input array
    int T = PyArray_DIM(input_array_np1, 0);
    
    if (T-tau*(p-1) < 0) {
        PyErr_SetString(PyExc_ValueError, "T-tau*(p-1) < 0. Try choosing smaller delay or dim");
        Py_DECREF(input_array_np1);
        Py_DECREF(input_array_np2);
        return NULL;
    }

    if (T != PyArray_DIM(input_array_np2, 0)) {
        PyErr_SetString(PyExc_ValueError, "Arrays must have the same length");
        Py_DECREF(input_array_np1);
        Py_DECREF(input_array_np2);
        return NULL;
    }

    // Get data pointer from the NumPy array
    double* ts1 = (double*)PyArray_DATA(input_array_np1);
    double* ts2 = (double*)PyArray_DATA(input_array_np2);

    // Call the C function
    double* dv1 = delay_vectors(ts1, T, tau, p);
    double* dv2 = delay_vectors(ts2, T, tau, p);
    int* symbols1 = delay_vectors_to_symbols(dv1, T-tau*(p-1), p, 0);
    int* symbols2 = delay_vectors_to_symbols(dv2, T-tau*(p-1), p, 0);
    int* result = get_transcripts(symbols1, symbols2, T-tau*(p-1), p);

    long* result_long = malloc(sizeof(long) * (T-tau*(p-1)));
    for (int i=0; i<T-tau*(p-1); i++) {

       result_long[i] = (long)result[i]; 

    }
    free(result);

    if (result == NULL) {
        // Handle error in delay_vectors
        Py_DECREF(input_array_np1);
        Py_DECREF(input_array_np2);
        return NULL;
    }

    // Convert the C result to a NumPy array
    npy_intp dims[2] = {T - tau * (p - 1), p};
    PyObject* output_array = PyArray_SimpleNewFromData(2, dims, NPY_INT64, result_long);
    //PyArray_ENABLEFLAGS((PyArrayObject*)output_array, NPY_ARRAY_OWNDATA);
    // Attach the free function to the NumPy array's destructor
    PyObject *capsule = PyCapsule_New((void*)result_long, NULL, capsule_cleanup);
    PyArray_SetBaseObject((PyArrayObject*)output_array, capsule);
    
    // Clean up
   
    free(dv1);
    free(dv2);
    free(symbols1);
    free(symbols2);
    Py_DECREF(input_array_np1);
    Py_DECREF(input_array_np2);

    return output_array;
}

// wrapper for calculating permutation entropy of an ordinal pattern sequence
static PyObject* PE_wrapper(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* input_array;
    PyArrayObject* input_array_np;
    int tau, p;

    // Parse the input arguments

    static char* kwlist[] = {
        "data", "delay", "dim", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oii", kwlist, &input_array, &tau, &p)) {
        return NULL;
    }

    // Convert the input object to a NumPy array
    input_array_np = (PyArrayObject*)PyArray_FROM_OTF(input_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (input_array_np == NULL) {
        return NULL;
    }

    if ((PyArray_NDIM(input_array_np) != 1)) {
        PyErr_SetString(PyExc_ValueError, "Input array is not 1-dimensional.");
        Py_DECREF(input_array_np);
        return NULL;
    }

    // Set L based on the length of the input array
    int T = PyArray_DIM(input_array_np, 0);
    
    if (T-tau*(p-1) < 0) {
        PyErr_SetString(PyExc_ValueError, "T-tau*(p-1) < 0. Try choosing smaller delay or dim");
        Py_DECREF(input_array_np);
        return NULL;
    }
    
    // Get data pointer from the NumPy array
    double* ts = (double*)PyArray_DATA(input_array_np);

    // Call the C function
    double* dv = delay_vectors(ts, T, tau, p);
    int* symbols = delay_vectors_to_symbols(dv, T-tau*(p-1), p, 0);
    double result = permutation_entropy(symbols, p, T-tau*(p-1));

    // Clean up
    free(dv);
    free(symbols);
    Py_DECREF(input_array_np);
 
    return Py_BuildValue("d", result);
}

// wrapper to calculate the Jensen-Shannon divergence for order classes between two time series
static PyObject* JS_wrapper(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* input_array1;
    PyObject* input_array2;
    PyArrayObject* input_array_np1;
    PyArrayObject* input_array_np2;
    int tau, p;

    // Parse the input arguments

    static char* kwlist[] = {
        "data1", "data2", "delay", "dim", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOii", kwlist, &input_array1, &input_array2, &tau, &p)) {
        return NULL;
    }

    // Convert the input object to a NumPy array
    input_array_np1 = (PyArrayObject*)PyArray_FROM_OTF(input_array1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (input_array_np1 == NULL) {
        return NULL;
    }
    input_array_np2 = (PyArrayObject*)PyArray_FROM_OTF(input_array2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (input_array_np2 == NULL) {
        return NULL;
    }

    // Check input array dimensions
    if ((PyArray_NDIM(input_array_np1) != 1) || (PyArray_NDIM(input_array_np2) != 1)) {
        PyErr_SetString(PyExc_ValueError, "Input arrays are not 1-dimensional.");
        Py_DECREF(input_array_np1);
        Py_DECREF(input_array_np2);
        return NULL;
    }

    // Set L based on the length of the input array
    int T = PyArray_DIM(input_array_np1, 0);
    
    if (T-tau*(p-1) < 0) {
        PyErr_SetString(PyExc_ValueError, "T-tau*(p-1) < 0. Try choosing smaller delay or dim");
        Py_DECREF(input_array_np1);
        Py_DECREF(input_array_np2);
        return NULL;
    }

    if (T != PyArray_DIM(input_array_np2, 0)) {
        PyErr_SetString(PyExc_ValueError, "Arrays must have the same length");
        Py_DECREF(input_array_np1);
        Py_DECREF(input_array_np2);
    }

    // Get data pointer from the NumPy array
    double* ts1 = (double*)PyArray_DATA(input_array_np1);
    double* ts2 = (double*)PyArray_DATA(input_array_np2);

    // Call the C function
    double* dv1 = delay_vectors(ts1, T, tau, p);
    double* dv2 = delay_vectors(ts2, T, tau, p);

    int* symbols1 = delay_vectors_to_symbols(dv1, T-tau*(p-1), p, 0);
    int* symbols2 = delay_vectors_to_symbols(dv2, T-tau*(p-1), p, 0);

    double result = calc_JS_C(symbols1, symbols2, p, T-tau*(p-1));

    // Clean up
    free(dv1);
    free(dv2);
    free(symbols1);
    free(symbols2);
    Py_DECREF(input_array_np1);
    Py_DECREF(input_array_np2);
 
    return Py_BuildValue("d", result);
}

//wrapper to calculate Coupling Complexity between two time series
static PyObject* CC_wrapper(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* input_array;
    PyArrayObject* input_array_np;
    int tau, p;

    // Parse the input arguments
    
    static char* kwlist[] = {
        "data", "delay", "dim", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oii", kwlist, &input_array, &tau, &p)) {
        return NULL;
    }

    // Convert the input object to a NumPy array
    input_array_np = (PyArrayObject*)PyArray_FROM_OTF(input_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (input_array_np == NULL) {
        return NULL;
    }

    // Check input array dimensions
    if (PyArray_NDIM(input_array_np) < 2) {
        PyErr_SetString(PyExc_ValueError, "Input array must be at least 2-dimensional.");
        Py_DECREF(input_array_np);
        return NULL;
    }

    npy_intp T = PyArray_DIM(input_array_np, 0);
    npy_intp N = PyArray_DIM(input_array_np, 1);

    if (T - tau * (p - 1) < 0) {
        PyErr_SetString(PyExc_ValueError, "T - tau*(p-1) < 0. Try choosing smaller delay or dimension.");
        Py_DECREF(input_array_np);
        return NULL;
    }

    // Allocate memory for the flattened array (column-wise)
    double* flattened = malloc(sizeof(double) * T * N);
    if (flattened == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for flattened array.");
        Py_DECREF(input_array_np);
        return NULL;
    }

    // Fill the flattened array column-wise
    for (npy_intp col = 0; col < N; col++) {
        for (npy_intp row = 0; row < T; row++) {
            double val = *(double*)PyArray_GETPTR2(input_array_np, row, col);
            flattened[col * T + row] = val;
        }
    }

    // Call the function (assuming the function can now handle T, N, flattened...)
    double result = coupling_complexity(flattened, N, T, tau, p);

    // Clean up
    free(flattened);
    Py_DECREF(input_array_np);

    return Py_BuildValue("d", result);
}

//wrapper to calculate Coupling Complexity between two symbol series 
static PyObject* CC_symb_wrapper(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* input_array;
    PyArrayObject* input_array_np;

    // Parse the input arguments
    static char* kwlist[] = {
        "symbols", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &input_array)) {
        return NULL;
    }

    // Convert the input object to a NumPy array
    input_array_np = (PyArrayObject*)PyArray_FROM_OTF(input_array, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (input_array_np == NULL) {
        return NULL;
    }

    // Check input array dimensions
    if (PyArray_NDIM(input_array_np) != 3) {
        PyErr_SetString(PyExc_ValueError, "Input array must be 3-dimensional (N, L, p).");
        Py_DECREF(input_array_np);
        return NULL;
    }

    npy_intp N = PyArray_DIM(input_array_np, 0);
    npy_intp L = PyArray_DIM(input_array_np, 1);
    npy_intp p = PyArray_DIM(input_array_np, 2);

    npy_intp total_size = N * L * p;

    // Allocate memory for the flattened array
    int* flattened = malloc(sizeof(int) * total_size);
    if (flattened == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for flattened array.");
        Py_DECREF(input_array_np);
        return NULL;
    }

    // Fill the flattened array: iterate over L, then N, then p
    for (npy_intp n = 0; n < N; n++) {
        for (npy_intp l = 0; l < L; l++) {
            for (npy_intp pi = 0; pi < p; pi++) {
                int val = *(int*)PyArray_GETPTR3(input_array_np, n, l, pi);
                flattened[(n * L + l) * p + pi] = val;
            }
        }
    }

    // Call the function (pass flattened, N, L, tau, p)
    double result = coupling_complexity_symbols(flattened, N, L, p);

    // Clean up
    free(flattened);
    Py_DECREF(input_array_np);

    return Py_BuildValue("d", result);
}

//wrapper to calculate ordinal pattern series from a time series
static PyObject* find_symbols_wrapper(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *input_array;
    PyArrayObject *input_array_np;
    int tau, p;

    // Parse the input arguments
    static char* kwlist[] = {
        "data", "delay", "dim", NULL
    };
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oii", kwlist, &input_array, &tau, &p)) {
        return NULL;
    }

    // Convert the input object to a NumPy array
    input_array_np = (PyArrayObject *)PyArray_FROM_OTF(input_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (input_array_np == NULL) {
        return NULL;
    }

    // Check input array dimensions
    int ndim = PyArray_NDIM(input_array_np);
    if (ndim != 1) {
        PyErr_SetString(PyExc_ValueError, "Input array must be 1-dimensional.");
        Py_DECREF(input_array_np);
        return NULL;
    }

    // Set L based on the length of the input array
    int T = PyArray_DIM(input_array_np, 0);

    if (T-tau*(p-1) < 0) {
        PyErr_SetString(PyExc_ValueError, "T-tau*(p-1) < 0. Try choosing smaller delay or dim");
        Py_DECREF(input_array_np);
        return NULL;
    }

    // Get data pointer from the NumPy array
    double *ts = (double *)PyArray_DATA(input_array_np);

    // Call the C function
    double* dv = delay_vectors(ts, T, tau, p);
    int* result = delay_vectors_to_symbols(dv, T-tau*(p-1), p, 0);
    
    long* result_long = malloc(sizeof(long)*(T-tau*(p-1))*p);

    for (int i=0; i<(T-tau*(p-1))*p; i++){
       result_long[i] = (long)result[i]; 
    }
    
    free(result);

    // Convert the C result to a NumPy array
    npy_intp dims[2] = {T-tau*(p-1), p};
    PyObject* output_array = PyArray_SimpleNewFromData(2, dims, NPY_INT64, result_long);

    // Attach the free function to the NumPy array's destructor
    PyObject *capsule = PyCapsule_New((void*)result_long, NULL, capsule_cleanup);
    PyArray_SetBaseObject((PyArrayObject*)output_array, capsule);
    
    // Clean up
    free(dv);
    Py_DECREF(input_array_np);

    return output_array;
}

// wrapper to calulate order classes from time series 
static PyObject* oc_wrapper(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *input_array;
    PyArrayObject *input_array_np;
    int tau, p;

    // Parse the input arguments
    
    static char* kwlist[] = {
        "data", "delay", "dim", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oii", kwlist, &input_array, &tau, &p)) {
        return NULL;
    }

    // Convert the input object to a NumPy array
    input_array_np = (PyArrayObject *)PyArray_FROM_OTF(input_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (input_array_np == NULL) {
        return NULL;
    }

    // Check input array dimensions
    int ndim = PyArray_NDIM(input_array_np);
    if (ndim != 1) {
        PyErr_SetString(PyExc_ValueError, "Input array must be 1-dimensional.");
        Py_DECREF(input_array_np);
        return NULL;
    }

    // Set L based on the length of the input array
    int T = PyArray_DIM(input_array_np, 0);
    
    if (T-tau*(p-1) < 0) {
        PyErr_SetString(PyExc_ValueError, "T-tau*(p-1) < 0. Try choosing smaller delay or dim");
        Py_DECREF(input_array_np);
        return NULL;
    }

    // Get data pointer from the NumPy array
    double *ts = (double *)PyArray_DATA(input_array_np);

    // Call the C function
    double* dv = delay_vectors(ts, T, tau, p);
    int* symbols = delay_vectors_to_symbols(dv, T-tau*(p-1), p, 0);
    int* result = find_order_class_array(symbols, T-tau*(p-1), p);

    long* result_long = malloc(sizeof(long) * (T-tau*(p-1)));

    for (int i=0; i<T-tau*(p-1); i++) {
        result_long[i] = (long)result[i];
    }

    free(result);

    // Convert the C result to a NumPy array
    npy_intp dims[1] = {T-tau*(p-1)};
    PyObject* output_array = PyArray_SimpleNewFromData(1, dims, NPY_INT64, result_long);
    
    //PyArray_ENABLEFLAGS(output_array, NPY_ARRAY_OWNDATA);
    PyObject *capsule = PyCapsule_New((void*)result_long, NULL, capsule_cleanup);
    PyArray_SetBaseObject((PyArrayObject*)output_array, capsule);

    //Clean up
    free(dv);
    free(symbols);
    Py_DECREF(input_array_np);

    return output_array;
}

// wrapper to calculate order classes from ordinal pattern sequence
static PyObject* oc_wrapper_transcripts(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *input_array;
    PyArrayObject *input_array_np;

    // Parse the input arguments
    static char* kwlist[] = {
        "symbols",  NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &input_array)) {
        return NULL;
    }

    // Convert the input object to a NumPy array
    input_array_np = (PyArrayObject *)PyArray_FROM_OTF(input_array, NPY_INT, NPY_ARRAY_FORCECAST);
    if (input_array_np == NULL) {
        return NULL;
    }

    // Check input array dimensions
    int ndim = PyArray_NDIM(input_array_np);
    if (ndim != 2) {
        PyErr_SetString(PyExc_ValueError, "Input array must be 2-dimensional.");
        Py_DECREF(input_array_np);
        return NULL;
    }

    int *ts = (int *)PyArray_DATA(input_array_np);

    // Set L based on the length of the input array
    int L = PyArray_DIM(input_array_np, 0);
    int cols = PyArray_DIM(input_array_np, 1);
    long* ocs = malloc(L * sizeof(long));
    //return Py_BuildValue("i", cols);
    int* symbol = malloc(cols * sizeof(int));
    for (int i=0; i < L; ++i) {
        
        for (int j=0; j < cols; ++j) {
            symbol[j] = ts[i*cols+j];
        }
        ocs[i] = (long)find_order_class(symbol, cols);
    }
    free(symbol);

    // Convert the C result to a NumPy array
    npy_intp dims[1] = {L};
    PyObject* output_array = PyArray_SimpleNewFromData(1, dims, NPY_INT64, ocs);
    
    //PyArray_ENABLEFLAGS(output_array, NPY_ARRAY_OWNDATA);
    PyObject *capsule = PyCapsule_New((void*)ocs, NULL, capsule_cleanup);
    PyArray_SetBaseObject((PyArrayObject*)output_array, capsule);
    //Clean up
   
    Py_DECREF(input_array_np);

    return output_array;
}

//wrapper to calculate transcript mutual information between two time series
static PyObject* TMI_wrapper(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *input_array1;
    PyArrayObject *input_array_np1;
    PyObject *input_array2;
    PyArrayObject *input_array_np2;
    int Lambda, tau, p;

    // Parse the input arguments
    static char* kwlist[] = {
        "data1", "data2","symbol_delay", "delay", "dim",  NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOiii", kwlist, &input_array1, &input_array2, &Lambda, &tau, &p)) {
        return NULL;
    }

    // Convert the input object to a NumPy array
    input_array_np1 = (PyArrayObject *)PyArray_FROM_OTF(input_array1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (input_array_np1 == NULL) {
        return NULL;
    }
    input_array_np2 = (PyArrayObject *)PyArray_FROM_OTF(input_array2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (input_array_np2 == NULL) {
        return NULL;
    }

    // Check input array dimensions
    int ndim1 = PyArray_NDIM(input_array_np1);
    int ndim2 = PyArray_NDIM(input_array_np2);
    if ((ndim1 != 1) || (ndim2 != 1)) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be 1-dimensional.");
        Py_DECREF(input_array_np1);
        Py_DECREF(input_array_np2);
        return NULL;
    }

    int T1 = PyArray_DIM(input_array_np1, 0);
    int T2 = PyArray_DIM(input_array_np2, 0);
    if (T1 != T2) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must have the same length.");
        Py_DECREF(input_array_np1);
        Py_DECREF(input_array_np2);
        return NULL;
    }
    int T = T1;

    double *ts1 = (double *)PyArray_DATA(input_array_np1);
    double *ts2 = (double *)PyArray_DATA(input_array_np2);

    double* tmi = transcript_mutual_info(ts1, ts2, T, Lambda, tau, p);
    // Convert the C result to a NumPy array
    npy_intp dims[1] = {2};
    PyObject* output_array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, tmi);
    
    //PyArray_ENABLEFLAGS(output_array, NPY_ARRAY_OWNDATA);
    PyObject *capsule = PyCapsule_New((void*)tmi, NULL, capsule_cleanup);
    PyArray_SetBaseObject((PyArrayObject*)output_array, capsule);
    //Clean up
    Py_DECREF(input_array_np1);
    Py_DECREF(input_array_np2);

    return output_array;
}

// wrapper to find order class of an ordinal pattern
static PyObject* find_order_class_wrapper(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* input_array;
    PyArrayObject* input_array_np;

    // Parse the input arguments
    static char* kwlist[] = {
        "symbols",  NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &input_array)) {
        return NULL;
    }

    // Convert the input object to a NumPy array
    input_array_np = (PyArrayObject*)PyArray_FROM_OTF(input_array, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (input_array_np == NULL) {
        return NULL;
    }

    // Check input array dimensions
    int ndim = PyArray_NDIM(input_array_np);
    
    if (ndim != 1) {
        PyErr_SetString(PyExc_ValueError, "Input array must be 1-dimensional.");
        Py_DECREF(input_array_np);
        return NULL;
    }

    // Set L based on the length of the input array
    int p = PyArray_DIM(input_array_np, 0);

    // Call the C function
    int *symbol = (int *)PyArray_DATA(input_array_np);
    int result = find_order_class(symbol,p);

    if (result == NULL) {
        // Handle error in delay_vectors
        Py_DECREF(input_array_np);
        return NULL;
    }
    
    // Clean up
    Py_DECREF(input_array_np);
    
    return Py_BuildValue("i", result);
}

static PyMethodDef methods[] = {
    {"delay_vectors", (PyCFunction)delay_vectors_wrapper, METH_VARARGS | METH_KEYWORDS, 
     "Generate delay vectors from a time series.\n\n"
     "Args:\n"
     "    data (numpy.ndarray, list): 1D array of time series data.\n"
     "    tau (int): Embedding delay for constructing vectors.\n"
     "    p (int): Embedding dimension for constructing vectors.\n"
     "Returns:\n"
     "    numpy.ndarray: 2D array of delay vectors."},

    {"get_transcripts", (PyCFunction)transcripts_wrapper, METH_VARARGS | METH_KEYWORDS, 
     "Generate symbol series from a time series.\n\n"
     "Args:\n"
     "    Time series 1 (numpy.ndarray, list): Source time series.\n"
     "    Time series 2 (numpy.ndarray, list): Target time series.\n"
     "    tau (int): Embedding delay for constructing vectors.\n"
     "    p (int): Embedding dimension for constructing vectors.\n"
     "Returns:\n"
     "    numpy.ndarray (int32): Array of transcripts."},

    {"permutation_entropy", (PyCFunction)PE_wrapper, METH_VARARGS | METH_KEYWORDS,
     "Generate symbol series from a time series.\n\n"
     "Args:\n"
     "    Time series (numpy.ndarray, list): 1D array of time series data.\n"
     "    tau (int): Delay step for constructing vectors.\n"
     "    p (int): Delay step for constructing vectors.\n"
     "Returns:\n"
     "    numpy.ndarray (int32): Array of symbols."},

    {"find_symbols", (PyCFunction)find_symbols_wrapper, METH_VARARGS | METH_KEYWORDS, 
     "Generate symbol series from a time series.\n\n"
     "Args:\n"
     "    Time series (numpy.ndarray, list): 1D array of time series data.\n"
     "    tau (int): Delay step for constructing vectors.\n"
     "    p (int): Delay step for constructing vectors.\n"
     "Returns:\n"
     "    numpy.ndarray (int32): Array of symbols."},

    {"find_order_classes", (PyCFunction)oc_wrapper, METH_VARARGS | METH_KEYWORDS, 
     "Generate symbol series from a time series.\n\n"
     "Args:\n"
     "    Time series (numpy.ndarray, list): 1D array of time series data.\n"
     "    tau (int): Delay step for constructing vectors.\n"
     "    p (int): Delay step for constructing vectors.\n"
     "Returns:\n"
     "    numpy.ndarray (int32): Array of symbols."},

    {"coupling_complexity", (PyCFunction)CC_wrapper, METH_VARARGS | METH_KEYWORDS,
     "Generate symbol series from a time series.\n\n"
     "Args:\n"
     "    Time series (numpy.ndarray): 1D array of time series data.\n"
     "    tau (int): Embedding delay for constructing vectors.\n"
     "    p (int): Embedding dimension for constructing vectors.\n"
     "Returns:\n"
     "    numpy.ndarray (int32): Array of symbols."},

    {"transcript_mi", (PyCFunction)TMI_wrapper, METH_VARARGS | METH_KEYWORDS, 
     "Generate symbol series from a time series.\n\n"
     "Args:\n"
     "    Time series (numpy.ndarray): 1D array of time series data.\n"
     "    tau (int): Delay step for constructing vectors.\n"
     "    p (int): Delay step for constructing vectors.\n"
     "Returns:\n"
     "    numpy.ndarray (int32): Array of symbols."},

    {"coupling_complexity_symbols", (PyCFunction)CC_symb_wrapper, METH_VARARGS | METH_KEYWORDS, 
     "Generate symbol series from a time series.\n\n"
     "Args:\n"
     "    Time series (numpy.ndarray): 1D array of time series data.\n"
     "    tau (int): Delay step for constructing vectors.\n"
     "    p (int): Delay step for constructing vectors.\n"
     "Returns:\n"
     "    numpy.ndarray (int32): Array of symbols."},

    {"ocs_from_symbol_series", (PyCFunction)oc_wrapper_transcripts, METH_VARARGS | METH_KEYWORDS, 
     "Generate symbol series from a time series.\n\n"
     "Args:\n"
     "    Symbols (numpy.ndarray): 2D array of ordinal patterns.\n"
     "    tau (int): Delay step for constructing vectors.\n"
     "    p (int): Delay step for constructing vectors.\n"
     "Returns:\n"
     "    numpy.ndarray (int32): Array of symbols."},

    {"find_order_class", (PyCFunction)find_order_class_wrapper, METH_VARARGS | METH_KEYWORDS, 
     "Generate symbol series from a time series.\n\n"
     "Args:\n"
     "    data (numpy.ndarray): 2D array of time series data.\n"
     "    tau (int): Delay step for constructing vectors.\n"
     "    p (int): Delay step for constructing vectors.\n"
     "Returns:\n"
     "    numpy.ndarray (int32): Array of symbols."},

    {"JS_C", (PyCFunction)JS_wrapper, METH_VARARGS | METH_KEYWORDS, "Calculate Jensen-Shannon divergence between two time series.\n\n"
     "Args:\n"
     "    ts1 (numpy.ndarray): 2D array of time series data.\n"
     "    ts2 (numpy.ndarray): 2D array of time series data.\n"
     "    tau (int): Delay step for constructing vectors.\n"
     "    p (int): Delay step for constructing vectors.\n"
     "Returns:\n"
     "    float: Jensen-Shannon divergence" },
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "transcripts",  // Module name
    NULL,
    -1,
    methods};

PyMODINIT_FUNC PyInit_transcripts(void) {
    import_array();  // Initialize NumPy API

    return PyModule_Create(&module);
}
