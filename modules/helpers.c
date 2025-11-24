#include "uthash.h" 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <Python.h>
//#include <numpy/arrayobject.h>

typedef struct {
    int* key;          // pointer to the joint rank vector
    int key_len;       // length of the key (N-1)
    int count;         // occurrence count
    UT_hash_handle hh; // Makes this struct hashable
} JointEntry;

//Not used, but may be useful if different 2d allocation is needed
void free_memory_2d(double** v_i_list, int rows) {
	// Free the allocated memory
    for (int i = 0; i < rows; ++i) {
        free(v_i_list[i]);
    }
    free(v_i_list);
}


//destructor for 2d arrays allocated in the code
void free_contiguous_int(int** arr) {
    free(arr[0]);
    free(arr);
}

//destructor for 2d arrays allocated in the code
void free_contiguous_double(double** arr) {
    free(arr[0]);
    free(arr);
}


//Destructor function for numpy
void capsule_cleanup(PyObject *capsule) {
    void *memory = PyCapsule_GetPointer(capsule, NULL);
    free(memory);
}

////Argsort as in NumPy
//void argsort(int* indices, double* values, int size) {
//    // Initialize indices array
//    for (int i = 0; i < size; ++i) {
//        indices[i] = i;
//    }
//
//    // Perform argsort
//    for (int i = 0; i < size; ++i) {
//        for (int j = i + 1; j < size; ++j) {
//            if (values[indices[i]] > values[indices[j]]) {
//                // Swap indices if necessary
//                int temp = indices[i];
//                indices[i] = indices[j];
//                indices[j] = temp;
//            }
//        }
//    }
//}

// Stable argsort matching numpy's mergesort behavior
void argsort(int* indices, const double* values, int size) {
    // initialize indices
    for (int i = 0; i < size; ++i)
        indices[i] = i;

    // stable insertion sort (O(n^2) but stable and simple)
    for (int i = 1; i < size; ++i) {
        int key = indices[i];
        double key_val = values[key];
        int j = i - 1;

        while (j >= 0 && values[indices[j]] > key_val) {
            indices[j + 1] = indices[j];
            j--;
        }
        // preserves order for ties (stable)
        indices[j + 1] = key;
    }
}

//checks if given array is a sequence of 1,...,N
int is_increasing_sequence(int* arr, int length) {
    for (int i = 0; i < length - 1; ++i) {
        if (arr[i] >= arr[i + 1]) {
            return 0;  // not strictly increasing
        }
    }
    return 1;  // strictly increasing
}

int merge_count(int* arr, int* temp, int left, int mid, int right) {
    int i = left;      // index for left subarray
    int j = mid + 1;   // index for right subarray
    int k = left;      // index for temp
    int inv_count = 0;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
            inv_count += (mid - i + 1);  // count inversions
        }
    }

    // copy remaining elements
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];

    // copy back to arr
    for (i = left; i <= right; i++) {
        arr[i] = temp[i];
    }

    return inv_count;
}

// recursive merge sort + inversion count
int sort_count(int* arr, int* temp, int left, int right) {
    int inv_count = 0;
    if (left < right) {
        int mid = (left + right) / 2;
        inv_count += sort_count(arr, temp, left, mid);
        inv_count += sort_count(arr, temp, mid + 1, right);
        inv_count += merge_count(arr, temp, left, mid, right);
    }
    return inv_count;
}

// inversion count wrapper
int inversion_count(int* arr, int n) {
    int* temp = (int*)malloc(n * sizeof(int));
    int inv_count = sort_count(arr, temp, 0, n - 1);
    free(temp);
    return inv_count;
}

// Compute Kendall tau distance between two permutations
int kendall_distance(int* a, int* b, int p) {
    int* pos = (int*)malloc(p * sizeof(int));
    int* arr = (int*)malloc(p * sizeof(int));

    // Map values of a to positions
    for (int i = 0; i < p; i++) {
        pos[a[i]] = i;
    }

    // Translate b into array of indices according to a
    for (int i = 0; i < p; i++) {
        arr[i] = pos[b[i]];
    }

    int result = inversion_count(arr, p);

    free(pos);
    free(arr);

    return result;
}

// Compute cayley distance between two permutations
int cayley_distance(int* a, int* b, int p) {
    int* visited = calloc(p, sizeof(int)); 
    int* pos_a = malloc(p*sizeof(int));
    int cycles = 0;
    int j;

    // allocate array with -1, good for debugging
    for (int i = 0; i < p; i++) {
        pos_a[i] = -1;
    }

    // map a onto positions
    for (int i = 0; i < p; i++) {
        pos_a[a[i]] = i;  
    }

    // go through all cycles and count
    // if visited == True , cycles has already been gone through
    for (int i=0; i<p; ++i) {
        if (!visited[i]) {
            cycles++;
            j = i;

            while (!visited[j]) {
            visited[j] = 1;
            j = pos_a[b[j]];

            }
        }
    }

    free(visited);
    free(pos_a);

    return p - cycles;
}

/*
int compare_keys(const void* a, const void* b, size_t len) {
    return memcmp(a, b, len * sizeof(int));
}
*/

unsigned hash_fn(const void* key, size_t len) {
    const int* data = (const int*)key;
    unsigned hash = 5381;
    for (size_t i = 0; i < len; ++i)
        // hash = ((hash << 5) + hash) + data[i]; // hash * 33 + data[i]
        hash = ((hash << 5) + hash) + (data[i] ^ (i * 2654435761));
    return hash;
}


// convert list of ordinal patterns to hash table to simplify entropy computation
double joint_entropy_ranks(int* ranks, int num_pairs, int L) {
    JointEntry* entries = NULL;

    for (int t = 0; t < L; ++t) {
        int* key = malloc(num_pairs * sizeof(int));
        for (int i = 0; i < num_pairs; ++i) {
            key[i] = ranks[i * L + t];
        }

        JointEntry* entry = NULL;
        HASH_FIND(hh, entries, key, num_pairs * sizeof(int), entry);

        if (entry) {
            entry->count++;
            free(key); // key is duplicate; free it
        } else {
            entry = malloc(sizeof(JointEntry));
            entry->key = key;
            entry->key_len = num_pairs;
            entry->count = 1;
            HASH_ADD_KEYPTR(hh, entries, entry->key, num_pairs * sizeof(int), entry);
        }
    }

    double entropy = 0.0;
    JointEntry* e, *tmp;
    for (e = entries; e != NULL; e = e->hh.next) {
        double p = (double)e->count / L;
        entropy -= p * log2(p);
    }

    HASH_ITER(hh, entries, e, tmp) {
        HASH_DEL(entries, e);
        free(e->key);
        free(e);
    }

    return entropy;
}
