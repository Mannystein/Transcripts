#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "helpers.h" //Memory management, indexing and sorting
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

/** Some definitions that stay the same throughout the whole package:
* T: Length of time series
* tau: Embedding delay
* p: Embedding dimension
* L: Length of the symbol series
*/

// Landau function describes the maximum order class for a given p
const int LANDAU[13] = {1,1,2,3,4,6,6,12,15,20,30,30,60};


// Function to compute factorial
int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; i++)
        result *= i;
    return result;
}

// compute Kullback-Leibler distance for two probability distributions of length prob_len
double D_KL(double* P1, double* P2, int prob_len) {
    double sum = 0;
    for (int i = 0; i < prob_len; i++) {
        if (P1[i] > 0 && P2[i] > 0) {
            sum += P1[i] * log2(P1[i] / P2[i]);
        } else if (P1[i] > 0 && P2[i] == 0) {
            // KL divergence is undefined if P2[i] is 0 while P1[i] is not
            printf("Error: P2[%d] is zero, leading to an undefined KL divergence.\n", i);
            return INFINITY;
        }
    }

    return sum;
}

// function to generate permutation from a given permutation rank (lexicographic order)
void permutation_from_rank(int rank, int len, int result[]) {
    int available[len]; // Array of available numbers
    for (int i = 0; i < len; i++) available[i] = i;

    int fact = factorial(len - 1);

    for (int i = 0; i < len; i++) {
        int index = rank / fact;  // Determine which element to take
        result[i] = available[index];

        // Shift elements left to remove used number
        for (int j = index; j < len - i - 1; j++) {
            available[j] = available[j + 1];
        }

        rank %= fact;
        fact = (i + 1 < len) ? fact / (len - i - 1) : 1;  // Reduce factorial
    }
}

// function to compute the rank of a permutation (lexicographic order)
int permutation_rank(int arr[], int len) {
    int rank = 0, used[len], fact = factorial(len - 1);

    memset(used, 0, sizeof(used));
    for (int i = 0; i < len; i++) {
        int count = 0;
        for (int j = 0; j < arr[i]; j++)
            if (!used[j]) count++;

        rank += count * fact;
        fact = (i + 1 < len) ? fact / (len - i - 1) : 1;
        used[arr[i]] = 1;
    }
    return rank;
}

//calculate takens delay embedding
double* delay_vectors(const double* ts, int T, int tau, int p) {
    int i, j;
    int L = T-tau*(p-1);

    double* dv = malloc((L*p) * sizeof(double));
   
    for (i = 0; i < L; ++i) {
        for (j = 0; j < p; ++j) {
            dv[i * p + j] = ts[i + j * tau];
            //printf("%f \n", v_i_list[i][j]);
        }
    }

    return dv;
}

// create consecutive partitions of length p
double* partitions(const double* ts, int T, int p) {
    int i, j;
    int L = T / p;  // number of full partitions

    double* dv = malloc((L * p) * sizeof(double));

    for (i = 0; i < L; ++i) {
        for (j = 0; j < p; ++j) {
            dv[i * p + j] = ts[i * p + j];
        }
    }

    return dv;
}

// symbolization
int* delay_vectors_to_symbols(double* dv, int rows, int cols, int F) {
    int i, j;

    // Allocate memory for the argsorted indices
    int* sorted_indices = malloc(rows * cols * sizeof(int));
    if (!sorted_indices) {
        // Handle memory allocation failure
        return NULL;
    }

    // Perform argsort for each row
    for (i = 0; i < rows; ++i) {
        int* indices = malloc(cols * sizeof(int));
        if (!indices) {
            // Free previously allocated memory before returning
            free(sorted_indices);
            return NULL;
        }

        // Compute the argsort for the row
        argsort(indices, &dv[i*cols], cols);

        // Copy the argsorted indices into the output array
        for (j = 0; j < cols; ++j) {
            sorted_indices[i * cols + j] = indices[j];
        }

        // Free memory for the temporary indices array
        free(indices);
    }

    // Adjust indices if F is true
    if (F) {
        for (i = 0; i < rows; ++i) {
            for (j = 0; j < cols; ++j) {
                sorted_indices[i * cols + j]++;
            }
        }
    }

    return sorted_indices;
}

// calculate transcript between two ordinal patterns
int* get_transcript2(int* symbol1, int*symbol2, int size) {
    int* invert = malloc(size*sizeof(int));
    int* transcript = malloc(size*sizeof(int));
    for (int i=0; i<size; ++i) {
        invert[symbol1[i]] = i;
    }

    for (int i=0; i<size; ++i) {
        transcript[i] = invert[symbol2[i]];
    }

    free(invert);

    return transcript;
}

// compute transcripts between two series of OPs
int* get_transcripts(int* symbols1, int* symbols2, int rows, int cols) {
    int* transcripts = malloc(rows * cols * sizeof(int));

    for (int i = 0; i < rows; ++i) {
        int* invert = malloc(cols * sizeof(int));

        // Build inverse permutation for the i-th row of symbols1
        for (int j = 0; j < cols; ++j) {
            invert[symbols1[i * cols + j]] = j;
        }

        // Apply inverse to the i-th row of symbols2
        for (int j = 0; j < cols; ++j) {
            transcripts[i * cols + j] = invert[symbols2[i * cols + j]];
        }

        free(invert);
    }

    return transcripts;
}

// calculate the order class of an ordinal pattern
int find_order_class(int* transcript, int length) {
    int* tau = malloc(length * sizeof(int));
    for (int i = 0; i < length; ++i) {
        tau[i] = transcript[i];
    }

    int oc = 1;

    while (!is_increasing_sequence(tau, length)) {
        for (int i = 0; i < length; ++i) {
            tau[i] = transcript[tau[i]];
        }
        oc++;
    }

    free(tau);
    return oc;
}

// calculate the order classes for a series of OPs 
int* find_order_class_array(int* transcripts, int n_transcripts, int p) {
    int* order_classes = malloc(n_transcripts * sizeof(int));
    int* tau = malloc(p * sizeof(int)); // Shared buffer

    for (int i = 0; i < n_transcripts; ++i) {
        int* transcript = &transcripts[i * p];

        // Initialize tau
        for (int j = 0; j < p; ++j) {
            tau[j] = transcript[j];
        }

        int oc = 1;

        // Iterate until tau becomes sorted (strictly increasing)
        while (!is_increasing_sequence(tau, p)) {
            for (int j = 0; j < p; ++j) {
                tau[j] = transcript[tau[j]];
            }
            oc++;
        }

        order_classes[i] = oc;
    }

    free(tau);
    return order_classes;
}

// calculate Jensen-Shannon divergence for order classes
double calc_JS_C(int* A, int* B, int subarray_len, int num_subarrays) {

    int factorial_val = factorial(subarray_len);
    int landau_val = LANDAU[subarray_len];
    int matrix_size = factorial_val * factorial_val;

    // flattened 2D matrices
    double* joint_count = calloc(matrix_size, sizeof(double));
    double* ind_count = calloc(matrix_size, sizeof(double));

    double* margA = calloc(factorial_val, sizeof(double));
    double* margB = calloc(factorial_val, sizeof(double));
    double* oc_probs = calloc(landau_val, sizeof(double));
    double* oc_probs_ind = calloc(landau_val, sizeof(double));
    double* mixing_probs = calloc(landau_val, sizeof(double));

    // joint probability accumulation
    for (int i = 0; i < num_subarrays; ++i) {
        int* subA = &A[i * subarray_len];
        int* subB = &B[i * subarray_len];
        int rank_A = permutation_rank(subA, subarray_len);
        int rank_B = permutation_rank(subB, subarray_len);
        joint_count[rank_A * factorial_val + rank_B] += (1.0 / num_subarrays);
    }

    // marginals from joint_count
    for (int i = 0; i < factorial_val; ++i) {
        for (int j = 0; j < factorial_val; ++j) {
            double val = joint_count[i * factorial_val + j];
            margA[i] += val;
            margB[j] += val;
        }
    }

    // independent probabilities
    for (int i = 0; i < factorial_val; ++i) {
        for (int j = 0; j < factorial_val; ++j) {
            ind_count[i * factorial_val + j] = margA[i] * margB[j];
        }
    }

    // accumulate order class probabilities
    int* permA = malloc(subarray_len * sizeof(int));
    int* permB = malloc(subarray_len * sizeof(int));
    for (int i = 0; i < factorial_val; ++i) {
        for (int j = 0; j < factorial_val; ++j) {
            double p_joint = joint_count[i * factorial_val + j];
            double p_ind = ind_count[i * factorial_val + j];
            if (p_joint > 0 || p_ind > 0) {
                permutation_from_rank(i, subarray_len, permA);
                permutation_from_rank(j, subarray_len, permB);
                int* transcript = get_transcript2(permA, permB, subarray_len);
                int oc = find_order_class(transcript, subarray_len);
                oc_probs[oc - 1] += p_joint;
                oc_probs_ind[oc - 1] += p_ind;
                free(transcript);
            }
        }
    }
    free(permA);
    free(permB);

    // Jensen-Shannon divergence
    for (int i = 0; i < landau_val; ++i) {
        mixing_probs[i] = 0.5 * (oc_probs[i] + oc_probs_ind[i]);
    }

    double D_P_M = D_KL(oc_probs, mixing_probs, landau_val);
    double D_Q_M = D_KL(oc_probs_ind, mixing_probs, landau_val);

    free(joint_count);
    free(ind_count);
    free(margA);
    free(margB);
    free(oc_probs);
    free(oc_probs_ind);
    free(mixing_probs);

    return 0.5 * (D_P_M + D_Q_M);
}

double CMI_symbols(int* symb1, int* symb2, int len, int p) {
    int* ranks = malloc(2*len*sizeof(int));
    for (int i=0; i<len; ++i) {
        ranks[i] = permutation_rank(&symb1[i*p],p);
        ranks[i+len] = permutation_rank(&symb2[i*p],p);
    }
    
    double H1 = joint_entropy_ranks(ranks,1,len);
    double H2 = joint_entropy_ranks(&ranks[len],1,len);
    double H12 = joint_entropy_ranks(ranks,2,len);
    free(ranks);
    return H1+H2-H12;
}

double permutation_entropy(int* symbols, int p, int len) {
    int* ranks = malloc(len*sizeof(int));
    for (int i=0; i<len; ++i) {
        ranks[i] = permutation_rank(&symbols[i*p],p);
        }
    
    double H = joint_entropy_ranks(ranks,1,len);

    return H;
}

double* transcript_mutual_info(double* ts1, double* ts2, int T, int Lambda, int tau, int p) {
    double* dv1 = NULL;
    double* dv2 = NULL;
    int* symb1 = NULL;
    int* symb2 = NULL;
    int L;

    if (tau == 0) {
        L = T / p;  // or adjusted if delay_vectors affects length
        dv1 = partitions(ts1, T, p);
        dv2 = partitions(ts2, T, p);
        symb1 = delay_vectors_to_symbols(ts1, L, p, 0);
        symb2 = delay_vectors_to_symbols(ts2, L, p, 0);
    } 
    else if (tau > 0) {
        L = T - tau * (p - 1);
        dv1 = delay_vectors(ts1, T, tau, p);
        dv2 = delay_vectors(ts2, T, tau, p);
        symb1 = delay_vectors_to_symbols(dv1, L, p, 0);
        symb2 = delay_vectors_to_symbols(dv2, L, p, 0);
    }

    int LL = L - Lambda;

    int* tau_ts = get_transcripts(symb1, symb2, LL, p);
    int* tau_st = get_transcripts(symb2, symb1, LL, p);
    int* tau_As = get_transcripts(symb1, &symb1[Lambda * p], LL, p);
    int* tau_At = get_transcripts(symb2, &symb2[Lambda * p], LL, p);

    double* return_arr = malloc(2 * sizeof(double));
    return_arr[0] = CMI_symbols(tau_At, tau_st, LL, p);
    return_arr[1] = CMI_symbols(tau_As, tau_ts, LL, p);

    // Cleanup
    if (dv1) free(dv1);
    if (dv2) free(dv2);
    if (symb1) free(symb1);
    if (symb2) free(symb2);
    free(tau_st);
    free(tau_ts);
    free(tau_As);
    free(tau_At);

    return return_arr;
}

double coupling_complexity_symbols(int* symbol_series, int N, int L, int p) {
    double min_H_alpha = 10000000;
    int* symbol_ranks = malloc(N*L*sizeof(int));
    int* transcript_ranks = malloc((N-1)*L*sizeof(int));
    for (int i=0; i<N; ++i) {
        int* temp_symb = &symbol_series[i*L*p];

        for (int j=0; j<L;++j) {
            symbol_ranks[i*L+j] = permutation_rank(&temp_symb[j*p],p);
          // printf("%i \n", symbol_ranks[i*L+j]);
        }

        // for (int j=0; j<L;++j) {
        //     for (int k=0; k<p; ++k) {
        //         printf("%i, ", temp_symb[j*p+k]);
        //     }
        //     printf("\n");
        // }
        double H_temp = joint_entropy_ranks(&symbol_ranks[i*L],1,L);
        if (H_temp < min_H_alpha) {
            min_H_alpha = H_temp;
        }
    }

    for (int i=0; i<(N-1); ++i) {
        int* temp_trs = get_transcripts(&symbol_series[i*L*p], &symbol_series[(i+1)*L*p], L, p);
        for (int j=0; j<L; ++j) {
            transcript_ranks[i*L+j] = permutation_rank(&temp_trs[j*p], p);
        }
        free(temp_trs);
    }

    double H_alpha = joint_entropy_ranks(symbol_ranks, N, L);
    double H_tau = joint_entropy_ranks(transcript_ranks, N-1, L);
    
    free(symbol_ranks);
    free(transcript_ranks);

    return min_H_alpha-(H_alpha-H_tau);
}

double coupling_complexity(double* time_series, int N, int T, int tau, int p) {
    int L = T-tau*(p-1);
    double min_H_alpha = 10000000;
    int* symbols = malloc(N*L*p*sizeof(int));
    int* symbol_ranks = malloc(N*L*sizeof(int));
    int* transcript_ranks = malloc((N-1)*L*sizeof(int));
    double* temp_dv;
    for (int i=0; i<N; ++i) {
        // printf("%i\n", i);
        if (tau == 0) {
            temp_dv = partitions(&time_series[i*T], T, p);
        }
        else {
            temp_dv = delay_vectors(&time_series[i*T], T, tau, p);
        }
        int* temp_symb = delay_vectors_to_symbols(temp_dv, L, p, 0);
        for (int j=0; j<L;++j) {
            symbol_ranks[i*L+j] = permutation_rank(&temp_symb[j*p],p);
          // printf("%i \n", symbol_ranks[i*L+j]);
        }
        double H_temp = joint_entropy_ranks(&symbol_ranks[i*L],1,L);
        if (H_temp < min_H_alpha) {
            min_H_alpha = H_temp;
        }

        memcpy(&symbols[i*L*p], temp_symb, L*p*sizeof(int));
        free(temp_dv);
        free(temp_symb);
    }
    for (int i=0; i<(N-1); ++i) {
        int* temp_trs = get_transcripts(&symbols[i*L*p], &symbols[(i+1)*L*p], L, p);
        for (int j=0; j<L; ++j) {
            transcript_ranks[i*L+j] = permutation_rank(&temp_trs[j*p], p);
        }
        free(temp_trs);
    }

    double H_alpha = joint_entropy_ranks(symbol_ranks, N, L);
    double H_tau = joint_entropy_ranks(transcript_ranks, N-1, L);
    
    free(symbols);
    free(symbol_ranks);
    free(transcript_ranks);

    return min_H_alpha-(H_alpha-H_tau);
}
