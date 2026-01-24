#include "dcl.h"

// Baseline implementation for HLS.
// Students will optimize this (loops, memory access, etc.).
void top_kernel(data_t A[N_ROWS][N_COLS],
                data_t C[N_ROWS][N_COLS]) {
    // Intermediate buffer for row-normalized values
    static data_t tmp[N_ROWS][N_COLS];
    #pragma HLS ARRAY_PARTITION variable=tmp cyclic factor=16 dim=2
    #pragma HLS ARRAY_PARTITION variable=A   cyclic factor=16 dim=2
    #pragma HLS ARRAY_PARTITION variable=C   cyclic factor=16 dim=2

    #pragma HLS loop_flatten
    // Phase 1: Row-wise normalization
    phase_1: for (int i = 0; i < N_ROWS; i++) {
        data_t row_sum = 0.0;
        // Compute row sum!
        #pragma HLS PIPELINE II=1
        #pragma HLS unroll factor=16
        compute_row: for (int j = 0; j < N_COLS; j++) {
            row_sum += A[i][j];
        }
        // Avoid division by zero, add small bias
        //data_t denom = row_sum + (data_t)1.0;
        // Normalize each element in the row
        #pragma HLS PIPELINE II=1
        #pragma HLS unroll factor=16
        norm_row: for (int j = 0; j < N_COLS; j++) {
            #pragma HLS unroll factor=16
            tmp[i][j] = A[i][j] / row_sum + (data_t)1.0;
        }
    }

    // Phase 2: Column-wise scaling
    for (int j = 0; j < N_COLS; j++) {
        #pragma HLS loop_flatten
        data_t col_sum = 0.0;
        // Compute column sum of normalized values
        #pragma HLS PIPELINE II=1
        compute_col: for (int i = 0; i < N_ROWS; i++) {
            col_sum += tmp[i][j];
        }
        // Compute average as scale
        //data_t scale = col_sum / (data_t)N_ROWS;
        #pragma HLS PIPELINE II=1
        #pragma HLS unroll factor=16
        // Apply scale to each element in the column
        scale_col: for (int i = 0; i < N_ROWS; i++) {
            C[i][j] = tmp[i][j] * col_sum / (data_t)N_ROWS;
        }
    }
}
