#include "dcl.h"

// Phase 1: Row-wise normalization
void row_normalize(data_t A[N_ROWS][N_COLS],
                   data_t tmp[N_ROWS][N_COLS]) {
#pragma HLS INLINE off
    
    for (int i = 0; i < N_ROWS; i++) {
#pragma HLS PIPELINE II=1
        data_t row_sum = 0.0;
        
        // Compute row sum
        for (int j = 0; j < N_COLS; j++) {
#pragma HLS UNROLL
            row_sum += A[i][j];
        }
        
        // Avoid division by zero
        data_t denom = row_sum + (data_t)1.0;
        
        // Normalize each element in the row
        for (int j = 0; j < N_COLS; j++) {
#pragma HLS UNROLL
            tmp[i][j] = A[i][j] / denom;
        }
    }
}

// Phase 2: Column-wise scaling
void column_scale(data_t tmp[N_ROWS][N_COLS],
                  data_t C[N_ROWS][N_COLS]) {
#pragma HLS INLINE off
    
    for (int j = 0; j < N_COLS; j++) {
#pragma HLS PIPELINE II=1
        data_t col_sum = 0.0;
        
        // Compute column sum
        for (int i = 0; i < N_ROWS; i++) {
#pragma HLS UNROLL
            col_sum += tmp[i][j];
        }
        
        // Compute average as scale
        data_t scale = col_sum / (data_t)N_ROWS;
        
        // Apply scale to each element
        for (int i = 0; i < N_ROWS; i++) {
#pragma HLS UNROLL
            C[i][j] = tmp[i][j] * scale;
        }
    }
}

// Top-level kernel with dataflow
void top_kernel(data_t A[N_ROWS][N_COLS],
                data_t C[N_ROWS][N_COLS]) {
    
    // Intermediate buffer
    static data_t tmp[N_ROWS][N_COLS];
#pragma HLS ARRAY_PARTITION variable=tmp complete dim=2
#pragma HLS ARRAY_PARTITION variable=tmp cyclic factor=8 dim=1
    
    // Partition input/output arrays
#pragma HLS ARRAY_PARTITION variable=A complete dim=2
#pragma HLS ARRAY_PARTITION variable=A cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=C complete dim=2
#pragma HLS ARRAY_PARTITION variable=C cyclic factor=8 dim=1
    
    // Enable dataflow between phases
#pragma HLS DATAFLOW
    
    row_normalize(A, tmp);
    column_scale(tmp, C);
}