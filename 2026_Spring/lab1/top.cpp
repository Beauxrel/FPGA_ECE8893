#include "dcl.h"

// Baseline implementation for HLS.
// Students will optimize this (loops, memory access, etc.).
void top_kernel(data_t A[N_ROWS][N_COLS],
                data_t C[N_ROWS][N_COLS]) {
    // Intermediate buffer for row-normalized values
    static data_t tmp[N_ROWS][N_COLS];
    static data_t row_buf[N_COLS];
#pragma HLS INTERFACE m_axi port=A bundle=gmem0
#pragma HLS INTERFACE m_axi port=C bundle=gmem1
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS ARRAY_PARTITION variable=tmp cyclic factor=32 dim=1
//#pragma HLS ARRAY_PARTITION variable=A   cyclic factor=32 dim=2
//#pragma HLS ARRAY_PARTITION variable=C   cyclic factor=32 dim=1

    // Phase 1: Row-wise normalization
    phase_1: for (int i = 0; i < N_ROWS; i++) {
        data_t row_sum = 0.0;

        // Compute row sum!
        compute_row: for (int j = 0; j < N_COLS; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS unroll factor=4
            row_sum += A[i][j];
        }

        // Avoid division by zero, add small bias
        data_t denom = row_sum + (data_t)1.0;
        data_t recip = (data_t)1.0 / denom;

        // Normalize each element in the row
        norm_row: for (int j = 0; j < N_COLS; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS unroll factor=4
            tmp[i][j] = row_buf[j] / denom;
        }
    }

    // Phase 2: Column-wise scaling
    for (int j = 0; j < N_COLS; j++) {
        data_t col_sum = 0.0;

        // Compute column sum of normalized values
        for (int i = 0; i < N_ROWS; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS unroll factor=4
            col_sum += tmp[i][j];
        }

        // Compute average as scale
        data_t scale = col_sum / (data_t)N_ROWS;

        // Apply scale to each element in the column
        for (int i = 0; i < N_ROWS; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS unroll factor=4
            C[i][j] = tmp[i][j] * scale;
        }
    }
}
