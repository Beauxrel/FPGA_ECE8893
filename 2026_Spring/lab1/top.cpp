#include "dcl.h"

// Baseline implementation for HLS.
// Students will optimize this (loops, memory access, etc.).
void top_kernel(data_t A[N_ROWS][N_COLS],
                data_t C[N_ROWS][N_COLS]) {
    // Intermediate buffer for row-normalized values
    static data_t arr_1[N_ROWS][N_COLS];
    static data_t tmp[N_ROWS][N_COLS];

#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=return

    // BRAM buffers
    static data_t A_bram[N_ROWS][N_COLS];
    static data_t C_bram[N_ROWS][N_COLS];
    static data_t tmp[N_ROWS][N_COLS];

#pragma HLS bind_storage variable=A_bram type=ram_2p impl=bram
#pragma HLS bind_storage variable=C_bram type=ram_2p impl=bram
#pragma HLS bind_storage variable=tmp    type=ram_2p impl=bram

#pragma HLS ARRAY_PARTITION variable=A_bram cyclic factor=32 dim=2
#pragma HLS ARRAY_PARTITION variable=C_bram cyclic factor=32 dim=1
#pragma HLS ARRAY_PARTITION variable=tmp    cyclic factor=32 dim=1

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

        // Normalize each element in the row
        norm_row: for (int j = 0; j < N_COLS; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS unroll factor=4
            tmp[i][j] = A[i][j] / denom;
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
