#include "dcl.h"

void top_kernel(data_t A_DRAM[N_ROWS][N_COLS],
                data_t C_DRAM[N_ROWS][N_COLS]) {
#pragma HLS interface m_axi port=A_DRAM offset=slave bundle=A
#pragma HLS interface m_axi port=C_DRAM offset=slave bundle=C
#pragma HLS interface s_axilite port=return

    data_t A[N_ROWS][N_COLS];
    data_t tmp[N_ROWS][N_COLS];
    data_t C[N_ROWS][N_COLS];

    data_t row_sum_arr[N_ROWS];
    data_t col_sum_arr[N_COLS];

    // Load A
    for (int i = 0; i < N_ROWS; i++) {
        for (int j = 0; j < N_COLS; j++) {
#pragma HLS PIPELINE II=1
            A[i][j] = A_DRAM[i][j];
        }
    }

    // Phase 1a: row sums (store each row's sum)
    for (int i = 0; i < N_ROWS; i++) {
        data_t row_sum = 0.0;
        for (int j = 0; j < N_COLS; j++) {
#pragma HLS PIPELINE II=1
            row_sum += A[i][j];
        }
        row_sum_arr[i] = row_sum;
    }

    // Phase 1b: row normalize using per-row denom
    for (int i = 0; i < N_ROWS; i++) {
        data_t denom = row_sum_arr[i] + (data_t)1.0;
        for (int j = 0; j < N_COLS; j++) {
#pragma HLS PIPELINE II=1
            tmp[i][j] = A[i][j] / denom;
        }
    }

    // Phase 2a: column sums (store each column's sum)
    for (int j = 0; j < N_COLS; j++) {
        data_t col_sum = 0.0;
        for (int i = 0; i < N_ROWS; i++) {
#pragma HLS PIPELINE II=1
            col_sum += tmp[i][j];
        }
        col_sum_arr[j] = col_sum;
    }

    // Phase 2b: scale using per-column scale
    for (int j = 0; j < N_COLS; j++) {
        data_t scale = col_sum_arr[j] / (data_t)N_ROWS;
        for (int i = 0; i < N_ROWS; i++) {
#pragma HLS PIPELINE II=1
            C[i][j] = tmp[i][j] * scale;
        }
    }

    // Store C
    for (int i = 0; i < N_ROWS; i++) {
        for (int j = 0; j < N_COLS; j++) {
#pragma HLS PIPELINE II=1
            C_DRAM[i][j] = C[i][j];
        }
    }
}
