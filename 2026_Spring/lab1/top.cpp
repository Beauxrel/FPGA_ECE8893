#include "dcl.h"

// Baseline implementation for HLS.
// Students will optimize this (loops, memory access, etc.).
void top_kernel(data_t A_DRAM[N_ROWS][N_COLS],
                data_t C_DRAM[N_ROWS][N_COLS])
{
#pragma HLS interface mode=m_axi port=A_DRAM offset=slave bundle=A \
  max_read_burst_length=64 num_read_outstanding=16 latency=0

#pragma HLS interface mode=m_axi port=C_DRAM offset=slave bundle=C \
  max_write_burst_length=64 num_write_outstanding=16 latency=0
#pragma HLS interface s_axilite port = return

    // On-chip buffers for A_DRAM and C_DRAM
    data_t A[N_ROWS][N_COLS];
    data_t C[N_ROWS][N_COLS];
    data_t row_sum[N_ROWS];
    data_t col_sum[N_COLS];
    // Intermediate buffer for row-normalized values
    data_t tmp[N_ROWS][N_COLS];
#pragma HLS ARRAY_PARTITION variable = tmp cyclic factor = 32 dim = 2
#pragma HLS ARRAY_PARTITION variable = A cyclic factor = 32 dim = 2
#pragma HLS ARRAY_PARTITION variable = C cyclic factor = 32 dim = 2

dram_to_bram_outer:
    for (int i = 0; i < N_ROWS; i++){
    dram_to_bram_inner:
        for (int j = 0; j < N_COLS; j++){
#pragma HLS PIPELINE II=1
            A[i][j] = A_DRAM[i][j];
        }
    }
col_init:
// init col_sum
    for (int i = 0; i < N_COLS; i++){
#pragma HLS PIPELINE II=1
    row_sum[i] = 0;
    }
    // Phase 1: Row-wise normalization
phase_1:
    for (int i = 0; i < N_ROWS; i++){
#pragma HLS PIPELINE II=1
compute_row:
        for (int j = 0; j < N_COLS; j++){
#pragma HLS UNROLL factor=32
            row_sum[i] += A[i][j];
        }
    }
    // Avoid division by zero, add small bias
    
phase_2:
    for (int i = 0; i < N_ROWS; i++){
#pragma HLS PIPELINE II=1
        data_t denom = row_sum[i] + (data_t)1.0;
    div_loop:
        for (int j = 0; j < N_COLS; j++)
        {
#pragma HLS UNROLL factor=32
            tmp[i][j] = A[i][j] / denom;
        }
    }
col_init:
// init col_sum
    for (int j = 0; j < N_COLS; j++){
#pragma HLS PIPELINE II=1
    col_sum[j] = 0;
    }
phase_3:
    // Phase 2: Column-wise scaling
    for (int j = 0; j < N_COLS; j++){
#pragma HLS PIPELINE II=1
col_sum:
        for (int i = 0; i < N_ROWS; i++){
#pragma HLS UNROLL factor=32
            col_sum[j] += tmp[i][j];
        }
    }

phase_4:
    for (int j = 0; j < N_COLS; j++){
#pragma HLS PIPELINE II=1
    data_t scale = col_sum[j] / (data_t)N_ROWS;
    col_scaling:
        for (int i = 0; i < N_ROWS; i++){
#pragma HLS UNROLL factor=32
            C[i][j] = tmp[i][j] * scale;
        }
    }

bram_to_dram_outer:
    for (int i = 0; i < N_ROWS; i++)
    {
#pragma HLS PIPELINE II=64
    bram_to_dram_inner:
        for (int j = 0; j < N_COLS; j++)
        {
#pragma HLS PIPELINE II=1
            C_DRAM[i][j] = C[i][j];
        }
    }
}