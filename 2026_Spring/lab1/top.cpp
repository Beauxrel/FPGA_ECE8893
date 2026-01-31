#include "dcl.h"

// Baseline implementation for HLS.
// Students will optimize this (loops, memory access, etc.).
void top_kernel(data_t A_DRAM[N_ROWS][N_COLS],
                data_t C_DRAM[N_ROWS][N_COLS])
{
#pragma HLS interface mode=m_axi port = A_DRAM offset = slave bundle = A max_widen_bitwidth=512 max_read_burst_length=64
#pragma HLS interface mode=m_axi port = C_DRAM offset = slave bundle = C max_widen_bitwidth=512 max_read_burst_length=64
#pragma HLS interface s_axilite port = return

    // On-chip buffers for A_DRAM and C_DRAM
    data_t A[N_ROWS][N_COLS];
    data_t C[N_ROWS][N_COLS];
    data_t row_sum = 0.0;
    data_t col_sum = 0.0;

#pragma HLS ARRAY_PARTITION variable = A cyclic factor = 32 dim = 1
#pragma HLS ARRAY_PARTITION variable = C cyclic factor = 32 dim = 2

dram_to_bram_outer:
    for (int i = 0; i < N_ROWS; i++)
    {
    dram_to_bram_inner:
        for (int j = 0; j < N_COLS; j++)
        {
#pragma HLS PIPELINE II=1
            A[i][j] = A_DRAM[i][j];
        }
    }

    // Intermediate buffer for row-normalized values
    data_t tmp[N_ROWS][N_COLS];
#pragma HLS ARRAY_PARTITION variable = tmp cyclic factor = 8 dim = 2
    // Phase 1: Row-wise normalization
phase_1:
    for (int i = 0; i < N_ROWS; i++)
    {
        row_sum = 0.0;
        // Compute row sum
    compute_row:
        for (int j = 0; j < N_COLS; j++)
        {
#pragma HLS PIPELINE II = 1
            row_sum += A[i][j];
        }
    }
phase_2:
    for (int i = 0; i < N_ROWS; i++)
    {
        // Avoid division by zero, add small bias
        data_t denom = row_sum + (data_t)1.0;
    div_loop:
        for (int j = 0; j < N_COLS; j++)
        {
#pragma HLS PIPELINE II = 1
            tmp[i][j] = A[i][j] / denom;
        }
    }
phase_3:
    // Phase 2: Column-wise scaling

    for (int j = 0; j < N_COLS; j++)
    {
        col_sum = 0.0;
        // Compute column sum of normalized values
    col_sum:
        for (int i = 0; i < N_ROWS; i++)
        {
#pragma HLS PIPELINE II = 1
            col_sum += tmp[i][j];
        }
    }
phase_4:
    for (int j = 0; j < N_COLS; j++)
    {
        // Compute average as scale
        data_t scale = col_sum / (data_t)N_ROWS;

        // Apply scale to each element in the column
    col_scaling:
        for (int i = 0; i < N_ROWS; i++)
        {
#pragma HLS PIPELINE II = 1
            C[i][j] = tmp[i][j] * scale;
        }
    }

bram_to_dram_outer:
    for (int i = 0; i < N_ROWS; i++)
    {
    bram_to_dram_inner:
        for (int j = 0; j < N_COLS; j++)
        {
            C_DRAM[i][j] = C[i][j];
        }
    }
}