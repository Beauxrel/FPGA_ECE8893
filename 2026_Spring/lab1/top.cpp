#include "dcl.h"

void top_kernel(data_t A_DRAM[N_ROWS][N_COLS],
                data_t C_DRAM[N_ROWS][N_COLS])
{
#pragma HLS interface m_axi port = A_DRAM offset = slave bundle = A \
    max_read_burst_length=256 num_read_outstanding=16 latency=64
#pragma HLS interface m_axi port = C_DRAM offset = slave bundle = C \
    max_write_burst_length=256 num_write_outstanding=16 latency=64
#pragma HLS interface s_axilite port = return

    // On-chip buffers
    data_t A[N_ROWS][N_COLS];
    data_t C[N_ROWS][N_COLS];
    data_t tmp[N_ROWS][N_COLS];
    
#pragma HLS ARRAY_PARTITION variable = A cyclic factor = 32 dim = 2
#pragma HLS ARRAY_PARTITION variable = C cyclic factor = 32 dim = 2
#pragma HLS ARRAY_PARTITION variable = tmp cyclic factor = 64 dim = 2

    // Load from DRAM
dram_to_bram_outer:
    for (int i = 0; i < N_ROWS; i++)
    {
dram_to_bram_inner:
        for (int j = 0; j < N_COLS; j++)
        {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
            A[i][j] = A_DRAM[i][j];
        }
    }

    // Phase 1: Row-wise normalization
phase_1:
    for (int i = 0; i < N_ROWS; i++)
    {
        data_t row_sum = 0.0;
        
        // Compute row sum
compute_row_sum:
        for (int j = 0; j < N_COLS; j++)
        {
#pragma HLS UNROLL
            row_sum += A[i][j];
        }
        
        // Normalize row
        data_t denom = row_sum + (data_t)1.0;
normalize_row:
        for (int j = 0; j < N_COLS; j++)
        {
#pragma HLS UNROLL
            tmp[i][j] = A[i][j] / denom;
        }
    }

    // Phase 2: Column-wise scaling
phase_2:
    for (int j = 0; j < N_COLS; j++)
    {
        data_t col_sum_val = 0.0;  // Fixed: renamed variable
        
        // Compute column sum
compute_col_sum:
        for (int i = 0; i < N_ROWS; i++)
        {
#pragma HLS UNROLL
            col_sum_val += tmp[i][j];
        }
        
        // Apply scaling
        data_t scale = col_sum_val / (data_t)N_ROWS;
col_scaling:
        for (int i = 0; i < N_ROWS; i++)
        {
#pragma HLS UNROLL
            C[i][j] = tmp[i][j] * scale;
        }
    }

    // Write back to DRAM (removed duplicate)
bram_to_dram_outer:
    for (int i = 0; i < N_ROWS; i++)
    {
bram_to_dram_inner:
        for (int j = 0; j < N_COLS; j++)
        {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
            C_DRAM[i][j] = C[i][j];
        }
    }
}