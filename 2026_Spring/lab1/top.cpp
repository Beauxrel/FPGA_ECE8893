#include "dcl.h"

// Latency-optimized baseline (same math), with row-major Phase 3/4,
// inner-loop pipelining for AXI bursts, and partition/unroll alignment.
void top_kernel(data_t A_DRAM[N_ROWS][N_COLS],
                data_t C_DRAM[N_ROWS][N_COLS])
{
#pragma HLS interface mode=m_axi port=A_DRAM offset=slave bundle=A \
  max_read_burst_length=64 num_read_outstanding=16 latency=0
#pragma HLS interface mode=m_axi port=C_DRAM offset=slave bundle=C \
  max_write_burst_length=64 num_write_outstanding=16 latency=0
#pragma HLS interface s_axilite port=return

    // On-chip buffers
    data_t A[N_ROWS][N_COLS];
    data_t C[N_ROWS][N_COLS];
    data_t row_sum[N_ROWS];
    data_t col_sum_buf[N_COLS];
    data_t tmp[N_ROWS][N_COLS];
    data_t scale[N_COLS];

    // Partitioning to enable parallel column access (vectorize over j)
#pragma HLS ARRAY_PARTITION variable=A   cyclic factor=32 dim=2
#pragma HLS ARRAY_PARTITION variable=C   cyclic factor=32 dim=2
#pragma HLS ARRAY_PARTITION variable=tmp cyclic factor=32 dim=2
#pragma HLS ARRAY_PARTITION variable=scale cyclic factor=32 dim=1
    // (row_sum/col_sum_buf left unpartitioned; col_sum_buf accessed sequentially)

    // ---------------------------------------------------------------------
    // DRAM -> BRAM (burst-friendly): pipeline inner loop
    // ---------------------------------------------------------------------
dram_to_bram_outer:
    for (int i = 0; i < N_ROWS; i++) {
    dram_to_bram_inner:
        for (int j = 0; j < N_COLS; j++) {
#pragma HLS PIPELINE II=1
            A[i][j] = A_DRAM[i][j];
        }
    }

    // ---------------------------------------------------------------------
    // Phase 1: row-wise sums (vectorized over columns)
    // ---------------------------------------------------------------------
phase_1:
    for (int i = 0; i < N_ROWS; i++) {
#pragma HLS PIPELINE II=1
        data_t acc = 0;
    compute_row:
        for (int j = 0; j < N_COLS; j++) {
#pragma HLS UNROLL factor=4
            acc += A[i][j];
        }
        row_sum[i] = acc;
    }

    // ---------------------------------------------------------------------
    // Phase 2: row-wise normalization tmp = A / (row_sum+1)
    // ---------------------------------------------------------------------
phase_2:
    for (int i = 0; i < N_ROWS; i++) {
#pragma HLS PIPELINE II=1
        data_t denom = row_sum[i] + (data_t)1.0;
    div_loop:
        for (int j = 0; j < N_COLS; j++) {
#pragma HLS UNROLL factor=4
            tmp[i][j] = A[i][j] / denom;
        }
    }

    // ---------------------------------------------------------------------
    // Phase 3: column sums of tmp (reordered to row-major for locality)
    // col_sum_buf[j] = sum_i tmp[i][j]
    // ---------------------------------------------------------------------
init_col_sum:
    for (int j = 0; j < N_COLS; j++) {
#pragma HLS PIPELINE II=1
        col_sum_buf[j] = 0;
    }

phase_3:
    for (int i = 0; i < N_ROWS; i++) {
#pragma HLS PIPELINE II=1
        for (int j = 0; j < N_COLS; j++) {
#pragma HLS UNROLL factor=4
            col_sum_buf[j] += tmp[i][j];
        }
    }

    // ---------------------------------------------------------------------
    // Compute scale per column once: scale[j] = col_sum_buf[j] / N_ROWS
    // ---------------------------------------------------------------------
compute_scale:
    for (int j = 0; j < N_COLS; j++) {
#pragma HLS PIPELINE II=1
        scale[j] = col_sum_buf[j] / (data_t)N_ROWS;
    }

    // ---------------------------------------------------------------------
    // Phase 4: apply scaling (row-major, vectorized over columns)
    // C[i][j] = tmp[i][j] * scale[j]
    // ---------------------------------------------------------------------
phase_4:
    for (int i = 0; i < N_ROWS; i++) {
#pragma HLS PIPELINE II=1
        for (int j = 0; j < N_COLS; j++) {
#pragma HLS UNROLL factor=4
            C[i][j] = tmp[i][j] * scale[j];
        }
    }

    // ---------------------------------------------------------------------
    // BRAM -> DRAM (burst-friendly): pipeline inner loop
    // ---------------------------------------------------------------------
bram_to_dram_outer:
    for (int i = 0; i < N_ROWS; i++) {
    bram_to_dram_inner:
        for (int j = 0; j < N_COLS; j++) {
#pragma HLS PIPELINE II=1
            C_DRAM[i][j] = C[i][j];
        }
    }
}
