#include "dcl.h"

// Baseline implementation for HLS.
// Students will optimize this (loops, memory access, etc.).
void top_kernel(data_t A_DRAM[N_ROWS][N_COLS],
                data_t C_DRAM[N_ROWS][N_COLS]) {
#pragma HLS interface m_axi port=A_DRAM offset=slave bundle=A
#pragma HLS interface m_axi port=C_DRAM offset=slave bundle=C
#pragma HLS interface s_axilite port=return

    // ---- Tiling knobs (keep small and power-of-2 to start) ----
    const int TILE_COLS = 32;

    // On-chip buffers for A_DRAM and C_DRAM
    data_t A[N_ROWS][N_COLS];
    data_t C[N_ROWS][N_COLS];
    // Intermediate buffer for row-normalized values
    data_t tmp[N_ROWS][N_COLS];

#pragma HLS ARRAY_PARTITION variable=A   cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=tmp cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=C   cyclic factor=16 dim=2

    // Read in the data from DRAM to BRAM (tiled over columns)
L1:    for (int i = 0; i < N_ROWS; i++) {
    L2:    for (int jb = 0; jb < N_COLS; jb += TILE_COLS) {
        L3:    for (int tj = 0; tj < TILE_COLS; tj++) {
#pragma HLS PIPELINE II=1
                int j = jb + tj;
                if (j < N_COLS) {
                    A[i][j] = A_DRAM[i][j];
                    C[i][j] = 0;
                }
            }
        }
    }

    // Phase 1: Row-wise normalization (tiled over columns)
L4:    for (int i = 0; i < N_ROWS; i++) {
        data_t row_sum = 0.0;

        // Compute row sum
    L5:    for (int jb = 0; jb < N_COLS; jb += TILE_COLS) {
#pragma HLS PIPELINE II=1
        L6:    for (int tj = 0; tj < TILE_COLS; tj++) {
#pragma HLS PIPELINE II=1
                int j = jb + tj;
                if (j < N_COLS) {
                    row_sum += A[i][j];
                }
            }
        }

        // Avoid division by zero, add small bias
        data_t denom = row_sum + (data_t)1.0;

        // Normalize each element in the row
L7:        for (int jb = 0; jb < N_COLS; jb += TILE_COLS) {
    L8:        for (int tj = 0; tj < TILE_COLS; tj++) {
#pragma HLS PIPELINE II=1
                int j = jb + tj;
                if (j < N_COLS) {
                    tmp[i][j] = A[i][j] / denom;
                }
            }
        }
    }

    // Phase 2: Column-wise scaling (tiled over columns; structure preserved)
L9:    for (int jb = 0; jb < N_COLS; jb += TILE_COLS) {

        data_t col_sum[TILE_COLS];
#pragma HLS ARRAY_PARTITION variable=col_sum complete

        // init sums for this column tile
L10:        for (int tj = 0; tj < TILE_COLS; tj++) {
#pragma HLS PIPELINE II=1
            col_sum[tj] = 0.0;
        }

        // Compute column sums of normalized values (for this tile)
L11:        for (int i = 0; i < N_ROWS; i++) {
    L12:        for (int tj = 0; tj < TILE_COLS; tj++) {
#pragma HLS PIPELINE II=1
                int j = jb + tj;
                if (j < N_COLS) {
                    col_sum[tj] += tmp[i][j];
                }
            }
        }

        // Apply scale to each element in the column (for this tile)
L13:        for (int tj = 0; tj < TILE_COLS; tj++) {
#pragma HLS PIPELINE II=1
            int j = jb + tj;
            if (j < N_COLS) {
                // Compute average as scale
                data_t scale = col_sum[tj] / (data_t)N_ROWS;

                // Apply scale down the column
        L14:        for (int i = 0; i < N_ROWS; i++) {
#pragma HLS PIPELINE II=8
#pragma HLS unroll factor=8
                    C[i][j] = tmp[i][j] * scale;
                }
            }
        }
    }

    // Write back from BRAM to DRAM (tiled over columns)
L15:    for (int i = 0; i < N_ROWS; i++) {
    L16:    for (int jb = 0; jb < N_COLS; jb += TILE_COLS) {
        L17:    for (int tj = 0; tj < TILE_COLS; tj++) {
#pragma HLS PIPELINE II=1
                int j = jb + tj;
                if (j < N_COLS) {
                    C_DRAM[i][j] = C[i][j];
                }
            }
        }
    }
}
