#include "dcl.h"

#define TILE_R 32
#define TILE_C 32

void top_kernel(data_t A[N_ROWS][N_COLS],
                data_t C[N_ROWS][N_COLS]) {

#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=return

    // ------------------------------------------------------------
    // Tile buffers (BRAM)
    // ------------------------------------------------------------
    data_t A_tile[TILE_R][TILE_C];
    data_t tmp_tile[TILE_R][TILE_C];
    data_t C_tile[TILE_R][TILE_C];

#pragma HLS bind_storage variable=A_tile   type=ram_2p impl=bram
#pragma HLS bind_storage variable=tmp_tile type=ram_2p impl=bram
#pragma HLS bind_storage variable=C_tile   type=ram_2p impl=bram

#pragma HLS ARRAY_PARTITION variable=A_tile   cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=tmp_tile cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=C_tile   cyclic factor=8 dim=2

    // ------------------------------------------------------------
    // Global column sums (BRAM)
    // ------------------------------------------------------------
    data_t col_sum_global[N_COLS];
#pragma HLS ARRAY_PARTITION variable=col_sum_global cyclic factor=32 dim=1

    // Initialize global column sums
    init_global: for (int j = 0; j < N_COLS; j++) {
#pragma HLS PIPELINE II=1
        col_sum_global[j] = 0.0;
    }

    // ------------------------------------------------------------
    // PASS 1: compute global column sums
    // ------------------------------------------------------------
    tile_row1: for (int ii = 0; ii < N_ROWS; ii += TILE_R) {
        tile_col1: for (int jj = 0; jj < N_COLS; jj += TILE_C) {

            // Load tile
            load_tile1: for (int i = 0; i < TILE_R; i++) {
                for (int j = 0; j < TILE_C; j++) {
#pragma HLS PIPELINE II=1
                    A_tile[i][j] = A[ii + i][jj + j];
                }
            }

            // Normalize rows and accumulate column sums
            norm_and_accum: for (int i = 0; i < TILE_R; i++) {
#pragma HLS PIPELINE off
                data_t row_sum = 0.0;

                // Row sum
                for (int j = 0; j < TILE_C; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS unroll factor=8
                    row_sum += A_tile[i][j];
                }

                data_t recip = (data_t)1.0 / (row_sum + (data_t)1.0);

                // Normalize + accumulate into global column sum
                for (int j = 0; j < TILE_C; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS unroll factor=8
                    data_t tmp_val = A_tile[i][j] * recip;
                    tmp_tile[i][j] = tmp_val;
                    col_sum_global[jj + j] += tmp_val;
                }
            }
        }
    }

    // ------------------------------------------------------------
    // Compute global scale
    // ------------------------------------------------------------
    data_t scale_global[N_COLS];
#pragma HLS ARRAY_PARTITION variable=scale_global cyclic factor=32 dim=1

    compute_scale: for (int j = 0; j < N_COLS; j++) {
#pragma HLS PIPELINE II=1
        scale_global[j] = col_sum_global[j] / (data_t)N_ROWS;
    }

    // ------------------------------------------------------------
    // PASS 2: compute output using global scale
    // ------------------------------------------------------------
    tile_row2: for (int ii = 0; ii < N_ROWS; ii += TILE_R) {
        tile_col2: for (int jj = 0; jj < N_COLS; jj += TILE_C) {

            // Load tile again
            load_tile2: for (int i = 0; i < TILE_R; i++) {
                for (int j = 0; j < TILE_C; j++) {
#pragma HLS PIPELINE II=1
                    A_tile[i][j] = A[ii + i][jj + j];
                }
            }

            // Normalize rows and apply global scale
            norm_and_scale: for (int i = 0; i < TILE_R; i++) {
#pragma HLS PIPELINE off
                data_t row_sum = 0.0;

                for (int j = 0; j < TILE_C; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS unroll factor=8
                    row_sum += A_tile[i][j];
                }

                data_t recip = (data_t)1.0 / (row_sum + (data_t)1.0);

                for (int j = 0; j < TILE_C; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS unroll factor=8
                    data_t tmp_val = A_tile[i][j] * recip;
                    C_tile[i][j] = tmp_val * scale_global[jj + j];
                }
            }

            // Store tile
            store_tile: for (int i = 0; i < TILE_R; i++) {
                for (int j = 0; j < TILE_C; j++) {
#pragma HLS PIPELINE II=1
                    C[ii + i][jj + j] = C_tile[i][j];
                }
            }
        }
    }
}
