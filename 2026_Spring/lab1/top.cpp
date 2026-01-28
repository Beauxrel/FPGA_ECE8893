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
    // Tile loops
    // ------------------------------------------------------------
    tile_row: for (int ii = 0; ii < N_ROWS; ii += TILE_R) {
        tile_col: for (int jj = 0; jj < N_COLS; jj += TILE_C) {

#pragma HLS PIPELINE off

            // ----------------------------------------------------
            // Load tile: DRAM → BRAM
            // ----------------------------------------------------
            load_tile: for (int i = 0; i < TILE_R; i++) {
                for (int j = 0; j < TILE_C; j++) {
#pragma HLS PIPELINE II=1
                    A_tile[i][j] = A[ii + i][jj + j];
                }
            }

            // ----------------------------------------------------
            // Phase 1: Row-wise normalization (tile-local)
            // ----------------------------------------------------
            norm_rows: for (int i = 0; i < TILE_R; i++) {
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
                    tmp_tile[i][j] = A_tile[i][j] * recip;
                }
            }

            // ----------------------------------------------------
            // Phase 2: Column-wise scaling (tile-local)
            // ----------------------------------------------------
            data_t col_sum[TILE_C];
            data_t scale[TILE_C];
#pragma HLS ARRAY_PARTITION variable=col_sum complete
#pragma HLS ARRAY_PARTITION variable=scale   complete

            // init
            init_cols: for (int j = 0; j < TILE_C; j++) {
#pragma HLS PIPELINE II=1
                col_sum[j] = 0.0;
            }

            // accumulate
            accum_cols: for (int i = 0; i < TILE_R; i++) {
                for (int j = 0; j < TILE_C; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS unroll factor=8
                    col_sum[j] += tmp_tile[i][j];
                }
            }

            // compute scale
            scale_cols: for (int j = 0; j < TILE_C; j++) {
#pragma HLS PIPELINE II=1
                scale[j] = col_sum[j] / (data_t)TILE_R;
            }

            // apply scale
            apply_scale: for (int i = 0; i < TILE_R; i++) {
                for (int j = 0; j < TILE_C; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS unroll factor=8
                    C_tile[i][j] = tmp_tile[i][j] * scale[j];
                }
            }

            // ----------------------------------------------------
            // Store tile: BRAM → DRAM
            // ----------------------------------------------------
            store_tile: for (int i = 0; i < TILE_R; i++) {
                for (int j = 0; j < TILE_C; j++) {
#pragma HLS PIPELINE II=1
                    C[ii + i][jj + j] = C_tile[i][j];
                }
            }
        }
    }
}
