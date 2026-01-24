#include "dcl.h"

// Baseline implementation for HLS with Phase 2 access changed to row-wise apply.
// (No reciprocal-multiply optimizations; divisions remain as requested.)
void top_kernel(data_t A[N_ROWS][N_COLS],
                data_t C[N_ROWS][N_COLS]) {
    // Intermediate buffer for row-normalized values
    static data_t tmp[N_ROWS][N_COLS];
    #pragma HLS ARRAY_PARTITION variable=tmp cyclic factor=16 dim=2
    #pragma HLS ARRAY_PARTITION variable=A   cyclic factor=16 dim=2
    #pragma HLS ARRAY_PARTITION variable=C   cyclic factor=16 dim=2

    // Row buffer for Phase 1 (read A once per element)
    data_t rowbuf[N_COLS];
    #pragma HLS ARRAY_PARTITION variable=rowbuf cyclic factor=16 dim=1

    // Column scales for Phase 2
    static data_t scale_arr[N_COLS];
    #pragma HLS ARRAY_PARTITION variable=scale_arr cyclic factor=16 dim=1

    // -------------------------
    // Phase 1: Row-wise normalization
    // -------------------------
    phase_1: for (int i = 0; i < N_ROWS; i++) {
        // NOTE: loop_flatten is often not helpful with aggressive unroll,
        // but keeping your style consistent is fine for experimentation.
        // #pragma HLS loop_flatten

        acc_t row_sum = 0;

        // Compute row sum + buffer row
        compute_row: for (int j = 0; j < N_COLS; j++) {
            #pragma HLS PIPELINE II=1
            data_t v = A[i][j];
            rowbuf[j] = v;
            row_sum += (acc_t)v;
        }

        // Avoid division by zero, add small bias
        acc_t denom = row_sum + (acc_t)1.0;

        // Normalize each element in the row (division kept)
        norm_row: for (int j = 0; j < N_COLS; j++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=16
            tmp[i][j] = (data_t)((acc_t)rowbuf[j] / denom);
        }
    }

    // -------------------------
    // Phase 2: Compute per-column scale (still reads tmp column-wise)
    // -------------------------
    col_scales: for (int j = 0; j < N_COLS; j++) {
        // #pragma HLS loop_flatten
        acc_t col_sum = 0;

        // Compute column sum of normalized values
        col_sum_loop: for (int i = 0; i < N_ROWS; i++) {
            #pragma HLS PIPELINE II=1
            col_sum += (acc_t)tmp[i][j];
        }

        // Compute average as scale (division kept)
        scale_arr[j] = (data_t)(col_sum / (acc_t)N_ROWS);
    }

    // -------------------------
    // Phase 2: Apply scaling ROW-WISE (this is the access change)
    // tmp is now read in row-major order: tmp[i][j] with j as inner loop.
    // -------------------------
    apply_scales: for (int i = 0; i < N_ROWS; i++) {
        apply_row: for (int j = 0; j < N_COLS; j++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=16
            C[i][j] = (data_t)((acc_t)tmp[i][j] * (acc_t)scale_arr[j]);
        }
    }
}
