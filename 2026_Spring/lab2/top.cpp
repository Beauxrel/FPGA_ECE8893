#include "dcl.h"

static data_t buf0[NX][NY];
static data_t buf1[NX][NY];
#pragma HLS array_partition variable = buf0 cyclic factor = 16 dim = 2
#pragma HLS array_partition variable = buf1 cyclic factor = 16 dim = 2

// Row cache arrays at file scope to avoid stale static-local state
static data_t row_prev[NY];
static data_t row_curr[NY];
static data_t row_next[NY];
#pragma HLS array_partition variable = row_prev cyclic factor = 16 dim = 1
#pragma HLS array_partition variable = row_curr cyclic factor = 16 dim = 1
#pragma HLS array_partition variable = row_next cyclic factor = 16 dim = 1

static void stencil_pass(data_t rd[NX][NY], data_t wr[NX][NY])
{
#pragma HLS inline off
#pragma HLS array_partition variable = rd cyclic factor = 16 dim = 2
#pragma HLS array_partition variable = wr cyclic factor = 16 dim = 2

    const acc_t wc = (acc_t)0.50;
    const acc_t wa = (acc_t)0.10;
    const acc_t wd = (acc_t)0.025;

// Preload row 0 and row 1 before the main loop
PRELOAD_CURR:
    for (int j = 0; j < NY; j++)
    {
#pragma HLS pipeline II = 1
        row_curr[j] = rd[0][j];
        row_next[j] = rd[1][j];
        row_prev[j] = rd[0][j]; // prev for i=0 is unused (boundary), init safely
    }

STENCIL_I:
    for (int i = 0; i < NX; i++)
    {
#pragma HLS loop_tripcount min = NX max = NX

        // First: load next row into row_next (i+2), or clamp at last row
        int next_row = (i + 2 < NX) ? (i + 2) : (NX - 1);
    LOAD_NEXT:
        for (int j = 0; j < NY; j++)
        {
#pragma HLS pipeline II = 1
            // Rotate: prev <- curr, curr <- next, next <- rd[next_row]
            row_prev[j] = row_curr[j];
            row_curr[j] = row_next[j];
            row_next[j] = rd[next_row][j];
        }

    // Now apply stencil using the rotated cache
    STENCIL_J:
        for (int j = 0; j < NY; j++)
        {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = row_prev inter false
#pragma HLS dependence variable = row_curr inter false
#pragma HLS dependence variable = row_next inter false
            if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1)
            {
                wr[i][j] = row_curr[j];
            }
            else
            {
                acc_t sum_axis =
                    (acc_t)row_prev[j] + (acc_t)row_next[j] +
                    (acc_t)row_curr[j - 1] + (acc_t)row_curr[j + 1];
                acc_t sum_diag =
                    (acc_t)row_prev[j - 1] + (acc_t)row_prev[j + 1] +
                    (acc_t)row_next[j - 1] + (acc_t)row_next[j + 1];
                acc_t center = (acc_t)row_curr[j];
                wr[i][j] = (data_t)(wc * center +
                                    wa * sum_axis +
                                    wd * sum_diag);
            }
        }
    }
}

void top_kernel(const data_t A_in[NX][NY], data_t A_out[NX][NY])
{
#pragma HLS interface m_axi port = A_in offset = slave bundle = gmem0 depth = 16384
#pragma HLS interface m_axi port = A_out offset = slave bundle = gmem1 depth = 16384
#pragma HLS interface s_axilite port = return

// =========================================================
// Direct copy — no bit-mangling, avoids the 24-bit truncation bug
// =========================================================
INIT_I:
    for (int i = 0; i < NX; i++)
    {
#pragma HLS loop_tripcount min = NX max = NX
    INIT_J:
        for (int j = 0; j < NY; j++)
        {
#pragma HLS pipeline II = 1
            buf0[i][j] = A_in[i][j];
        }
    }

// =========================================================
// Time stepping
// =========================================================
TIME:
    for (int t = 0; t < TSTEPS; t++)
    {
#pragma HLS loop_tripcount min = TSTEPS max = TSTEPS
        if (t & 1)
        {
            stencil_pass(buf1, buf0);
        }
        else
        {
            stencil_pass(buf0, buf1);
        }
    }

    // =========================================================
    // Write output from whichever buffer was written last
    // =========================================================
    data_t(*final_buf)[NY] = (TSTEPS & 1) ? buf1 : buf0;

OUT_I:
    for (int i = 0; i < NX; i++)
    {
#pragma HLS loop_tripcount min = NX max = NX
    OUT_J:
        for (int j = 0; j < NY; j++)
        {
#pragma HLS pipeline II = 1
            A_out[i][j] = final_buf[i][j];
        }
    }
}