#include "dcl.h"

// Ping-pong buffers split apart so HLS sees static indexing
static data_t buf0[NX][NY];
static data_t buf1[NX][NY];
#pragma HLS array_partition variable = buf0 cyclic factor = 16 dim = 2
#pragma HLS array_partition variable = buf1 cyclic factor = 16 dim = 2

static void stencil_pass(
    data_t rd[NX][NY],
    data_t wr[NX][NY])
{
#pragma HLS inline off
#pragma HLS array_partition variable = rd cyclic factor = 16 dim = 2
#pragma HLS array_partition variable = wr cyclic factor = 16 dim = 2

    const acc_t wc = (acc_t)0.50;
    const acc_t wa = (acc_t)0.10;
    const acc_t wd = (acc_t)0.025;

    // Row cache: keep 3 rows in registers/BRAMs for sliding window
    static data_t row_prev[NY];
    static data_t row_curr[NY];
    static data_t row_next[NY];
#pragma HLS array_partition variable = row_prev cyclic factor = 16 dim = 1
#pragma HLS array_partition variable = row_curr cyclic factor = 16 dim = 1
#pragma HLS array_partition variable = row_next cyclic factor = 16 dim = 1

// Preload first two rows
PRELOAD:
    for (int j = 0; j < NY; j++)
    {
#pragma HLS pipeline II = 1
        row_curr[j] = rd[0][j];
        row_next[j] = rd[1][j];
    }

STENCIL_I:
    for (int i = 0; i < NX; i++)
    {
    // Rotate row cache
    ROTATE:
        for (int j = 0; j < NY; j++)
        {
#pragma HLS pipeline II = 1
            row_prev[j] = row_curr[j];
            row_curr[j] = row_next[j];
            row_next[j] = (i + 2 < NX) ? rd[i + 2][j] : rd[i][j]; // clamp
        }

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

    ap_uint<512> *A_IN_WIDE = (ap_uint<512> *)A_in;
    ap_uint<512> *A_OUT_WIDE = (ap_uint<512> *)A_out;

// =========================================================
// Unpack input into buf0
// =========================================================
INIT_I:
    for (int i = 0; i < NX * NY / 16; i++)
    {
#pragma HLS pipeline II = 1
        ap_uint<512> chunk = A_IN_WIDE[i];
        for (int k = 0; k < 16; k++)
        {
#pragma HLS unroll
            int idx = i * 16 + k;
            ap_uint<32> tmp = chunk.range(32 * k + 31, 32 * k);
            data_t val;
            val.range(23, 0) = tmp.range(23, 0);
            buf0[idx / NY][idx % NY] = val;
        }
    }

// =========================================================
// Time stepping — statically alternating, no runtime rd/wr
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
    // Pack output
    // =========================================================
    data_t(*final_buf)[NY] = (TSTEPS & 1) ? buf1 : buf0;

OUT_I:
    for (int i = 0; i < NX * NY / 16; i++)
    {
#pragma HLS pipeline II = 1
        ap_uint<512> chunk = 0;
        for (int k = 0; k < 16; k++)
        {
#pragma HLS unroll
            int idx = i * 16 + k;
            data_t val = final_buf[idx / NY][idx % NY];
            ap_uint<32> packed = 0;
            packed.range(23, 0) = val.range(23, 0);
            chunk.range(32 * k + 31, 32 * k) = packed;
        }
        A_OUT_WIDE[i] = chunk;
    }
}