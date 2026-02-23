#include "dcl.h"

static data_t buf0[NX][NY];
static data_t buf1[NX][NY];

static void stencil_pass(data_t rd[NX][NY], data_t wr[NX][NY])
{
#pragma HLS inline off
#pragma HLS array_partition variable = rd cyclic factor = 16 dim = 2
#pragma HLS array_partition variable = wr cyclic factor = 16 dim = 2

    const acc_t wc = (acc_t)0.50;
    const acc_t wa = (acc_t)0.10;
    const acc_t wd = (acc_t)0.025;

    data_t lb0[NY], lb1[NY], lb2[NY];
#pragma HLS array_partition variable = lb0 cyclic factor = 16 dim = 1
#pragma HLS array_partition variable = lb1 cyclic factor = 16 dim = 1
#pragma HLS array_partition variable = lb2 cyclic factor = 16 dim = 1

// After PRIME: lb0=row0(prev, unused for i=0 boundary)
//              lb1=row0(curr for i=0)
//              lb2=row1(next for i=0)
// This is exactly correct for computing i=0 immediately.
PRIME:
    for (int j = 0; j < NY; j++)
    {
#pragma HLS pipeline II = 1
        lb0[j] = rd[0][j];
        lb1[j] = rd[0][j];
        lb2[j] = rd[1][j];
    }

STENCIL_I:
    for (int i = 0; i < NX; i++)
    {
#pragma HLS loop_tripcount min = NX max = NX

    // 1) COMPUTE stencil for row i using current lb0/lb1/lb2
    STENCIL_J:
        for (int j = 0; j < NY; j++)
        {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = lb0 inter false
#pragma HLS dependence variable = lb1 inter false
#pragma HLS dependence variable = lb2 inter false
            if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1)
            {
                wr[i][j] = lb1[j];
            }
            else
            {
                acc_t sum_axis =
                    (acc_t)lb0[j] + (acc_t)lb2[j] +
                    (acc_t)lb1[j - 1] + (acc_t)lb1[j + 1];
                acc_t sum_diag =
                    (acc_t)lb0[j - 1] + (acc_t)lb0[j + 1] +
                    (acc_t)lb2[j - 1] + (acc_t)lb2[j + 1];
                wr[i][j] = (data_t)(wc * (acc_t)lb1[j] + wa * sum_axis + wd * sum_diag);
            }
        }

        // 2) ROTATE after compute: prepare lb0/lb1/lb2 for row i+1
        //    lb0 <- lb1 (was curr, becomes prev)
        //    lb1 <- lb2 (was next, becomes curr)
        //    lb2 <- rd[i+2] (load the new next row)
        int load_row = (i + 2 < NX) ? (i + 2) : (NX - 1);
    ROTATE:
        for (int j = 0; j < NY; j++)
        {
#pragma HLS pipeline II = 1
            lb0[j] = lb1[j];
            lb1[j] = lb2[j];
            lb2[j] = rd[load_row][j];
        }
    }
}

void top_kernel(const data_t A_in[NX][NY], data_t A_out[NX][NY])
{
#pragma HLS interface m_axi port = A_in offset = slave bundle = gmem0 depth = 16384
#pragma HLS interface m_axi port = A_out offset = slave bundle = gmem1 depth = 16384
#pragma HLS interface s_axilite port = return
#pragma HLS array_partition variable = buf0 cyclic factor = 16 dim = 2
#pragma HLS array_partition variable = buf1 cyclic factor = 16 dim = 2

INIT_I:
    for (int i = 0; i < NX; i++)
    {
    INIT_J:
        for (int j = 0; j < NY; j++)
        {
#pragma HLS pipeline II = 1
            buf0[i][j] = A_in[i][j];
        }
    }

TIME:
    for (int t = 0; t < TSTEPS; t++)
    {
#pragma HLS loop_tripcount min = TSTEPS max = TSTEPS
        if (t & 1)
            stencil_pass(buf1, buf0);
        else
            stencil_pass(buf0, buf1);
    }

    data_t(*final_buf)[NY] = (TSTEPS & 1) ? buf1 : buf0;

OUT_I:
    for (int i = 0; i < NX; i++)
    {
    OUT_J:
        for (int j = 0; j < NY; j++)
        {
#pragma HLS pipeline II = 1
            A_out[i][j] = final_buf[i][j];
        }
    }
}