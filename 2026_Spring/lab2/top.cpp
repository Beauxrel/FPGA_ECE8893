#include "dcl.h"
#include "hls_math.h"

static data_t ping[NX][NY];
static data_t pong[NX][NY];

#pragma HLS array_partition variable=ping cyclic factor=2 dim=2
#pragma HLS array_partition variable=pong cyclic factor=2 dim=2

static void stencil_pass(data_t rd[NX][NY], data_t wr[NX][NY]) {
#pragma HLS inline off
#pragma HLS array_partition variable=rd cyclic factor=2 dim=2
#pragma HLS array_partition variable=wr cyclic factor=2 dim=2

    const acc_t wc = (acc_t)0.50;
    const acc_t wa = (acc_t)0.10;
    const acc_t wd = (acc_t)0.025;

    // Three line buffers — only 3*NY elements, fits in registers/LUTRAM
    data_t lb0[NY], lb1[NY], lb2[NY];
#pragma HLS array_partition variable=lb0 cyclic factor=2 dim=1
#pragma HLS array_partition variable=lb1 cyclic factor=2 dim=1
#pragma HLS array_partition variable=lb2 cyclic factor=2 dim=1

    // Prime: load rows 0 and 1
PRIME:
    for (int j = 0; j < NY; j++) {
#pragma HLS pipeline II=1
        lb0[j] = rd[0][j];
        lb1[j] = rd[0][j];  // row -1 clamp = row 0
        lb2[j] = rd[1][j];
    }

STENCIL_I:
    for (int i = 0; i < NX; i++) {
        // Sliding window over columns: need j-1, j, j+1
        // Use registers for the 3x3 window values
        data_t r00, r01, r02;  // lb0[j-1], lb0[j], lb0[j+1]
        data_t r10, r11, r12;  // lb1[j-1], lb1[j], lb1[j+1]
        data_t r20, r21, r22;  // lb2[j-1], lb2[j], lb2[j+1]

        // Preload column -1 (clamped to 0)
        r00 = lb0[0]; r10 = lb1[0]; r20 = lb2[0];
        r01 = lb0[0]; r11 = lb1[0]; r21 = lb2[0];

    STENCIL_J:
        for (int j = 0; j < NY; j++) {
#pragma HLS pipeline II=1
            // Shift window left
            r00 = r01; r01 = r02;
            r10 = r11; r11 = r12;
            r20 = r21; r21 = r22;

            // Load next column (clamp at boundary)
            int jn = (j + 1 < NY) ? j + 1 : NY - 1;
            r02 = lb0[jn];
            r12 = lb1[jn];
            r22 = lb2[jn];

            if (i == 0 || i == NX-1 || j == 0 || j == NY-1) {
                wr[i][j] = r11;
            } else {
                acc_t sum_axis = (acc_t)r01 + (acc_t)r21 +
                                 (acc_t)r10 + (acc_t)r12;
                acc_t sum_diag = (acc_t)r00 + (acc_t)r02 +
                                 (acc_t)r20 + (acc_t)r22;
                wr[i][j] = (data_t)(wc * (acc_t)r11 +
                                    wa * sum_axis +
                                    wd * sum_diag);
            }
        }

        // Rotate line buffers: lb0 <- lb1 <- lb2 <- next row
        int load_row = (i + 2 < NX) ? i + 2 : NX - 1;
    ROTATE:
        for (int j = 0; j < NY; j++) {
#pragma HLS pipeline II=1
            lb0[j] = lb1[j];
            lb1[j] = lb2[j];
            lb2[j] = rd[load_row][j];
        }
    }
}

void top_kernel(const data_t A_in[NX][NY], data_t A_out[NX][NY]) {
#pragma HLS interface m_axi port=A_in  offset=slave bundle=gmem0 depth=16384
#pragma HLS interface m_axi port=A_out offset=slave bundle=gmem1 depth=16384
#pragma HLS interface s_axilite port=return

INIT_I:
    for (int i = 0; i < NX; i++) {
    INIT_J:
        for (int j = 0; j < NY; j++) {
#pragma HLS pipeline II=1
            ping[i][j] = A_in[i][j];
        }
    }

TIME:
    for (int t = 0; t < TSTEPS; t++) {
#pragma HLS loop_tripcount min=TSTEPS max=TSTEPS
        if (t & 1) stencil_pass(pong, ping);
        else       stencil_pass(ping, pong);
    }

    data_t (*final_buf)[NY] = (TSTEPS & 1) ? pong : ping;

OUT_I:
    for (int i = 0; i < NX; i++) {
    OUT_J:
        for (int j = 0; j < NY; j++) {
#pragma HLS pipeline II=1
            A_out[i][j] = final_buf[i][j];
        }
    }
}