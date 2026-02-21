#include "dcl.h"

void top_kernel(const data_t A_in[NX][NY], data_t A_out[NX][NY]) {
#pragma HLS interface m_axi port=A_in  offset=slave bundle=gmem0 depth=16384
#pragma HLS interface m_axi port=A_out offset=slave bundle=gmem1 depth=16384
#pragma HLS interface s_axilite port=return

//     static data_t cur[NX][NY];
//     static data_t nxt[NX][NY];
// #pragma HLS array_partition variable=cur cyclic factor=2 dim=2
// #pragma HLS array_partition variable=nxt cyclic factor=2 dim=2

    const data_t wc = (data_t)0.50;
    const data_t wa = (data_t)0.10;
    const data_t wd = (data_t)0.025;

static data_t buf[2][NX][NY];
#pragma HLS array_partition variable=buf cyclic factor=2 dim=3

    // Copy input into buf[0]
    INIT_I: for (int i = 0; i < NX; i++) {
        INIT_J: for (int j = 0; j < NY; j++) {
            #pragma HLS pipeline II=1
            buf[0][i][j] = A_in[i][j];
        }
    }

    // Time stepping
    TIME: for (int t = 0; t < TSTEPS; t++) {
        int rd = t & 1;       // alternates 0, 1, 0, 1...
        int wr = 1 - rd;      // alternates 1, 0, 1, 0...

        STENCIL_I: for (int i = 0; i < NX; i++) {
            STENCIL_J: for (int j = 0; j < NY; j++) {
                #pragma HLS pipeline II=1
                #pragma HLS dependence variable=buf inter false
                if (i == 0 || i == NX-1 || j == 0 || j == NY-1) {
                    buf[wr][i][j] = buf[rd][i][j];
                } else {
                    acc_t sum_axis =
                        (acc_t)buf[rd][i-1][j] + (acc_t)buf[rd][i+1][j] +
                        (acc_t)buf[rd][i][j-1] + (acc_t)buf[rd][i][j+1];
                    acc_t sum_diag =
                        (acc_t)buf[rd][i-1][j-1] + (acc_t)buf[rd][i-1][j+1] +
                        (acc_t)buf[rd][i+1][j-1] + (acc_t)buf[rd][i+1][j+1];
                    acc_t center = (acc_t)buf[rd][i][j];
                    buf[wr][i][j] = (data_t)(
                        (acc_t)wc * center +
                        (acc_t)wa * sum_axis +
                        (acc_t)wd * sum_diag);
                }
            }
        }
        // No swap loop needed — rd/wr flip for free next iteration
    }

    // Write output — read from whichever buffer was last written
    int final_buf = TSTEPS & 1;
    OUT_I: for (int i = 0; i < NX; i++) {
        OUT_J: for (int j = 0; j < NY; j++) {
            #pragma HLS pipeline II=1
            A_out[i][j] = buf[final_buf][i][j];
        }
    }