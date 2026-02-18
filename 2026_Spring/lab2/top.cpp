#include "dcl.h"

void top_kernel(const data_t A_in[NX][NY], data_t A_out[NX][NY]) {
#pragma HLS interface m_axi port=A_in  offset=slave bundle=gmem0 depth=16384
#pragma HLS interface m_axi port=A_out offset=slave bundle=gmem1 depth=16384
#pragma HLS interface s_axilite port=return

    static data_t cur[NX][NY];
    static data_t nxt[NX][NY];
#pragma HLS array_partition variable=cur cyclic factor=2 dim=2
#pragma HLS array_partition variable=nxt cyclic factor=2 dim=2

    const data_t wc = (data_t)0.50;
    const data_t wa = (data_t)0.10;
    const data_t wd = (data_t)0.025;

    // Copy input into cur
    INIT_I: for (int i = 0; i < NX; i++) {
        INIT_J: for (int j = 0; j < NY; j++) {
#pragma HLS pipeline II=1
            cur[i][j] = A_in[i][j];
        }
    }

    // Time stepping
    TIME: for (int t = 0; t < TSTEPS; t++) {

        // Update all points (boundaries copied, interior computed)
        STENCIL_I: for (int i = 0; i < NX; i++) {
            STENCIL_J: for (int j = 0; j < NY; j++) {
#pragma HLS pipeline II=1
#pragma HLS dependence variable=cur inter false
                if (i == 0 || i == NX-1 || j == 0 || j == NY-1) {
                    nxt[i][j] = cur[i][j];
                } else {
                    acc_t sum_axis =
                        (acc_t)cur[i-1][j] + (acc_t)cur[i+1][j] +
                        (acc_t)cur[i][j-1] + (acc_t)cur[i][j+1];
                    acc_t sum_diag =
                        (acc_t)cur[i-1][j-1] + (acc_t)cur[i-1][j+1] +
                        (acc_t)cur[i+1][j-1] + (acc_t)cur[i+1][j+1];
                    acc_t center = (acc_t)cur[i][j];
                    nxt[i][j] = (data_t)((acc_t)wc * center +
                                          (acc_t)wa * sum_axis +
                                          (acc_t)wd * sum_diag);
                }
            }
        }

        // Swap nxt -> cur
        SWAP_I: for (int i = 0; i < NX; i++) {
            SWAP_J: for (int j = 0; j < NY; j++) {
#pragma HLS pipeline II=1
                cur[i][j] = nxt[i][j];
            }
        }
    }

    // Write output
    OUT_I: for (int i = 0; i < NX; i++) {
        OUT_J: for (int j = 0; j < NY; j++) {
#pragma HLS pipeline II=1
            A_out[i][j] = cur[i][j];
        }
    }
}