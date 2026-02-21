#include "dcl.h"

void top_kernel(const ap_uint<512> A_in[NX*NY/16], ap_uint<512> A_out[NX*NY/16]) {
#pragma HLS interface m_axi port=A_in  offset=slave bundle=gmem0 depth=1024
#pragma HLS interface m_axi port=A_out offset=slave bundle=gmem1 depth=1024
#pragma HLS interface s_axilite port=return

    const data_t wc = (data_t)0.50;
    const data_t wa = (data_t)0.10;
    const data_t wd = (data_t)0.025;

    static data_t buf[2][NX][NY];
#pragma HLS array_partition variable=buf cyclic factor=16 dim=3

    // Unpack input into buf[0]
INIT_I: for (int i = 0; i < NX*NY/16; i++) {
#pragma HLS pipeline II=1
        ap_uint<512> chunk = A_in[i];
        for (int k = 0; k < 16; k++) {
#pragma HLS unroll
            int idx = i * 16 + k;
            ap_uint<32> tmp = chunk.range(32*k+31, 32*k);
            buf[0][idx / NY][idx % NY] = *reinterpret_cast<data_t*>(&tmp);
        }
    }

    // Time stepping
TIME: for (int t = 0; t < TSTEPS; t++) {
        const int rd = t & 1;
        const int wr = 1 - rd;

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
    }

    // Pack output from last written buffer
    const int final_buf = TSTEPS & 1;
OUT_I: for (int i = 0; i < NX*NY/16; i++) {
#pragma HLS pipeline II=1
        ap_uint<512> chunk = 0;
        for (int k = 0; k < 16; k++) {
#pragma HLS unroll
            int idx = i * 16 + k;
            data_t val = buf[final_buf][idx / NY][idx % NY];
            chunk.range(32*k+31, 32*k) = *reinterpret_cast<ap_uint<32>*>(&val);
        }
        A_out[i] = chunk;
    }
}