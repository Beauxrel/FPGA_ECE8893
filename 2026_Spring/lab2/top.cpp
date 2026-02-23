#include "dcl.h"

void top_kernel(const data_t A_in[NX][NY], data_t A_out[NX][NY]) {
#pragma HLS interface m_axi port=A_in  offset=slave bundle=gmem0 depth=16384
#pragma HLS interface m_axi port=A_out offset=slave bundle=gmem1 depth=16384
#pragma HLS interface s_axilite port=return

    ap_uint<512> *A_IN_WIDE  = (ap_uint<512> *)A_in;
    ap_uint<512> *A_OUT_WIDE = (ap_uint<512> *)A_out;

    const data_t wc = (data_t)0.50;
    const data_t wa = (data_t)0.10;
    const data_t wd = (data_t)0.025;

    static data_t buf[2][NX][NY];
#pragma HLS array_partition variable=buf cyclic factor=32 dim=3

    // =========================================================
    // Unpack input into buf[0]
    // =========================================================
    INIT_I: for (int i = 0; i < NX*NY/16; i++) {
#pragma HLS pipeline II=1
        ap_uint<512> chunk = A_IN_WIDE[i];
        for (int k = 0; k < 16; k++) {
#pragma HLS unroll
            int idx = i * 16 + k;
            ap_uint<32> tmp = chunk.range(32*k+31, 32*k);
            data_t val;
            val.range(23, 0) = tmp.range(23, 0);
            buf[0][idx / NY][idx % NY] = val;
        }
    }

    // Time stepping
    TIME: for (int t = 0; t < TSTEPS; t++) {
        int rd = t & 1;       // alternates 0, 1, 0, 1...
        int wr = 1 - rd;      // alternates 1, 0, 1, 0...

        STENCIL_I: for (int i = 0; i < NX; i++) {
            STENCIL_J: for (int j = 0; j < NY; j++) {
                #pragma HLS pipeline II=1
                #pragma HLS unroll factor = 16
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

    // =========================================================
    // Pack output from last written buffer
    // =========================================================
    const int final_buf = TSTEPS & 1;

    OUT_I: for (int i = 0; i < NX*NY/16; i++) {
#pragma HLS pipeline II=1
        ap_uint<512> chunk = 0;
        for (int k = 0; k < 16; k++) {
#pragma HLS unroll factor = 8
            int idx = i * 16 + k;
            data_t val = buf[final_buf][idx / NY][idx % NY];
            ap_uint<32> packed = 0;
            packed.range(23, 0) = val.range(23, 0);
            chunk.range(32*k+31, 32*k) = packed;
        }
        A_OUT_WIDE[i] = chunk;
    }
}