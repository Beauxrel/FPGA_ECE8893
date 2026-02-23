#include "dcl.h"

static void stencil_pass(data_t rd[NX][NY], data_t wr[NX][NY]) {
#pragma HLS inline off
#pragma HLS array_partition variable=rd cyclic factor=2 dim=2
#pragma HLS array_partition variable=wr cyclic factor=2 dim=2

    const acc_t wc = (acc_t)0.50;
    const acc_t wa = (acc_t)0.10;
    const acc_t wd = (acc_t)0.025;

    data_t lb0[NY], lb1[NY], lb2[NY];
#pragma HLS array_partition variable=lb0 cyclic factor=2 dim=1
#pragma HLS array_partition variable=lb1 cyclic factor=2 dim=1
#pragma HLS array_partition variable=lb2 cyclic factor=2 dim=1

PRIME:
    for (int j = 0; j < NY; j++) {
#pragma HLS pipeline II=1
        lb0[j] = rd[0][j];
        lb1[j] = rd[0][j];
        lb2[j] = rd[1][j];
    }

STENCIL_I:
    for (int i = 0; i < NX; i++) {

        data_t r00 = lb0[0], r01 = lb0[0], r02 = lb0[1];
        data_t r10 = lb1[0], r11 = lb1[0], r12 = lb1[1];
        data_t r20 = lb2[0], r21 = lb2[0], r22 = lb2[1];

    STENCIL_J:
        for (int j = 0; j < NY; j++) {
#pragma HLS pipeline II=1
#pragma HLS dependence variable=lb0 inter false
#pragma HLS dependence variable=lb1 inter false
#pragma HLS dependence variable=lb2 inter false

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

            // Slide window left, load j+2
            r00 = r01; r01 = r02;
            r10 = r11; r11 = r12;
            r20 = r21; r21 = r22;

            int jn = (j + 2 < NY) ? j + 2 : NY - 1;
            r02 = lb0[jn];
            r12 = lb1[jn];
            r22 = lb2[jn];
        }

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

    ap_uint<512> *A_IN_WIDE  = (ap_uint<512> *)A_in;
    ap_uint<512> *A_OUT_WIDE = (ap_uint<512> *)A_out;

    static data_t buf0[NX][NY];
    static data_t buf1[NX][NY];
#pragma HLS array_partition variable=buf0 cyclic factor=2 dim=2
#pragma HLS array_partition variable=buf1 cyclic factor=2 dim=2

    // =========================================================
    // Unpack input into buf0
    // =========================================================
INIT_I:
    for (int i = 0; i < NX*NY/16; i++) {
#pragma HLS pipeline II=1
        ap_uint<512> chunk = A_IN_WIDE[i];
        for (int k = 0; k < 16; k++) {
#pragma HLS unroll
            int idx = i * 16 + k;
            ap_uint<32> tmp = chunk.range(32*k+31, 32*k);
            data_t val;
            val.range(23, 0) = tmp.range(23, 0);
            buf0[idx / NY][idx % NY] = val;
        }
    }

    // =========================================================
    // Time stepping with line buffer stencil
    // =========================================================
TIME:
    for (int t = 0; t < TSTEPS; t++) {
#pragma HLS loop_tripcount min=TSTEPS max=TSTEPS
        if (t & 1) stencil_pass(buf1, buf0);
        else       stencil_pass(buf0, buf1);
    }

    // =========================================================
    // Pack output from last written buffer
    // =========================================================
    const int use_buf1 = TSTEPS & 1;  // if odd steps, last write was to buf1

OUT_I:
    for (int i = 0; i < NX*NY/16; i++) {
#pragma HLS pipeline II=1
        ap_uint<512> chunk = 0;
        for (int k = 0; k < 16; k++) {
#pragma HLS unroll
            int idx = i * 16 + k;
            int row = idx / NY;
            int col = idx % NY;
            data_t val = use_buf1 ? buf1[row][col] : buf0[row][col];
            ap_uint<32> packed = 0;
            packed.range(23, 0) = val.range(23, 0);
            chunk.range(32*k+31, 32*k) = packed;
        }
        A_OUT_WIDE[i] = chunk;
    }
}