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
#pragma HLS array_partition variable=buf cyclic factor=16 dim=3

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

    // =========================================================
    // Time stepping with line buffer
    // =========================================================
    TIME: for (int t = 0; t < TSTEPS; t++) {
        const int rd = t & 1;
        const int wr = 1 - rd;

        // Line buffer: 3 rows cached locally
        static data_t lb[3][NY];
#pragma HLS array_partition variable=lb complete dim=1
#pragma HLS array_partition variable=lb cyclic factor=16 dim=2

        // -------------------------------------------------------
        // Prime line buffer: load rows 0 and 1
        // -------------------------------------------------------
        PRIME: for (int j = 0; j < NY; j++) {
#pragma HLS pipeline II=1
            lb[0][j] = buf[rd][0][j];
            lb[1][j] = buf[rd][1][j];
        }

        // -------------------------------------------------------
        // Copy top boundary row unchanged
        // -------------------------------------------------------
        TOP_BOUNDARY: for (int j = 0; j < NY; j++) {
#pragma HLS pipeline II=1
            buf[wr][0][j] = buf[rd][0][j];
        }

        // -------------------------------------------------------
        // Main stencil loop over interior rows
        // -------------------------------------------------------
        STENCIL_I: for (int i = 1; i < NX-1; i++) {

            // Load row i+1 into lb[2]
            LOAD_ROW: for (int j = 0; j < NY; j++) {
#pragma HLS pipeline II=1
                lb[2][j] = buf[rd][i+1][j];
            }

            // Left boundary
            buf[wr][i][0] = buf[rd][i][0];

            // Interior columns
            STENCIL_J: for (int j = 1; j < NY-1; j++) {
#pragma HLS pipeline II=1
#pragma HLS dependence variable=lb inter false
                acc_t sum_axis = (acc_t)lb[0][j]   + (acc_t)lb[2][j]   +
                                 (acc_t)lb[1][j-1] + (acc_t)lb[1][j+1];

                acc_t sum_diag = (acc_t)lb[0][j-1] + (acc_t)lb[0][j+1] +
                                 (acc_t)lb[2][j-1] + (acc_t)lb[2][j+1];

                acc_t center   = (acc_t)lb[1][j];

                buf[wr][i][j]  = (data_t)(
                                     (acc_t)wc * center +
                                     (acc_t)wa * sum_axis +
                                     (acc_t)wd * sum_diag
                                 );
            }

            // Right boundary
            buf[wr][i][NY-1] = buf[rd][i][NY-1];

            // Slide line buffer: lb[0] <- lb[1], lb[1] <- lb[2]
            SLIDE: for (int j = 0; j < NY; j++) {
#pragma HLS pipeline II=1
                lb[0][j] = lb[1][j];
                lb[1][j] = lb[2][j];
            }
        }

        // -------------------------------------------------------
        // Copy bottom boundary row unchanged
        // -------------------------------------------------------
        BOT_BOUNDARY: for (int j = 0; j < NY; j++) {
#pragma HLS pipeline II=1
            buf[wr][NX-1][j] = buf[rd][NX-1][j];
        }
    }

    // =========================================================
    // Pack output from last written buffer
    // =========================================================
    const int final_buf = TSTEPS & 1;

    OUT_I: for (int i = 0; i < NX*NY/16; i++) {
#pragma HLS pipeline II=1
        ap_uint<512> chunk = 0;
        for (int k = 0; k < 16; k++) {
#pragma HLS unroll
            int idx = i * 16 + k;
            data_t val = buf[final_buf][idx / NY][idx % NY];
            ap_uint<32> packed = 0;
            packed.range(23, 0) = val.range(23, 0);
            chunk.range(32*k+31, 32*k) = packed;
        }
        A_OUT_WIDE[i] = chunk;
    }
}