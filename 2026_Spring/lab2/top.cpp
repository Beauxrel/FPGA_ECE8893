#include "dcl.h"
#include "hls_math.h"

static data_t ping[NX][NY];
static data_t pong[NX][NY];

static void stencil_pass(data_t rd[NX][NY], data_t wr[NX][NY]) {
#pragma HLS inline off

    const acc_t wc = (acc_t)0.50;
    const acc_t wa = (acc_t)0.10;
    const acc_t wd = (acc_t)0.025;

    data_t lb0[NY], lb1[NY], lb2[NY];
#pragma HLS array_partition variable=ping cyclic factor=2 dim=2
#pragma HLS array_partition variable=pong cyclic factor=2 dim=2
#pragma HLS array_partition variable=lb0 cyclic factor=2 dim=1
#pragma HLS array_partition variable=lb1 cyclic factor=2 dim=1
#pragma HLS array_partition variable=lb2 cyclic factor=2 dim=1

    // Prime: lb0 = row 0 (clamped prev), lb1 = row 0, lb2 = row 1
PRIME:
    for (int j = 0; j < NY; j++) {
#pragma HLS pipeline II=1
        lb0[j] = rd[0][j];   // i=-1 clamped to i=0
        lb1[j] = rd[0][j];   // i=0
        lb2[j] = rd[1][j];   // i=1
    }

STENCIL_I:
    for (int i = 0; i < NX; i++) {

        // Initialize the 3-wide column window BEFORE the j loop
        // Preload j=0 and j=1 so window is ready at j=0
        data_t r00 = lb0[0], r01 = lb0[0], r02 = lb0[1]; // clamp j=-1 to j=0
        data_t r10 = lb1[0], r11 = lb1[0], r12 = lb1[1];
        data_t r20 = lb2[0], r21 = lb2[0], r22 = lb2[1];

    STENCIL_J:
        for (int j = 0; j < NY; j++) {
#pragma HLS pipeline II=1
#pragma HLS dependence variable=lb0 inter false
#pragma HLS dependence variable=lb1 inter false
#pragma HLS dependence variable=lb2 inter false

            // At start of iteration j, r_1 = col j-1, r_1 = col j, r_2 = col j+1
            // (already loaded before loop or end of last iter)

            if (i == 0 || i == NX-1 || j == 0 || j == NY-1) {
                wr[i][j] = r11;  // boundary: copy unchanged
            } else {
                acc_t sum_axis = (acc_t)r01 + (acc_t)r21 +
                                 (acc_t)r10 + (acc_t)r12;
                acc_t sum_diag = (acc_t)r00 + (acc_t)r02 +
                                 (acc_t)r20 + (acc_t)r22;
                wr[i][j] = (data_t)(wc*(acc_t)r11 + wa*sum_axis + wd*sum_diag);
            }

            // Slide window: shift left, load next column
            r00 = r01; r01 = r02;
            r10 = r11; r11 = r12;
            r20 = r21; r21 = r22;

            // Load j+2, clamped
            int jn = (j + 2 < NY) ? j + 2 : NY - 1;
            r02 = lb0[jn];
            r12 = lb1[jn];
            r22 = lb2[jn];
        }

        // Rotate line buffers
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