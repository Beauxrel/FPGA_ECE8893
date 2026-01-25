void top_kernel(data_t A[N_ROWS][N_COLS],
                data_t C[N_ROWS][N_COLS]) {
  static data_t tmp[N_ROWS][N_COLS];
  static data_t col_acc[N_COLS];
  static data_t scale[N_COLS];

#pragma HLS ARRAY_PARTITION variable=tmp     cyclic factor=32 dim=2
#pragma HLS ARRAY_PARTITION variable=col_acc cyclic factor=32 dim=1
#pragma HLS ARRAY_PARTITION variable=scale   cyclic factor=32 dim=1

  // init col_acc
  init: for (int j=0;j<N_COLS;j++) {
#pragma HLS PIPELINE II=1
    col_acc[j] = 0;
  }

  // Phase 1: normalize + accumulate column sums
  for (int i=0;i<N_ROWS;i++) {
    data_t row_sum = 0;

    for (int j=0;j<N_COLS;j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
      row_sum += A[i][j];
    }

    data_t denom = row_sum + (data_t)1.0;
    // If data_t is float, consider computing reciprocal once:
    data_t inv = (data_t)1.0 / denom;

    for (int j=0;j<N_COLS;j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
      data_t v = A[i][j] * inv;      // multiply instead of divide each element
      tmp[i][j] = v;
      col_acc[j] += v;
    }
  }

  // compute scales
  for (int j=0;j<N_COLS;j++) {
#pragma HLS PIPELINE II=1
    scale[j] = col_acc[j] / (data_t)N_ROWS;
  }

  // Phase 2: apply scale (row-major access)
  for (int i=0;i<N_ROWS;i++) {
    for (int j=0;j<N_COLS;j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
      C[i][j] = tmp[i][j] * scale[j];
    }
  }
}
