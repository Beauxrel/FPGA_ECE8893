#include "dcl.h"

// AES-128 encryption — single 128-bit block
// Kernels: key_expansion -> add_round_key -> sub_bytes -> shift_rows -> mix_columns
// Called in sequence from top_kernel for each of the 10 AES rounds.

// ============================================================
// Kernel 1: key_expansion
// ============================================================
static void key_expansion(const data_t key[16], word_t w[KEY_WORDS]) {
    uint8_t temp[4];
    uint8_t a, b, c, d;

    for (int i = 0; i < NK; i++) {
        uint8_t k0 = key[i * 4 + 0];
        uint8_t k1 = key[i * 4 + 1];
        uint8_t k2 = key[i * 4 + 2];
        uint8_t k3 = key[i * 4 + 3];
        w[i] = ((word_t)k0 << 24)
             | ((word_t)k1 << 16)
             | ((word_t)k2 <<  8)
             | ((word_t)k3 <<  0);
    }

    for (int i = NK; i < KEY_WORDS; i++) {
        a = (uint8_t)((w[i - 1] >> 24) & 0xFF);
        b = (uint8_t)((w[i - 1] >> 16) & 0xFF);
        c = (uint8_t)((w[i - 1] >>  8) & 0xFF);
        d = (uint8_t)((w[i - 1] >>  0) & 0xFF);

        if (i % NK == 0) {
            uint8_t rot_a = b;
            uint8_t rot_b = c;
            uint8_t rot_c = d;
            uint8_t rot_d = a;
            temp[0] = SBOX[rot_a] ^ RCON[i / NK];
            temp[1] = SBOX[rot_b];
            temp[2] = SBOX[rot_c];
            temp[3] = SBOX[rot_d];
        } else {
            temp[0] = a;
            temp[1] = b;
            temp[2] = c;
            temp[3] = d;
        }

        word_t prev = w[i - NK];
        w[i] = prev ^ (((word_t)temp[0] << 24)
                     | ((word_t)temp[1] << 16)
                     | ((word_t)temp[2] <<  8)
                     | ((word_t)temp[3] <<  0));
    }
}

// ============================================================
// Kernel 2: add_round_key
// ============================================================
static void add_round_key(aes_state_t state, const word_t w[KEY_WORDS], int round) {
    for (int col = 0; col < NB; col++) {
        word_t word = w[round * NB + col];

        uint8_t rk0 = (uint8_t)((word >> 24) & 0xFF);
        uint8_t rk1 = (uint8_t)((word >> 16) & 0xFF);
        uint8_t rk2 = (uint8_t)((word >>  8) & 0xFF);
        uint8_t rk3 = (uint8_t)((word >>  0) & 0xFF);

        uint8_t s0 = state[0][col];
        uint8_t s1 = state[1][col];
        uint8_t s2 = state[2][col];
        uint8_t s3 = state[3][col];

        state[0][col] = s0 ^ rk0;
        state[1][col] = s1 ^ rk1;
        state[2][col] = s2 ^ rk2;
        state[3][col] = s3 ^ rk3;
    }
}

// ============================================================
// Kernel 3: sub_bytes
// ============================================================
static void sub_bytes(aes_state_t state) {
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < NB; col++) {
            uint8_t original    = state[row][col];
            uint8_t substituted = SBOX[original];
            state[row][col]     = substituted;
        }
    }
}

// ============================================================
// Kernel 4: shift_rows
// ============================================================
static void shift_rows(aes_state_t state) {
    uint8_t temp;

    // Row 0: no shift

    // Row 1: shift left by 1
    temp           = state[1][0];
    state[1][0]    = state[1][1];
    state[1][1]    = state[1][2];
    state[1][2]    = state[1][3];
    state[1][3]    = temp;

    // Row 2: shift left by 2
    temp           = state[2][0];
    state[2][0]    = state[2][2];
    state[2][2]    = temp;
    temp           = state[2][1];
    state[2][1]    = state[2][3];
    state[2][3]    = temp;

    // Row 3: shift left by 3 (= right by 1)
    temp           = state[3][3];
    state[3][3]    = state[3][2];
    state[3][2]    = state[3][1];
    state[3][1]    = state[3][0];
    state[3][0]    = temp;
}

// ============================================================
// Kernel 5: mix_columns
// ============================================================
static uint8_t xtime(uint8_t x) {
    return (uint8_t)(((x << 1) & 0xFF) ^ ((x & 0x80) ? 0x1b : 0x00));
}

static void mix_columns(aes_state_t state) {
    for (int col = 0; col < NB; col++) {
        uint8_t s0 = state[0][col];
        uint8_t s1 = state[1][col];
        uint8_t s2 = state[2][col];
        uint8_t s3 = state[3][col];

        uint8_t p0 = xtime(s0);
        uint8_t p1 = xtime(s1);
        uint8_t p2 = xtime(s2);
        uint8_t p3 = xtime(s3);

        state[0][col] = p0 ^ (p1 ^ s1) ^ s2 ^ s3;
        state[1][col] = s0 ^ p1 ^ (p2 ^ s2) ^ s3;
        state[2][col] = s0 ^ s1 ^ p2 ^ (p3 ^ s3);
        state[3][col] = (p0 ^ s0) ^ s1 ^ s2 ^ p3;
    }
}

// ============================================================
// Top kernel: AES-128 encryption of a single 128-bit block
// ============================================================
void top_kernel(const data_t plaintext[NUM_BLOCKS * 16],
                const data_t key[16],
                      data_t out[NUM_BLOCKS * 16]) {

    word_t      w[KEY_WORDS];
    aes_state_t state;

    for (int b = 0; b < NUM_BLOCKS; b++) {

        // Load block b into state matrix column by column
        for (int col = 0; col < NB; col++) {
            for (int row = 0; row < 4; row++) {
                state[row][col] = plaintext[b * 16 + col * 4 + row];
            }
        }

        // Kernel 1: expand key into round key schedule
        key_expansion(key, w);

        // Initial round key addition
        add_round_key(state, w, 0);

        // Rounds 1-9: all four kernels
        for (int round = 1; round <= NR - 1; round++) {
            sub_bytes(state);
            shift_rows(state);
            mix_columns(state);
            add_round_key(state, w, round);
        }

        // Final round: no mix_columns
        sub_bytes(state);
        shift_rows(state);
        add_round_key(state, w, NR);

        // Unload state into output buffer column by column
        for (int col = 0; col < NB; col++) {
            for (int row = 0; row < 4; row++) {
                out[b * 16 + col * 4 + row] = state[row][col];
            }
        }

    }
}