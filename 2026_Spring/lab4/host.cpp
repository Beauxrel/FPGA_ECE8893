// host.cpp
// AES-128 encryption golden kernel — 5 distinct computational kernels
// Encrypts a single 128-bit block using a 128-bit key
// Kernels: key_expansion, sub_bytes, shift_rows, mix_columns, add_round_key

#include "dcl.h"
#include <stdio.h>
#include <string.h>

// ============================================================
// AES S-Box (forward substitution table)
// Each byte of state is replaced by its S-Box value.
// This is the core nonlinear component of AES.
// ============================================================
// Round constants for key expansion (Rcon)
// AES state is a 4x4 byte matrix stored row-major
// state[row][col], 0 <= row,col < 4
typedef uint8_t aes_state_t[4][4];

// ============================================================
// Kernel 1: key_expansion
// Expands the 16-byte (128-bit) input key into 44 words
// (176 bytes) of round key material. The first 4 words are
// the original key. Each subsequent word depends on the
// previous word and the word 4 positions back, with every
// 4th word also applying SubWord (S-Box on each byte) and
// XOR with a round constant (Rcon). This creates strong
// inter-round key dependencies used by add_round_key.
// ============================================================
void key_expansion(const uint8_t key[16], uint32_t w[KEY_WORDS]) {
    uint32_t i;
    uint8_t  temp[4];
    uint8_t  a, b, c, d;

    // Load the original key into the first NK words
    for (i = 0; i < NK; i++) {
        uint8_t k0 = key[i * 4 + 0];
        uint8_t k1 = key[i * 4 + 1];
        uint8_t k2 = key[i * 4 + 2];
        uint8_t k3 = key[i * 4 + 3];
        w[i] = ((uint32_t)k0 << 24)
             | ((uint32_t)k1 << 16)
             | ((uint32_t)k2 <<  8)
             | ((uint32_t)k3 <<  0);
    }

    // Expand remaining words
    for (i = NK; i < KEY_WORDS; i++) {
        // Extract bytes of the previous word
        a = (uint8_t)((w[i - 1] >> 24) & 0xFF);
        b = (uint8_t)((w[i - 1] >> 16) & 0xFF);
        c = (uint8_t)((w[i - 1] >>  8) & 0xFF);
        d = (uint8_t)((w[i - 1] >>  0) & 0xFF);

        if (i % NK == 0) {
            // RotWord: rotate bytes left by one position
            uint8_t rot_a = b;
            uint8_t rot_b = c;
            uint8_t rot_c = d;
            uint8_t rot_d = a;

            // SubWord: apply S-Box to each byte
            temp[0] = SBOX[rot_a];
            temp[1] = SBOX[rot_b];
            temp[2] = SBOX[rot_c];
            temp[3] = SBOX[rot_d];

            // XOR first byte with round constant
            temp[0] = temp[0] ^ RCON[i / NK];
        } else {
            temp[0] = a;
            temp[1] = b;
            temp[2] = c;
            temp[3] = d;
        }

        // New word = word NK positions back XOR temp
        uint32_t prev = w[i - NK];
        w[i] = prev ^ (((uint32_t)temp[0] << 24)
                     | ((uint32_t)temp[1] << 16)
                     | ((uint32_t)temp[2] <<  8)
                     | ((uint32_t)temp[3] <<  0));
    }
}

// ============================================================
// Kernel 2: add_round_key
// XORs the current AES state with the round key for round r.
// Each round key is 4 words (128 bits) extracted from the
// expanded key schedule. The XOR is applied column by column,
// byte by byte. This is the only step that directly introduces
// key material into the state and must occur every round.
// ============================================================
void add_round_key(aes_state_t state, const uint32_t w[KEY_WORDS], int round) {
    int col;

    // Process each of the 4 columns in the state matrix
    for (col = 0; col < NB; col++) {
        // Fetch the round key word for this column from the key schedule.
        // Round r uses words w[r*4] through w[r*4+3].
        uint32_t word = w[round * NB + col];

        // Extract all four bytes of the round key word (big-endian)
        uint8_t rk0 = (uint8_t)((word >> 24) & 0xFF);
        uint8_t rk1 = (uint8_t)((word >> 16) & 0xFF);
        uint8_t rk2 = (uint8_t)((word >>  8) & 0xFF);
        uint8_t rk3 = (uint8_t)((word >>  0) & 0xFF);

        // Read current state bytes for this column
        uint8_t s0 = state[0][col];
        uint8_t s1 = state[1][col];
        uint8_t s2 = state[2][col];
        uint8_t s3 = state[3][col];

        // XOR state bytes with round key bytes and write back.
        // XOR is its own inverse, so the same operation decrypts.
        state[0][col] = s0 ^ rk0;
        state[1][col] = s1 ^ rk1;
        state[2][col] = s2 ^ rk2;
        state[3][col] = s3 ^ rk3;
    }
}

// ============================================================
// Kernel 3: sub_bytes
// Substitutes every byte in the 4x4 state matrix with its
// corresponding value from the AES S-Box. The S-Box is derived
// from the multiplicative inverse in GF(2^8) followed by an
// affine transformation, making this the primary source of
// nonlinearity and confusion in AES. Operates on all 16 bytes
// of state independently with no inter-byte dependencies.
// ============================================================
void sub_bytes(aes_state_t state) {
    int row, col;
    uint8_t original, substituted;

    // Apply the S-Box to every byte in the 4x4 state matrix.
    // The S-Box lookup is the only nonlinear operation in AES —
    // it is what makes the cipher resistant to linear cryptanalysis.
    for (row = 0; row < 4; row++) {
        for (col = 0; col < NB; col++) {
            // Read the current byte at position (row, col)
            original = state[row][col];

            // The S-Box maps each of the 256 possible byte values
            // to a unique output. It is constructed from the
            // multiplicative inverse in GF(2^8) composed with an
            // affine transformation to eliminate fixed points.
            substituted = SBOX[original];

            // Overwrite with the substituted value
            state[row][col] = substituted;
        }
    }
}

// ============================================================
// Kernel 4: shift_rows
// Cyclically shifts the bytes in each row of the state matrix
// to the left by a row-dependent offset: row 0 shifts by 0,
// row 1 by 1, row 2 by 2, row 3 by 3. This provides diffusion
// across columns — bytes that were in the same column are
// spread to different columns, so subsequent MixColumns will
// mix bytes from different original columns together.
// ============================================================
void shift_rows(aes_state_t state) {
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

    // Row 3: shift left by 3 (equivalent to shift right by 1)
    temp           = state[3][3];
    state[3][3]    = state[3][2];
    state[3][2]    = state[3][1];
    state[3][1]    = state[3][0];
    state[3][0]    = temp;
}

// ============================================================
// Helper: xtime
// Multiplies a byte by 2 in GF(2^8) using the AES irreducible
// polynomial x^8 + x^4 + x^3 + x + 1 (0x11b). If the high
// bit is set the result is reduced by XOR with 0x1b.
// Used internally by mix_columns.
// ============================================================
static uint8_t xtime(uint8_t x) {
    return (uint8_t)(((x << 1) & 0xFF) ^ ((x & 0x80) ? 0x1b : 0x00));
}

// ============================================================
// Kernel 5: mix_columns
// Treats each column of the state as a 4-element polynomial
// over GF(2^8) and multiplies it by a fixed matrix:
//   [2 3 1 1]
//   [1 2 3 1]
//   [1 1 2 3]
//   [3 1 1 2]
// All arithmetic is in GF(2^8). This mixes the four bytes of
// each column together, providing diffusion across rows. When
// combined with shift_rows, every output byte depends on every
// input byte within two rounds (the wide trail strategy).
// Not applied on the final round per the AES specification.
// ============================================================
void mix_columns(aes_state_t state) {
    int col;
    uint8_t s0, s1, s2, s3;
    uint8_t p0, p1, p2, p3;

    for (col = 0; col < NB; col++) {
        // Load the four bytes of this column
        s0 = state[0][col];
        s1 = state[1][col];
        s2 = state[2][col];
        s3 = state[3][col];

        // Compute xtime (multiply by 2 in GF(2^8)) for each byte
        p0 = xtime(s0);
        p1 = xtime(s1);
        p2 = xtime(s2);
        p3 = xtime(s3);

        // Apply the MixColumns matrix multiplication in GF(2^8).
        // Multiplication by 3 = xtime(x) XOR x.
        // Multiplication by 1 = identity.
        state[0][col] = p0 ^ (p1 ^ s1) ^ s2 ^ s3;
        state[1][col] = s0 ^ p1 ^ (p2 ^ s2) ^ s3;
        state[2][col] = s0 ^ s1 ^ p2 ^ (p3 ^ s3);
        state[3][col] = (p0 ^ s0) ^ s1 ^ s2 ^ p3;
    }
}

// ============================================================
// Golden kernel: chains all 5 kernels for AES-128 encryption
// Plaintext and key are 16-byte arrays. Ciphertext written
// to out[16]. Follows the standard AES round structure:
//   - Initial AddRoundKey
//   - 9 full rounds (SubBytes, ShiftRows, MixColumns, AddRoundKey)
//   - 1 final round (SubBytes, ShiftRows, AddRoundKey — no MixColumns)
// ============================================================
void golden_kernel(const uint8_t plaintext[NUM_BLOCKS * 16],
                   const uint8_t key[16],
                   uint8_t       out[NUM_BLOCKS * 16])
{
    uint32_t     w[KEY_WORDS];
    aes_state_t  state;
    int          round, row, col;

    for (int b = 0; b < NUM_BLOCKS; b++) {

        // Load block b into state matrix column by column
        for (col = 0; col < NB; col++) {
            for (row = 0; row < 4; row++) {
                state[row][col] = plaintext[b * 16 + col * 4 + row];
            }
        }

        // Expand the key into the round key schedule
        key_expansion(key, w);

        // Initial round key addition
        add_round_key(state, w, 0);

        // Rounds 1 through 9: full rounds
        for (round = 1; round <= NR - 1; round++) {
            sub_bytes(state);
            shift_rows(state);
            mix_columns(state);
            add_round_key(state, w, round);
        }

        // Final round: no MixColumns
        sub_bytes(state);
        shift_rows(state);
        add_round_key(state, w, NR);

        // Unload state into output buffer column by column
        for (col = 0; col < NB; col++) {
            for (row = 0; row < 4; row++) {
                out[b * 16 + col * 4 + row] = state[row][col];
            }
        }

    }
}

// ============================================================
// Forward declaration of HLS top function (defined in top.cpp)
// ============================================================
void top_kernel(const data_t plaintext[NUM_BLOCKS * 16],
                const data_t key[16],
                      data_t out[NUM_BLOCKS * 16]);

// ============================================================
// Main — FIPS 197 test vector
// Plaintext:  00112233445566778899aabbccddeeff
// Key:        000102030405060708090a0b0c0d0e0f
// Ciphertext: 69c4e0d86a7b0430d8cdb78070b4c55a
// Verified against OpenSSL aes-128-ecb
// ============================================================
int main() {
    const uint8_t plaintext[16] = {
        0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,
        0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff
    };
    const uint8_t key[16] = {
        0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
        0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f
    };
    const uint8_t expected[16] = {
        0x69,0xc4,0xe0,0xd8,0x6a,0x7b,0x04,0x30,
        0xd8,0xcd,0xb7,0x80,0x70,0xb4,0xc5,0x5a
    };

    // Build NUM_BLOCKS plaintexts: block 0 is the FIPS vector,
    // subsequent blocks are incremented by 1 in the first byte
    uint8_t all_plain[NUM_BLOCKS * 16];
    uint8_t all_golden[NUM_BLOCKS * 16] = {0};
    uint8_t all_result[NUM_BLOCKS * 16] = {0};

    for (int b = 0; b < NUM_BLOCKS; b++) {
        for (int i = 0; i < 16; i++)
            all_plain[b * 16 + i] = plaintext[i];
        all_plain[b * 16] ^= (uint8_t)b;   // make each block distinct
    }

    golden_kernel(all_plain, key, all_golden);
    top_kernel(all_plain, key, all_result);

    // Check block 0 against known ciphertext
    printf("Expected: ");
    for (int i = 0; i < 16; i++) printf("%02x", expected[i]);
    printf("\nGolden:   ");
    for (int i = 0; i < 16; i++) printf("%02x", all_golden[i]);
    printf("\nTop:      ");
    for (int i = 0; i < 16; i++) printf("%02x", all_result[i]);
    printf("\n");

    int errors = 0;
    for (int i = 0; i < NUM_BLOCKS * 16; i++)
        if (all_golden[i] != all_result[i]) errors++;
    for (int i = 0; i < 16; i++)
        if (all_golden[i] != expected[i]) errors++;

    if (errors == 0) printf("PASS\n");
    else             printf("FAIL: %d mismatches\n", errors);

    return errors;
}