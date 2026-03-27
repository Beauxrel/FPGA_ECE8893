// host.cpp
// Bare-bones SHA-256 golden kernel + comparison against top()

#include <stdint.h>
#include <stdio.h>

#define BLOCK_WORDS 16
#define HASH_WORDS  8

static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static const uint32_t H0[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

#define ROTR(x,n)   (((x)>>(n))|((x)<<(32-(n))))
#define CH(x,y,z)   (((x)&(y))^(~(x)&(z)))
#define MAJ(x,y,z)  (((x)&(y))^((x)&(z))^((y)&(z)))
#define EP0(x)      (ROTR(x,2) ^ROTR(x,13)^ROTR(x,22))
#define EP1(x)      (ROTR(x,6) ^ROTR(x,11)^ROTR(x,25))
#define SIG0(x)     (ROTR(x,7) ^ROTR(x,18)^((x)>>3))
#define SIG1(x)     (ROTR(x,17)^ROTR(x,19)^((x)>>10))

// Forward declaration of top function (defined in top.cpp)
void sha256_top(uint32_t block[BLOCK_WORDS], uint32_t hash[HASH_WORDS]);

// Golden reference implementation
void sha256_golden(uint32_t block[BLOCK_WORDS], uint32_t hash[HASH_WORDS]) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = block[i];
    for (int i = 16; i < 64; i++)
        w[i] = SIG1(w[i-2]) + w[i-7] + SIG0(w[i-15]) + w[i-16];

    uint32_t a=H0[0], b=H0[1], c=H0[2], d=H0[3];
    uint32_t e=H0[4], f=H0[5], g=H0[6], h=H0[7];

    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + EP1(e) + CH(e,f,g) + K[i] + w[i];
        uint32_t t2 = EP0(a) + MAJ(a,b,c);
        h=g; g=f; f=e; e=d+t1;
        d=c; c=b; b=a; a=t1+t2;
    }

    hash[0]=H0[0]+a; hash[1]=H0[1]+b; hash[2]=H0[2]+c; hash[3]=H0[3]+d;
    hash[4]=H0[4]+e; hash[5]=H0[5]+f; hash[6]=H0[6]+g; hash[7]=H0[7]+h;
}

int main() {
    // Test vector: SHA-256("abc"), padded to one 512-bit block
    uint32_t block[BLOCK_WORDS] = {
        0x61626380, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000018
    };

    uint32_t golden[HASH_WORDS] = {0};
    uint32_t result[HASH_WORDS] = {0};

    sha256_golden(block, golden);
    sha256_top(block, result);

    printf("Golden: ");
    for (int i = 0; i < HASH_WORDS; i++) printf("%08x ", golden[i]);
    printf("\nTop:    ");
    for (int i = 0; i < HASH_WORDS; i++) printf("%08x ", result[i]);
    printf("\n");

    int errors = 0;
    for (int i = 0; i < HASH_WORDS; i++)
        if (golden[i] != result[i]) errors++;

    if (errors == 0) printf("PASS\n");
    else             printf("FAIL: %d mismatches\n", errors);

    return errors;
}