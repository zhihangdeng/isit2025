#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gf_rand.h>
#include <smmintrin.h>
#include <stdint.h>

double get_time_diff(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
}

static void *malloc16(int size) {
    void *mem = malloc(size+16+sizeof(void*));
    void **ptr = (void**)((long)(mem+16+sizeof(void*)) & ~(15));
    ptr[-1] = mem;
    return ptr;
}

static void free16(void *ptr) {
    free(((void**)ptr)[-1]);
}

#define talloc(type, num) (type *) malloc16(sizeof(type)*(num))

static void usage(char *s)
{
    fprintf(stderr, "usage: lrc_time_ring k m w seed iterations bufsize - Test and time LRC in a particular Ring.\n");
    fprintf(stderr, "       \n");
    fprintf(stderr, "       w must be 16.  k+m must be <= 2^w.\n");
    fprintf(stderr, "       \n");
    fprintf(stderr, "This tests:        lrc_encode()\n");
    fprintf(stderr, "                   lrc_decode()\n");
    if (s != NULL) fprintf(stderr, "%s\n", s);
    exit(1);
}

uint32_t *inv_table;

void create_inv_table(int n) {
    inv_table = talloc(uint32_t, (1U << n));
    if (!inv_table) {
        fprintf(stderr, "Failed to allocate memory for inv_table\n");
        return;
    }

    for (uint32_t i = 0; i < (1U << n); i++) {
        if (i % 2 == 0) {
            inv_table[i] = 0;
        } else {
            uint32_t b = i - 1;
            uint32_t t = 1;
            uint32_t x = 0;
            for (int j = 0; j < n; j++) {
                x = (x + t) % (1U << n);
                t = (t * (-b)) % (1U << n);
            }
            inv_table[i] = x;
        }
    }
}

uint32_t *v_table;

uint32_t *g_table;

void create_v_and_g_table(int n) {
    v_table = talloc(uint32_t, (1U << n));
    g_table = talloc(uint32_t, (1U << n));
    if (!v_table || !g_table) {
        fprintf(stderr, "Failed to allocate memory for v_table or g_table\n");
        return;
    }

    for (uint32_t i = 1; i < (1U << n); i++) {
        uint32_t a = i;
        v_table[i] = 0;
        while (a % 2 == 0) {
            a >>= 1;
            v_table[i] += 1;
        }
        g_table[i] = a;
    }
}

void lrc_encode(int k, int m, int w, int n, char **data, char **coding, int bufsize) {
    // Dummy implementation of LRC ring encoding
    uint32_t mask = (1 << n) - 1;
    memset(coding[0], 0, bufsize);
    memset(coding[1], 0, bufsize);
    memset(coding[2], 0, 2 * bufsize);

    for (int i = 0; i < k; i++) {
        if (i < k / 2) {
            uint16_t *dst = (uint16_t *)coding[0];
            uint16_t *src = (uint16_t *)data[i];
            for (int j = 0; j < bufsize / sizeof(uint16_t); j++) {
                dst[j] = (dst[j] + src[j]);
            }
        } else {
            uint16_t *dst = (uint16_t *)coding[1];
            uint16_t *src = (uint16_t *)data[i];
            for (int j = 0; j < bufsize / sizeof(uint16_t); j++) {
                dst[j] = (dst[j] + src[j]);
            }
        }
        uint32_t *dst = (uint32_t *)coding[2];
        uint16_t *src = (uint16_t *)data[i];
        for (int j = 0; j < 2 * bufsize / sizeof(uint32_t); j++) {
            dst[j] = (dst[j] + (2 * i + 1) * (uint32_t) src[j]) & mask;
        }
    }
}

void lrc_encode_sse(int k, int m, int w, int n, char **data, char **coding, int bufsize) {
    uint32_t mask = (1 << n) - 1;
    memset(coding[0], 0, bufsize);
    memset(coding[1], 0, bufsize);
    memset(coding[2], 0, 2 * bufsize);
    
    for (int i = 0; i < k; i++) {
        uint16_t *src = (uint16_t *)data[i];
        
        if (i < k / 2) {
            uint16_t *dst = (uint16_t *)coding[0];
            __m128i *dst_sse = (__m128i *)dst;
            __m128i *src_sse = (__m128i *)src;

            for (int j = 0; j < bufsize / sizeof(__m128i); j++) {
                __m128i dst_vec = _mm_loadu_si128(&dst_sse[j]);
                __m128i src_vec = _mm_loadu_si128(&src_sse[j]);
                __m128i sum_vec = _mm_add_epi16(dst_vec, src_vec);
                _mm_storeu_si128(&dst_sse[j], sum_vec);
            }
            
        } else {
            uint16_t *dst = (uint16_t *)coding[1];
            __m128i *dst_sse = (__m128i *)dst;
            __m128i *src_sse = (__m128i *)src;

            for (int j = 0; j < bufsize / sizeof(__m128i); j++) {
                __m128i dst_vec = _mm_loadu_si128(&dst_sse[j]);
                __m128i src_vec = _mm_loadu_si128(&src_sse[j]);
                __m128i sum_vec = _mm_add_epi16(dst_vec, src_vec);
                _mm_storeu_si128(&dst_sse[j], sum_vec);
            }
        }
        
        uint32_t *dst = (uint32_t *)coding[2];
        __m128i *dst_sse = (__m128i *)dst;
        __m128i *src_sse = (__m128i *)src;
        uint32_t coef = 2 * i + 1;
        
        __m128i coef_vec = _mm_set1_epi32(coef);
        __m128i mask_vec = _mm_set1_epi32(mask);
        
        for (int j = 0; j < bufsize / sizeof(__m128i); j++) {
            __m128i dst_vec = _mm_loadu_si128(&dst_sse[2*j]);
            __m128i src_vec = _mm_loadu_si128(&src_sse[j]);
            __m128i src_vec_low = _mm_cvtepu16_epi32(src_vec);
            __m128i prod_vec = _mm_mullo_epi32(coef_vec, src_vec_low);
            __m128i sum_vec = _mm_add_epi32(dst_vec, prod_vec);
            sum_vec = _mm_and_si128(sum_vec, mask_vec);
            _mm_storeu_si128(&dst_sse[2*j], sum_vec);

            dst_vec = _mm_loadu_si128(&dst_sse[2*j+1]);
            src_vec = _mm_bsrli_si128(src_vec, 8);
            __m128i src_vec_high = _mm_cvtepu16_epi32(src_vec);
            prod_vec = _mm_mullo_epi32(coef_vec, src_vec_high);
            sum_vec = _mm_add_epi32(dst_vec, prod_vec);
            sum_vec = _mm_and_si128(sum_vec, mask_vec);
            _mm_storeu_si128(&dst_sse[2*j+1], sum_vec);
        }
        
    }
}

int *ring_lrc_erasures_to_erased(int k, int m, int *erasures) {
  int td;
  int t_non_erased;
  int *erased;
  int i;

  td = k + m;
  erased = talloc(int, td);
  if (!erased) {
    fprintf(stderr, "Failed to allocate memory for erased\n");
    return NULL;
  }
  t_non_erased = td;

  for (i = 0; i < td; i++) erased[i] = 0;

  for (i = 0; erasures[i] != -1; i++) {
    if (erased[erasures[i]] == 0) {
      erased[erasures[i]] = 1;
      t_non_erased--;
      if (t_non_erased < k) {
        free(erased);
        return NULL;
      }
    }
  }
  return erased;
}

void lrc_decode(int k, int m, int w, int n, int *erasures, char **data, char **coding, int bufsize) {
    // Dummy implementation of LRC ring decoding
    uint32_t mask = (1 << n) - 1;
    int *erased = ring_lrc_erasures_to_erased(k, m, erasures);
    
    int group1 = 0;
    int group2 = 0;
    int global = 0;
    for (int i = 0; erasures[i] != -1; i++) {
        if (erasures[i] < k / 2 || erasures[i] == k) {
            group1++;
        } else if (erasures[i] < k || erasures[i] == k + 1) {
            group2++;
        } else {
            global++;
        }
    }
    if (group1 == 1) {
        if (erased[k]) {
            // Decode the local parity symbol
            uint16_t *dst = (uint16_t *)coding[0];
            memset(dst, 0, bufsize);
            for (int i = 0; i < k / 2; i++) {
                uint16_t *src = (uint16_t *)data[i];
                for (int j = 0; j < bufsize / sizeof(uint16_t); j++) {
                    dst[j] = (dst[j] + src[j]);
                }
            }

        }else {
            // Decode the data symbol
            int idx = -1;
            for (int i = 0; i < k / 2; i++) {
                if (erased[i]) {
                    idx = i;
                    break;
                }
            }
            uint16_t *dst = (uint16_t *)data[idx];
            memcpy(dst, coding[0], bufsize);
            for (int i = 0; i < k / 2; i++) {
                if (i != idx) {
                    uint16_t *src = (uint16_t *)data[i];
                    for (int j = 0; j < bufsize / sizeof(uint16_t); j++) {
                        dst[j] = (dst[j] - src[j]);
                    }
                }
            }
        }
    }
    if (group2 == 1) {
        if (erased[k + 1]) {
            // Decode the local parity symbol
            uint16_t *dst = (uint16_t *)coding[1];
            memset(dst, 0, bufsize);
            for (int i = k / 2; i < k; i++) {
                uint16_t *src = (uint16_t *)data[i];
                for (int j = 0; j < bufsize / sizeof(uint16_t); j++) {
                    dst[j] = (dst[j] + src[j]);
                }
            }
        }else {
            // Decode the data symbol
            int idx = -1;
            for (int i = k / 2; i < k; i++) {
                if (erased[i]) {
                    idx = i;
                    break;
                }
            }
            uint16_t *dst = (uint16_t *)data[idx];
            memcpy(dst, coding[1], bufsize);
            for (int i = k / 2; i < k; i++) {
                if (i != idx) {
                    uint16_t *src = (uint16_t *)data[i];
                    for (int j = 0; j < bufsize / sizeof(uint16_t); j++) {
                        dst[j] = (dst[j] - src[j]);
                    }
                }
            }
        }
    }
    if (group1 == 2) {
        if (erased[k]) {
            // Decode the data symbol
            int idx = -1;
            for (int i = 0; i < k / 2; i++) {
                if (erased[i]) {
                    idx = i;
                    break;
                }
            }

            uint32_t *tmp = (uint32_t *) talloc(char, 2 * bufsize);
            memcpy(tmp, coding[2], 2 * bufsize);
            for (int i = 0; i < k; i++) {
                if (i != idx) {
                    uint16_t *src = (uint16_t *)data[i];
                    for (int j = 0; j < 2 * bufsize / sizeof(uint32_t); j++) {
                        tmp[j] = (tmp[j] - (2 * i + 1) * (uint32_t) src[j]) & mask;
                    }
                }
            }

            uint16_t *dst = (uint16_t *)data[idx];
            for (int j = 0; j < bufsize / sizeof(uint16_t); j++) {
                uint32_t *src = tmp;
                dst[j] = (uint16_t) ((inv_table[2 * idx + 1] * src[j]) & mask);
            }

            free16(tmp);
            
            // Decode the local parity symbol
            dst = (uint16_t *)coding[0];
            memset(dst, 0, bufsize);
            for (int i = 0; i < k / 2; i++) {
                uint16_t *src = (uint16_t *)data[i];
                for (int j = 0; j < bufsize / sizeof(uint16_t); j++) {
                    dst[j] = (dst[j] + src[j]);
                }
            }

        }else {
            // Decode the data symbols
            int idx1 = -1, idx2 = -1;
            for (int i = 0; i < k / 2; i++) {
                if (erased[i] && idx1 == -1) {
                    idx1 = i;
                } else if (erased[i] && idx2 == -1) {
                    idx2 = i;
                    break;
                }
            }

            uint32_t v = v_table[2 * (idx2 - idx1)];
            uint32_t g = g_table[2 * (idx2 - idx1)];
            uint16_t *syndrome1 = (uint16_t *) talloc(char, bufsize);
            uint32_t *syndrome2 = (uint32_t *) talloc(char, 2 * bufsize);
            memcpy(syndrome1, coding[0], bufsize);
            memcpy(syndrome2, coding[2], 2 * bufsize);

            for (int i = 0; i < k; i++) {
                if (i != idx1 && i != idx2) {
                    uint16_t *src = (uint16_t *) data[i];
                    if (i < k / 2) {
                        for (int j = 0; j < bufsize / sizeof(uint16_t); j++) {
                            syndrome1[j] = (syndrome1[j] - src[j]);
                        }
                    }
                    for (int j = 0; j < 2 * bufsize / sizeof(uint32_t); j++) {
                        syndrome2[j] = (syndrome2[j] - (2 * i + 1) * (uint32_t) src[j]) & mask;
                    }
                }
            }

            for (int j = 0; j < bufsize / sizeof(uint16_t); j++) {
                uint16_t *dst;
                uint16_t *src;
                uint32_t tmp = ((syndrome2[j] - (2 * idx1 + 1) * (uint32_t) syndrome1[j]) * inv_table[g] & mask) >> v;
                if (tmp >= (1U << 16)) {
                    tmp = ((syndrome2[j] - (2 * idx1 + 1) * ((uint32_t) syndrome1[j] + (1U << 16))) * inv_table[g] & mask) >> v;
                }
                dst = (uint16_t *) data[idx2];
                dst[j] = (uint16_t) tmp;
                dst = (uint16_t *) data[idx1];
                src = (uint16_t *) data[idx2];
                dst[j] = syndrome1[j] - src[j];
            }

            free16(syndrome1);
            free16(syndrome2);
        }
    }
    if (group2 == 2) {
        if (erased[k]) {
            // Decode the data symbol
            int idx = -1;
            for (int i = k / 2; i < k; i++) {
                if (erased[i]) {
                    idx = i;
                    break;
                }
            }

            uint32_t *tmp = (uint32_t *) talloc(char, 2 * bufsize);
            memcpy(tmp, coding[2], 2 * bufsize);
            for (int i = 0; i < k; i++) {
                if (i != idx) {
                    uint16_t *src = (uint16_t *)data[i];
                    for (int j = 0; j < 2 * bufsize / sizeof(uint32_t); j++) {
                        tmp[j] = (tmp[j] - (2 * i + 1) * (uint32_t) src[j]) & mask;
                    }
                }
            }

            uint16_t *dst = (uint16_t *)data[idx];
            for (int j = 0; j < bufsize / sizeof(uint16_t); j++) {
                uint32_t *src = tmp;
                dst[j] = (uint16_t) ((inv_table[2 * idx + 1] * src[j]) & mask);
            }
            free16(tmp);
            
            // Decode the local parity symbol
            dst = (uint16_t *)coding[0];
            memset(dst, 0, bufsize);
            for (int i = k / 2; i < k; i++) {
                uint16_t *src = (uint16_t *)data[i];
                for (int j = 0; j < bufsize / sizeof(uint16_t); j++) {
                    dst[j] = (dst[j] + src[j]);
                }
            }

        }else {
            // Decode the data symbols
            int idx1 = -1, idx2 = -1;
            for (int i = k / 2; i < k; i++) {
                if (erased[i] && idx1 == -1) {
                    idx1 = i;
                } else if (erased[i] && idx2 == -1) {
                    idx2 = i;
                    break;
                }
            }

            uint32_t v = v_table[2 * (idx2 - idx1)];
            uint32_t g = g_table[2 * (idx2 - idx1)];
            uint16_t *syndrome1 = (uint16_t *) talloc(char, bufsize);
            uint32_t *syndrome2 = (uint32_t *) talloc(char, 2 * bufsize);
            memcpy(syndrome1, coding[1], bufsize);
            memcpy(syndrome2, coding[2], 2 * bufsize);

            for (int i = 0; i < k; i++) {
                if (i != idx1 && i != idx2) {
                    uint16_t *src = (uint16_t *) data[i];
                    if (i >= k / 2) {
                        for (int j = 0; j < bufsize / sizeof(uint16_t); j++) {
                            syndrome1[j] = (syndrome1[j] - src[j]);
                        }
                    }
                    for (int j = 0; j < 2 * bufsize / sizeof(uint32_t); j++) {
                        syndrome2[j] = (syndrome2[j] - (2 * i + 1) * (uint32_t) src[j]) & mask;
                    }
                }
            }

            for (int j = 0; j < bufsize / sizeof(uint16_t); j++) {
                uint16_t *dst;
                uint16_t *src;
                uint32_t tmp = ((syndrome2[j] - (2 * idx1 + 1) * (uint32_t) syndrome1[j]) * inv_table[g] & mask) >> v;
                if (tmp >= (1U << 16)) {
                    tmp = ((syndrome2[j] - (2 * idx1 + 1) * ((uint32_t) syndrome1[j] + (1U << 16))) * inv_table[g] & mask) >> v;
                }
                dst = (uint16_t *) data[idx2];
                dst[j] = (uint16_t) tmp;
                dst = (uint16_t *) data[idx1];
                src = (uint16_t *) data[idx2];
                dst[j] = syndrome1[j] - src[j];
            }

            free16(syndrome1);
            free16(syndrome2);
        }
    }
    if (global == 1) {
        // Decode the global parity symbol
        uint32_t *dst = (uint32_t *)coding[2];
        memset(dst, 0, 2 * bufsize);
        for (int i = 0; i < k; i++) {
            uint16_t *src = (uint16_t *)data[i];
            for (int j = 0; j < 2 * bufsize / sizeof(uint32_t); j++) {
                dst[j] = (dst[j] + (2 * i + 1) *  (uint32_t) src[j]) & mask;
            }
        }
    }

    free16(erased);
}

void lrc_decode_sse(int k, int m, int w, int n, int *erasures, char **data, char **coding, int bufsize) {
    // Dummy implementation of LRC ring decoding
    uint32_t mask = (1 << n) - 1;
    int *erased = ring_lrc_erasures_to_erased(k, m, erasures);
    
    int group1 = 0;
    int group2 = 0;
    int global = 0;
    for (int i = 0; erasures[i] != -1; i++) {
        if (erasures[i] < k / 2 || erasures[i] == k) {
            group1++;
        } else if (erasures[i] < k || erasures[i] == k + 1) {
            group2++;
        } else {
            global++;
        }
    }

    if (group1 == 1) {
        if (erased[k]) {
            // Decode the local parity symbol
            uint16_t *dst = (uint16_t *)coding[0];
            memset(dst, 0, bufsize);
            for (int i = 0; i < k / 2; i++) {
                uint16_t *src = (uint16_t *)data[i];
                __m128i *dst_sse = (__m128i *)dst;
                __m128i *src_sse = (__m128i *)src;

                for (int j = 0; j < bufsize / sizeof(__m128i); j++) {
                    __m128i dst_vec = _mm_loadu_si128(&dst_sse[j]);
                    __m128i src_vec = _mm_loadu_si128(&src_sse[j]);
                    __m128i sum_vec = _mm_add_epi16(dst_vec, src_vec);
                    _mm_storeu_si128(&dst_sse[j], sum_vec);
                }
            }
        } else {
            // Decode the data symbol
            int idx = -1;
            for (int i = 0; i < k / 2; i++) {
                if (erased[i]) {
                    idx = i;
                    break;
                }
            }
            uint16_t *dst = (uint16_t *)data[idx];
            memcpy(dst, coding[0], bufsize);
            for (int i = 0; i < k / 2; i++) {
                if (i != idx) {
                    uint16_t *src = (uint16_t *)data[i];
                    __m128i *dst_sse = (__m128i *)dst;
                    __m128i *src_sse = (__m128i *)src;

                    for (int j = 0; j < bufsize / sizeof(__m128i); j++) {
                        __m128i dst_vec = _mm_loadu_si128(&dst_sse[j]);
                        __m128i src_vec = _mm_loadu_si128(&src_sse[j]);
                        __m128i sum_vec = _mm_sub_epi16(dst_vec, src_vec);
                        _mm_storeu_si128(&dst_sse[j], sum_vec);
                    }
                }
            }
        }
    }

    if (group2 == 1) {
        if (erased[k + 1]) {
            // Decode the local parity symbol
            uint16_t *dst = (uint16_t *)coding[1];
            memset(dst, 0, bufsize);
            for (int i = k / 2; i < k; i++) {
                uint16_t *src = (uint16_t *)data[i];
                __m128i *dst_sse = (__m128i *)dst;
                __m128i *src_sse = (__m128i *)src;

                for (int j = 0; j < bufsize / sizeof(__m128i); j++) {
                    __m128i dst_vec = _mm_loadu_si128(&dst_sse[j]);
                    __m128i src_vec = _mm_loadu_si128(&src_sse[j]);
                    __m128i sum_vec = _mm_add_epi16(dst_vec, src_vec);
                    _mm_storeu_si128(&dst_sse[j], sum_vec);
                }
            }
        } else {
            // Decode the data symbol
            int idx = -1;
            for (int i = k / 2; i < k; i++) {
                if (erased[i]) {
                    idx = i;
                    break;
                }
            }
            uint16_t *dst = (uint16_t *)data[idx];
            memcpy(dst, coding[1], bufsize);
            for (int i = k / 2; i < k; i++) {
                if (i != idx) {
                    uint16_t *src = (uint16_t *)data[i];
                    __m128i *dst_sse = (__m128i *)dst;
                    __m128i *src_sse = (__m128i *)src;

                    for (int j = 0; j < bufsize / sizeof(__m128i); j++) {
                        __m128i dst_vec = _mm_loadu_si128(&dst_sse[j]);
                        __m128i src_vec = _mm_loadu_si128(&src_sse[j]);
                        __m128i sum_vec = _mm_sub_epi16(dst_vec, src_vec);
                        _mm_storeu_si128(&dst_sse[j], sum_vec);
                    }
                }
            }
        }
    }

    if (group1 == 2) {
        if (erased[k]) {
            // Decode the data symbol
            int idx = -1;
            for (int i = 0; i < k / 2; i++) {
                if (erased[i]) {
                    idx = i;
                    break;
                }
            }

            uint32_t *tmp = (uint32_t *) talloc(char, 2 * bufsize);
            memcpy(tmp, coding[2], 2 * bufsize);
            for (int i = 0; i < k; i++) {
                if (i != idx) {
                    uint32_t *dst = (uint32_t *)tmp;
                    uint16_t *src = (uint16_t *)data[i];
                    __m128i *dst_sse = (__m128i *)dst;
                    __m128i *src_sse = (__m128i *)src;
                    uint32_t coef = 2 * i + 1;
                    
                    __m128i coef_vec = _mm_set1_epi32(coef);
                    __m128i mask_vec = _mm_set1_epi32(mask);

                    for (int j = 0; j < bufsize / sizeof(__m128i); j++) {
                        __m128i dst_vec = _mm_loadu_si128(&dst_sse[2*j]);
                        __m128i src_vec = _mm_loadu_si128(&src_sse[j]);
                        __m128i src_vec_low = _mm_cvtepu16_epi32(src_vec);
                        __m128i prod_vec = _mm_mullo_epi32(coef_vec, src_vec_low);
                        __m128i sub_vec = _mm_sub_epi32(dst_vec, prod_vec);
                        sub_vec = _mm_and_si128(sub_vec, mask_vec);
                        _mm_storeu_si128(&dst_sse[2*j], sub_vec);

                        dst_vec = _mm_loadu_si128(&dst_sse[2*j+1]);
                        src_vec = _mm_bsrli_si128(src_vec, 8);
                        __m128i src_vec_high = _mm_cvtepu16_epi32(src_vec);
                        prod_vec = _mm_mullo_epi32(coef_vec, src_vec_high);
                        sub_vec = _mm_sub_epi32(dst_vec, prod_vec);
                        sub_vec = _mm_and_si128(sub_vec, mask_vec);
                        _mm_storeu_si128(&dst_sse[2*j+1], sub_vec);
                    }
                }
            }

            uint16_t *dst = (uint16_t *)data[idx];
            __m128i *dst_sse = (__m128i *)dst;
            __m128i *src_sse = (__m128i *)tmp;
            uint32_t inv_coef = inv_table[2 * idx + 1];
            __m128i inv_coef_vec = _mm_set1_epi32(inv_coef);
            __m128i mask_vec = _mm_set1_epi32(mask);

            for (int j = 0; j < bufsize / sizeof(__m128i); j++) {
                __m128i src_vec1 = _mm_loadu_si128(&src_sse[2*j]);
                __m128i src_vec2 = _mm_loadu_si128(&src_sse[2*j + 1]);

                __m128i prod_vec1 = _mm_mullo_epi32(inv_coef_vec, src_vec1);
                __m128i prod_vec2 = _mm_mullo_epi32(inv_coef_vec, src_vec2);

                prod_vec1 = _mm_and_si128(prod_vec1, mask_vec);
                prod_vec2 = _mm_and_si128(prod_vec2, mask_vec);

                __m128i res_vec = _mm_packus_epi32(prod_vec1, prod_vec2);

                _mm_storeu_si128(&dst_sse[j], res_vec);
            }

            free16(tmp);

            // Decode the local parity symbol
            dst = (uint16_t *)coding[0];
            memset(dst, 0, bufsize);
            for (int i = 0; i < k / 2; i++) {
                uint16_t *src = (uint16_t *)data[i];
                __m128i *dst_sse = (__m128i *)dst;
                __m128i *src_sse = (__m128i *)src;

                for (int j = 0; j < bufsize / sizeof(__m128i); j++) {
                    __m128i dst_vec = _mm_loadu_si128(&dst_sse[j]);
                    __m128i src_vec = _mm_loadu_si128(&src_sse[j]);
                    __m128i sum_vec = _mm_add_epi16(dst_vec, src_vec);
                    _mm_storeu_si128(&dst_sse[j], sum_vec);
                }
            }
            
        } else {
            // Decode the data symbols
            int idx1 = -1, idx2 = -1;
            for (int i = 0; i < k / 2; i++) {
                if (erased[i] && idx1 == -1) {
                    idx1 = i;
                } else if (erased[i] && idx2 == -1) {
                    idx2 = i;
                    break;
                }
            }
            
        }

    }
}

int main(int argc, char **argv) {
    int k, w, m, n = 20, iterations, bufsize;
    char **data, **coding, **old_values;
    int *erasures, *erased;
    uint32_t seed;

    if (argc < 8) usage(NULL);  
    if (sscanf(argv[1], "%d", &k) == 0 || k <= 0) usage("Bad k");
    if (sscanf(argv[2], "%d", &m) == 0 || m <= 0) usage("Bad m");
    if (sscanf(argv[3], "%d", &w) == 0 || w != 16) usage("Bad w");
    if (sscanf(argv[4], "%d", &seed) == 0) usage("Bad seed");
    if (sscanf(argv[5], "%d", &iterations) == 0) usage("Bad iterations");
    if (sscanf(argv[6], "%d", &bufsize) == 0) usage("Bad bufsize");
    if (k + m > (1 << w)) usage("k + m is too big");
    
    MOA_Seed(seed);
    create_inv_table(n);
    create_v_and_g_table(n);

    data = talloc(char *, k);
    for (int i = 0; i < k; i++) {
        data[i] = talloc(char, bufsize);
        MOA_Fill_Random_Region(data[i], bufsize);
        // if (i == 0) *(uint16_t *)data[i] = 65433;
        // if (i == 8) *(uint16_t *)data[i] = 48103;
        // printf("0x%hhx 0x%hhx %u\n", data[i][0], data[i][1], *(uint16_t *)data[i]);
    }
  
    coding = talloc(char *, m);
    old_values = talloc(char *, m);
    for (int i = 0; i < m; i++) {
        if (i < m - 1) {
            coding[i] = talloc(char, bufsize);
        } else {
            coding[i] = talloc(char, 2 * bufsize);
        }
        old_values[i] = talloc(char, bufsize);
    }

    struct timeval start, end; // 用于时间统计

    // 编码操作时间统计
    gettimeofday(&start, NULL); // 开始计时
    for (int i = 0; i < iterations; i++) {
        lrc_encode_sse(k, m, w, n, data, coding, bufsize);
    }
    gettimeofday(&end, NULL); // 结束计时
    double encode_time = get_time_diff(start, end);
    printf("[Ring-based LRC] Encode throughput for %d iterations: %.2f MB/s (%.2f sec)\n",
            iterations, (double)(k * iterations * bufsize / 1024 / 1024) / encode_time, encode_time);
    
    // printf("0x%hhx 0x%hhx %u\n", coding[0][0], coding[0][1], *(uint16_t *)coding[0]);
    // printf("0x%hhx 0x%hhx %u\n", coding[1][0], coding[1][1], *(uint16_t *)coding[1]);
    // printf("0x%hhx 0x%hhx 0x%hhx 0x%hhx %u\n", coding[2][0], coding[2][1], coding[2][2], coding[2][3], *(uint32_t *)coding[2]);

    erasures = talloc(int, (m+1));
    erased = talloc(int, (k+m));
    
    for (int erasure_num = 1; erasure_num <= 2; erasure_num++) {
        for (int i = 0; i < m+k; i++) erased[i] = 0;
        int i;
        for (i = 0; i < erasure_num; ) {
            erasures[i] = ((unsigned int)MOA_Random_W(w, 1))%(k+m);
            // if (i == 0) erasures[i] = 0;
            // if (i == 1) erasures[i] = 8;
            if (erased[erasures[i]] == 0) {
                erased[erasures[i]] = 1;
                memcpy(old_values[i], (erasures[i] < k) ? data[erasures[i]] : coding[erasures[i]-k], bufsize);
                bzero((erasures[i] < k) ? data[erasures[i]] : coding[erasures[i]-k], bufsize);
                i++;
            }
        }
        erasures[i] = -1;

        // 解码操作时间统计
        gettimeofday(&start, NULL); // 开始计时
        for (int i = 0; i < iterations; i++) {
            lrc_decode_sse(k, m, w, n, erasures, data, coding, bufsize);
        }
        gettimeofday(&end, NULL); // 结束计时
        double decode_time = get_time_diff(start, end);
        printf("[Ring-based LRC] Decode throughput for %d iterations with %d erasure: %.2f MB/s (%.2f sec)\n",
            iterations, erasure_num, (double)(k * iterations * bufsize / 1024 / 1024) / decode_time, decode_time);

        for (int i = 0; i < erasure_num; i++) {
            if (erasures[i] < k) {
                if (memcmp(data[erasures[i]], old_values[i], bufsize)) {
                    fprintf(stderr, "Decoding failed for %d!\n", erasures[i]);
                    goto CLEANUP;
                }
            } else {
                if (memcmp(coding[erasures[i]-k], old_values[i], bufsize)) {
                    fprintf(stderr, "Decoding failed for %d!\n", erasures[i]);
                    goto CLEANUP;
                }
            }
        }
    }

CLEANUP:
    free16(erasures);
    free16(erased);
    free16(inv_table);
    free16(v_table);
    free16(g_table);
    for (int i = 0; i < k; i++) {
        free16(data[i]);
    }
    for (int i = 0; i < m; i++) {
        free16(coding[i]);
        free16(old_values[i]);
    }
    free16(data);
    free16(coding);
    free16(old_values);
}