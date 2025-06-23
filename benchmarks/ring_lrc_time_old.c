#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gf_rand.h>
#include <smmintrin.h>

// 计算时间差的辅助函数
double get_time_diff(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
}

static void *malloc16(int size) {
    void *mem = malloc(size+16+sizeof(void*));
    void **ptr = (void**)((long)(mem+16+sizeof(void*)) & ~(15));
    ptr[-1] = mem;
    return ptr;
}

#define talloc(type, num) (type *) malloc16(sizeof(type)*(num))

uint32_t *inv_table;

void create_inv_table(int n) {
    inv_table = talloc(uint32_t, (1U << n));
    if (!inv_table) return;

    for (int i = 0; i < (1U << n); i++) {
        if (i % 2 == 0) {
            inv_table[i] = 0;
        } else {
            int b = i - 1;
            int t = 1;
            int x = 0;
            for (int j = 0; j < n; j++) {
                x = (x + t) % (1U << n);
                t = (t * (-b)) % (1U << n);
            }
            inv_table[i] = x;
        }
    }
}

uint32_t get_inv(uint32_t x) {
    return inv_table[x];
}


void ring_lrc_encode(int k, int m, int n, char **data, char **coding, int bufsize) {
    // Dummy implementation of LRC ring encoding
    int i;
    int init1 = 0;
    int init2 = 0;
    int init3 = 0;
    uint32_t mask = (1 << n) - 1;
    for (i = 0; i < k; i++) {
        if (i < k / 2) {
            if (init1 == 0) {
                memcpy(coding[0], data[i], bufsize);
                init1 = 1;
            } else {
                uint16_t *dst = (uint16_t *)coding[0];
                uint16_t *src = (uint16_t *)data[i];
                for (int j = 0; j < bufsize / sizeof(uint16_t); j++) {
                    dst[j] += src[j];
                }
            }
        }
        else {
            if (init2 == 0) {
                memcpy(coding[1], data[i], bufsize);
                init2 = 1;
            } else {
                uint16_t *dst = (uint16_t *)coding[1];
                uint16_t *src = (uint16_t *)data[i];
                for (int j = 0; j < bufsize / sizeof(uint16_t); j++) {
                    dst[j] += src[j];
                }
            }
        }

        if (init3 == 0) {
            memcpy(coding[2], data[i], bufsize);
            init3 = 1;
        } else {
            uint32_t *dst = (uint32_t *)coding[2];
            uint32_t *src = (uint32_t *)data[i];
            for (int j = 0; j < bufsize / sizeof(uint32_t); j++) {
                dst[j] = (dst[j] + (2*i+1) * src[j]) & mask;
            }
        }
    }
}

void ring_lrc_encode_sse(int k, int m, int n, char **data, char **coding, int bufsize) {
    // Dummy implementation of LRC ring encoding
    int i;
    int init1 = 0;
    int init2 = 0;
    int init3 = 0;
    uint32_t mask = (1 << n) - 1;
    for (i = 0; i < k; i++) {
        if (i < k / 2) {
            if (init1 == 0) {
                memcpy(coding[0], data[i], bufsize);
                init1 = 1;
            } else {
                uint16_t *dst = (uint16_t *)coding[0];
                uint16_t *src = (uint16_t *)data[i];
                __m128i *dst_sse = (__m128i *)dst;
                __m128i *src_sse = (__m128i *)src;

                for (int j = 0; j < bufsize / sizeof(__m128i); j++) {
                    __m128i d = _mm_load_si128(&dst_sse[j]); // 加载目标数据
                    __m128i s = _mm_load_si128(&src_sse[j]); // 加载源数据
                    d = _mm_add_epi16(d, s);                // 按 16 位加法
                    _mm_store_si128(&dst_sse[j], d);        // 存储结果
                }
            }
        }
        else {
            if (init2 == 0) {
                memcpy(coding[1], data[i], bufsize);
                init2 = 1;
            } else {
                uint16_t *dst = (uint16_t *)coding[1];
                uint16_t *src = (uint16_t *)data[i];
                __m128i *dst_sse = (__m128i *)dst;
                __m128i *src_sse = (__m128i *)src;

                for (int j = 0; j < bufsize / sizeof(__m128i); j++) {
                    __m128i d = _mm_load_si128(&dst_sse[j]); // 加载目标数据
                    __m128i s = _mm_load_si128(&src_sse[j]); // 加载源数据
                    d = _mm_add_epi16(d, s);                // 按 16 位加法
                    _mm_store_si128(&dst_sse[j], d);        // 存储结果
                }
            }
        }

        if (init3 == 0) {
            memset(coding[2], 0, 2 * bufsize);
            init3 = 1;
        }

        uint32_t *dst = (uint32_t *)coding[2];
        uint16_t *src = (uint16_t *)data[i];
        __m128i *dst_sse = (__m128i *)dst;
        uint64_t *src_sse = (uint64_t *)src;
        __m128i zero_vec = _mm_setzero_si128();
        __m128i mask_vec = _mm_set1_epi32(mask);
        __m128i factor_vec = _mm_set1_epi32(2 * i + 1);
        for (int j = 0; j < 2 * bufsize / sizeof(__m128i); j++) {
            __m128i dst_vec = _mm_load_si128(&dst_sse[j]);
            __m128i src_vec = _mm_loadu_si64(&src_sse[j]);
            __m128i src_vec_ext = _mm_unpacklo_epi16(src_vec, zero_vec); // 扩展为 32 位
            __m128i mul_vec = _mm_mullo_epi32(src_vec_ext, factor_vec);
            __m128i add_vec = _mm_add_epi32(dst_vec, mul_vec);
            __m128i result_vec = _mm_and_si128(add_vec, mask_vec); 
            _mm_store_si128(&dst_sse[j], result_vec);
        }
        
    }
}

int *ring_lrc_erasures_to_erased(int k, int *erasures)
{
  int td;
  int t_non_erased;
  int *erased;
  int i;

  td = k+3;
  erased = talloc(int, td);
  if (erased == NULL) return NULL;
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

void ring_lrc_decode_sse(int k, int m, int n, char **data, char **coding, int bufsize, int *erasures) {
    int i;
    int *erased, edd, group1, group2, global;
    uint32_t mask = (1 << n) - 1;
    erased = ring_lrc_erasures_to_erased(k, erasures);

    group1 = 0;
    group2 = 0;
    global = 0;
    for (i = 0; i < k+3; i++) {
        if (erased[i]) {
            if (i < k / 2 || i == k) {
                group1++;
            }else if (i < k || i == k + 1) {
                group2++;
            } else if (i == k + 2) {
                global++;
            }
        }
    }

    if (group1 == 1) {
        if (erased[k]) {
            int init1 = 0;
            for (i = 0; i < k / 2; i++) {
                if (init1 == 0) {
                    memcpy(coding[0], data[i], bufsize);
                    init1 = 1;
                } else {
                    uint16_t *dst = (uint16_t *)coding[0];
                    uint16_t *src = (uint16_t *)data[i];
                    __m128i *dst_sse = (__m128i *)dst;
                    __m128i *src_sse = (__m128i *)src;

                    for (int j = 0; j < bufsize / sizeof(__m128i); j++) {
                        __m128i d = _mm_load_si128(&dst_sse[j]); // 加载目标数据
                        __m128i s = _mm_load_si128(&src_sse[j]); // 加载源数据
                        d = _mm_add_epi16(d, s);                // 按 16 位加法
                        _mm_store_si128(&dst_sse[j], d);        // 存储结果
                    }
                }
            }
        } else {
            int idx;
            for (i = 0; i < k / 2; i++) {
                if (erased[i]) {
                    idx = i;
                    break;
                }
            }

            memcpy(data[idx], coding[0], bufsize);
            
            for(i = 0; i < k / 2; i++) {
                if (i != idx) {
                    uint16_t *dst = (uint16_t *)data[idx];
                    uint16_t *src = (uint16_t *)data[i];
                    __m128i *dst_sse = (__m128i *)dst;
                    __m128i *src_sse = (__m128i *)src;

                    for (int j = 0; j < bufsize / sizeof(__m128i); j++) {
                        __m128i d = _mm_load_si128(&dst_sse[j]); // 加载目标数据
                        __m128i s = _mm_load_si128(&src_sse[j]); // 加载源数据
                        d = _mm_sub_epi16(d, s);                // 按 16 位减法
                        _mm_store_si128(&dst_sse[j], d);        // 存储结果
                    }
                }
            }
        }
    }

    if (group2 == 1) {
        if (erased[k+1]) {
            int init2 = 0;
            for (i = k / 2; i < k; i++) {
                if (init2 == 0) {
                    memcpy(coding[1], data[i], bufsize);
                    init2 = 1;
                } else {
                    uint16_t *dst = (uint16_t *)coding[1];
                    uint16_t *src = (uint16_t *)data[i];
                    __m128i *dst_sse = (__m128i *)dst;
                    __m128i *src_sse = (__m128i *)src;

                    for (int j = 0; j < bufsize / sizeof(__m128i); j++) {
                        __m128i d = _mm_load_si128(&dst_sse[j]); // 加载目标数据
                        __m128i s = _mm_load_si128(&src_sse[j]); // 加载源数据
                        d = _mm_add_epi16(d, s);                // 按 16 位加法
                        _mm_store_si128(&dst_sse[j], d);        // 存储结果
                    }
                }
            }
        } else {
            int idx;
            for (i = k / 2; i < k; i++) {
                if (erased[i]) {
                    idx = i;
                    break;
                }
            }

            memcpy(data[idx], coding[1], bufsize);
            
            for(i = k / 2; i < k; i++) {
                if (i != idx) {
                    uint16_t *dst = (uint16_t *)data[idx];
                    uint16_t *src = (uint16_t *)data[i];
                    __m128i *dst_sse = (__m128i *)dst;
                    __m128i *src_sse = (__m128i *)src;

                    for (int j = 0; j < bufsize / sizeof(__m128i); j++) {
                        __m128i d = _mm_load_si128(&dst_sse[j]); // 加载目标数据
                        __m128i s = _mm_load_si128(&src_sse[j]); // 加载源数据
                        d = _mm_sub_epi16(d, s);                // 按 16 位减法
                        _mm_store_si128(&dst_sse[j], d);        // 存储结果
                    }
                }
            }
        }
    }
    
    if (group1 == 2) {
        if (erased[k]) {
            int idx;
            for (i = 0; i < k / 2; i++) {
                if (erased[i]) {
                    idx = i;
                    break;
                }
            }

            char *tmp = talloc(char, 2 * bufsize);
            memcpy(tmp, coding[2], 2 * bufsize);

            for (i = 0; i < k; i++) {
                if (i != idx) {
                    uint32_t *dst = (uint32_t *)tmp;
                    uint16_t *src = (uint16_t *)data[i];
                    __m128i *src_sse = (__m128i *)src;
                    __m128i *dst_sse = (__m128i *)dst;
                    __m128i zero_vec = _mm_setzero_si128();
                    __m128i mask_vec = _mm_set1_epi32(mask);
                    __m128i factor_vec = _mm_set1_epi32(2 * i + 1);
                    for (int j = 0; j < bufsize / sizeof(__m128i); j++) {
                        __m128i dst_vec = _mm_load_si128(&dst_sse[j]);
                        __m128i src_vec = _mm_loadl_epi64(&src_sse[j]);
                        __m128i src_vec_ext = _mm_unpacklo_epi16(src_vec, zero_vec); // 扩展为 32 位
                        __m128i mul_vec = _mm_mullo_epi32(src_vec_ext, factor_vec);
                        __m128i sub_vec = _mm_sub_epi32(dst_vec, mul_vec);
                        __m128i result_vec = _mm_and_si128(sub_vec, mask_vec);
                        _mm_store_si128(&dst_sse[j], result_vec);
                    }
                }
            }

            uint16_t *dst = (uint16_t *)data[idx];
            uint32_t *src = (uint32_t *)tmp;
            __m128i *src_sse = (__m128i *)src;
            __m128i *dst_sse = (__m128i *)dst;
            for (int j = 0; j < bufsize / sizeof(__m128i); j++) {

            }

            int init1 = 0;
            for (i = 0; i < k / 2; i++) {
                if (init1 == 0) {
                    memcpy(coding[0], data[i], bufsize);
                    init1 = 1;
                } else {
                    uint16_t *dst = (uint16_t *)coding[0];
                    uint16_t *src = (uint16_t *)data[i];
                    __m128i *dst_sse = (__m128i *)dst;
                    __m128i *src_sse = (__m128i *)src;

                    for (int j = 0; j < bufsize / sizeof(__m128i); j++) {
                        __m128i d = _mm_load_si128(&dst_sse[j]); // 加载目标数据
                        __m128i s = _mm_load_si128(&src_sse[j]); // 加载源数据
                        d = _mm_add_epi16(d, s);                // 按 16 位加法
                        _mm_store_si128(&dst_sse[j], d);        // 存储结果
                    }
                }
            }

        }else {

        }
    }

    if (group2 == 2) {
        if (erased[k+1]) {
            int idx;
            for (i = 0; i < k / 2; i++) {
                if (erased[i]) {
                    idx = i;
                    break;
                }
            }

            

            int init2 = 0;
            for (i = k / 2; i < k; i++) {
                if (init2 == 0) {
                    memcpy(coding[1], data[i], bufsize);
                    init2 = 1;
                } else {
                    uint16_t *dst = (uint16_t *)coding[1];
                    uint16_t *src = (uint16_t *)data[i];
                    __m128i *dst_sse = (__m128i *)dst;
                    __m128i *src_sse = (__m128i *)src;

                    for (int j = 0; j < bufsize / sizeof(__m128i); j++) {
                        __m128i d = _mm_load_si128(&dst_sse[j]); // 加载目标数据
                        __m128i s = _mm_load_si128(&src_sse[j]); // 加载源数据
                        d = _mm_add_epi16(d, s);                // 按 16 位加法
                        _mm_store_si128(&dst_sse[j], d);        // 存储结果
                    }
                }
            }
        }else {

        }
    }

    if (global == 1) {
        int init3 = 0;
        for (i = 0; i < k; i++) {
            if (init3 == 0) {
                memset(coding[2], 0, 2 * bufsize);
                init3 = 1;
            }
    
            uint32_t *dst = (uint32_t *)coding[2];
            uint16_t *src = (uint16_t *)data[i];
            __m128i *dst_sse = (__m128i *)dst;
            __m128i *src_sse = (__m128i *)src;
            __m128i zero_vec = _mm_setzero_si128();
            __m128i mask_vec = _mm_set1_epi32(mask);
            __m128i factor_vec = _mm_set1_epi32(2 * i + 1);
            for (int j = 0; j < bufsize / sizeof(__m128i); j++) {
                __m128i dst_vec = _mm_load_si128(&dst_sse[j]);
                __m128i src_vec = _mm_loadl_epi64(&src_sse[j]);
                __m128i src_vec_ext = _mm_unpacklo_epi16(src_vec, zero_vec); // 扩展为 32 位
                __m128i mul_vec = _mm_mullo_epi32(src_vec_ext, factor_vec);
                __m128i add_vec = _mm_add_epi32(dst_vec, mul_vec);
                __m128i result_vec = _mm_and_si128(add_vec, mask_vec); 
                _mm_store_si128(&dst_sse[j], result_vec);
            }
        }
    }
}

int main(int argc, char **argv) {
    int i;
    int k = 8; // Number of data blocks
    int m = 16;
    int n = 20;
    int bufsize = 1024; // Size of each block in bytes
    int iterations = 10000;
    uint32_t seed = 104;
    char **data, **coding, **old_values;
    int *erasures, *erased;

    MOA_Seed(seed);

    data = talloc(char *, k);
    for (i = 0; i < k; i++) {
        data[i] = talloc(char, bufsize);
        MOA_Fill_Random_Region(data[i], bufsize);
        printf("0x%hhx 0x%hhx\n", data[i][0], data[i][1]);
    }

    coding = talloc(char *, 3);
    old_values = talloc(char *, 3);
    for (i = 0; i < 2; i++) {
        coding[i] = talloc(char, bufsize);
        old_values[i] = talloc(char, bufsize);
    }
    coding[2] = talloc(char, 2 * bufsize);

    create_inv_table(n);

    struct timeval start, end; // 用于时间统计

    // 编码操作时间统计
    gettimeofday(&start, NULL); // 开始计时
    for (i = 0; i < iterations; i++) {
        ring_lrc_encode_sse(k, m, n, data, coding, bufsize);
    }
    gettimeofday(&end, NULL); // 结束计时
    double encode_time = get_time_diff(start, end);
    printf("[Ring-based LRC] Encode throughput for %d iterations: %.2f MB/s (%.2f sec)\n",
            iterations, (double)(k * iterations * bufsize / 1024 / 1024) / encode_time, encode_time);
    
    printf("0x%hhx 0x%hhx\n", coding[0][0], coding[0][1]);
    printf("0x%hhx 0x%hhx\n", coding[1][0], coding[1][1]);
    printf("0x%hhx 0x%hhx 0x%hhx 0x%hhx\n", coding[2][0], coding[2][1], coding[2][2], coding[2][3]);
    
    erasures = talloc(int, (3+1));
    erased = talloc(int, (k+3));
    for (int erasure_num = 1; erasure_num <= 1; erasure_num++) {
        for (i = 0; i < 3+k; i++) erased[i] = 0;
        for (i = 0; i < erasure_num; ) {
            erasures[i] = ((unsigned int)MOA_Random_W(16, 1))%(k+3);
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
        for (i = 0; i < iterations; i++) {
            ring_lrc_decode_sse(k, m, n, data, coding, bufsize, erasures);
        }
        gettimeofday(&end, NULL); // 结束计时
        double decode_time = get_time_diff(start, end);
        printf("[Ring-based LRC] Decode throughput for %d iterations with %d erasure: %.2f MB/s (%.2f sec)\n",
            iterations, erasure_num, (double)(k * iterations * bufsize / 1024 / 1024) / decode_time, decode_time);
    
        for (i = 0; i < erasure_num; i++) {
            if (erasures[i] < k) {
                if (memcmp(data[erasures[i]], old_values[i], bufsize)) {
                    fprintf(stderr, "Decoding failed for %d!\n", erasures[i]);
                    exit(1);
                }
            } else {
                if (memcmp(coding[erasures[i]-k], old_values[i], bufsize)) {
                    fprintf(stderr, "Decoding failed for %d!\n", erasures[i]);
                    exit(1);
                }
            }
        }
    }

    return 0;
}