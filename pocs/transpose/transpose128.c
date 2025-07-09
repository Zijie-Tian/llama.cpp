/*********************************************************************
 * 128 × 128 float matrix transpose
 *  Author : ChatGPT demo, 2025-07-09
 *  Build  : aarch64-linux-gnu-gcc -O3 -march=armv8-a+simd -funroll-loops \
 *           transpose128.c -o transpose
 *********************************************************************/
 #include <arm_neon.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <time.h>
 #include <string.h>
 
 #define N 128                 /* 矩阵尺寸 */
 #define REP 100               /* 基准测试重复次数 */
 
 /*---------------------------------------------------------------*/
 /* 4×4 NEON butterfly micro-kernel                               */
 /*---------------------------------------------------------------*/
 static inline void transpose4x4_neon(const float *src, float *dst,
                                      int src_stride, int dst_stride)
 {
     float32x4_t r0 = vld1q_f32(src + 0 * src_stride);
     float32x4_t r1 = vld1q_f32(src + 1 * src_stride);
     float32x4_t r2 = vld1q_f32(src + 2 * src_stride);
     float32x4_t r3 = vld1q_f32(src + 3 * src_stride);
 
     /* 第 1 层：32-bit lane 交错 */
     float32x4x2_t t0 = vtrnq_f32(r0, r1);
     float32x4x2_t t1 = vtrnq_f32(r2, r3);
 
     /* 第 2 层：64-bit 半寄存器交错 */
     float32x4_t c0 = vreinterpretq_f32_u64(
         vcombine_u64(vget_low_u64(vreinterpretq_u64_f32(t0.val[0])),
                      vget_low_u64(vreinterpretq_u64_f32(t1.val[0]))));
     float32x4_t c1 = vreinterpretq_f32_u64(
         vcombine_u64(vget_low_u64(vreinterpretq_u64_f32(t0.val[1])),
                      vget_low_u64(vreinterpretq_u64_f32(t1.val[1]))));
     float32x4_t c2 = vreinterpretq_f32_u64(
         vcombine_u64(vget_high_u64(vreinterpretq_u64_f32(t0.val[0])),
                      vget_high_u64(vreinterpretq_u64_f32(t1.val[0]))));
     float32x4_t c3 = vreinterpretq_f32_u64(
         vcombine_u64(vget_high_u64(vreinterpretq_u64_f32(t0.val[1])),
                      vget_high_u64(vreinterpretq_u64_f32(t1.val[1]))));
 
     vst1q_f32(dst + 0 * dst_stride, c0);
     vst1q_f32(dst + 1 * dst_stride, c1);
     vst1q_f32(dst + 2 * dst_stride, c2);
     vst1q_f32(dst + 3 * dst_stride, c3);
 }
 
 /*---------------------------------------------------------------*/
 /* 128×128 NEON 矩阵转置                                          */
 /*---------------------------------------------------------------*/
 void transpose128x128_neon(const float *src, float *dst)
 {
     for (int i = 0; i < N; i += 4)
         for (int j = 0; j < N; j += 4)
             transpose4x4_neon(src + i * N + j,
                               dst + j * N + i,
                               N, N);
 }
 
 /*---------------------------------------------------------------*/
 /* 128×128 Scalar（最朴素）版本                                   */
 /*---------------------------------------------------------------*/
 void transpose128x128_scalar(const float *src, float *dst)
 {
     for (int i = 0; i < N; ++i)
         for (int j = 0; j < N; ++j)
             dst[j * N + i] = src[i * N + j];
 }
 
 /*---------------------------------------------------------------*/
 static inline double secdiff(const struct timespec *a,
                              const struct timespec *b)
 {
     return (b->tv_sec - a->tv_sec) + 1e-9 * (b->tv_nsec - a->tv_nsec);
 }
 
 /*---------------------------------------------------------------*/
 /* main：生成随机矩阵 → 两种实现 → 校验 → 计时                        */
 /*---------------------------------------------------------------*/
 int main(void)
 {
     float *A, *B_neon, *B_scalar;
     if (posix_memalign((void **)&A, 64, N * N * sizeof(float)) ||
         posix_memalign((void **)&B_neon, 64, N * N * sizeof(float)) ||
         posix_memalign((void **)&B_scalar, 64, N * N * sizeof(float))) {
         perror("alloc");
         return 1;
     }
 
     /* 随机初始化 */
     for (int i = 0; i < N * N; ++i)
         A[i] = (float)rand() / RAND_MAX;
 
     /*-----------------------------------------------------------*/
     struct timespec t0, t1;
     clock_gettime(CLOCK_MONOTONIC, &t0);
     for (int r = 0; r < REP; ++r)
         transpose128x128_neon(A, B_neon);
     clock_gettime(CLOCK_MONOTONIC, &t1);
     double neon_time = secdiff(&t0, &t1) / REP;
 
     clock_gettime(CLOCK_MONOTONIC, &t0);
     for (int r = 0; r < REP; ++r)
         transpose128x128_scalar(A, B_scalar);
     clock_gettime(CLOCK_MONOTONIC, &t1);
     double scalar_time = secdiff(&t0, &t1) / REP;
 
     /*-----------------------------------------------------------*/
     /* 结果校验 */
     if (memcmp(B_neon, B_scalar, N * N * sizeof(float)) != 0) {
         fprintf(stderr, "ERROR: Results differ!\n");
         return 2;
     }
 
     printf("128×128 transpose (float) - per-run average over %d runs:\n", REP);
     printf("  NEON   : %.3f µs\n", neon_time * 1e6);
     printf("  Scalar : %.3f µs  (×%.1f slower)\n",
            scalar_time * 1e6, scalar_time / neon_time);
 
     free(A); free(B_neon); free(B_scalar);
     return 0;
 }
 