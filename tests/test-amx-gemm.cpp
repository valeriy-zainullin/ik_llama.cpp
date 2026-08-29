//
// Copyright (C) 2026 Valeriy Zainullin
// MIT license
// SPDX-License-Identifier: MIT
//
// Correctness and benchmark tests for the AMX-INT8 GEMM kernels
// (ggml-amx.cpp). Run without arguments for correctness, with "bench"
// for a throughput measurement on prefill-like shapes.
//

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-amx.h"
#include "ggml-quants.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace {

struct shape_t {
    int64_t m, n, k;
};

float fp16_to_fp32(uint16_t h) {
    ggml_fp16_t x = h;
    return ggml_compute_fp16_to_fp32(x);
}

// round-to-nearest-even float -> bf16 (back to float). Mirrors what the
// bf16 kernel bakes into its inputs, so the reference below checks the
// KERNEL, not the quantization noise.
float fp32_via_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    u &= 0xffff0000u;
    memcpy(&f, &u, 4);
    return f;
}

void run_gemm(int type, int64_t M, int64_t N, int64_t KB,
              const void * W, size_t w_stride,
              const void * A, size_t a_stride,
              float * dst, size_t ldc, int nth) {
    if (nth <= 1) {
        if (!ggml_amx_gemm(type, M, N, KB, W, w_stride, A, a_stride, dst, ldc, 0, 1)) {
            fprintf(stderr, "ggml_amx_gemm failed\n");
            exit(1);
        }
        return;
    }
    std::vector<std::thread> threads;
    for (int i = 1; i < nth; ++i) {
        threads.emplace_back([=] {
            if (!ggml_amx_gemm(type, M, N, KB, W, w_stride, A, a_stride, dst, ldc, i, nth)) {
                fprintf(stderr, "ggml_amx_gemm failed on thread %d\n", i);
                exit(1);
            }
        });
    }
    if (!ggml_amx_gemm(type, M, N, KB, W, w_stride, A, a_stride, dst, ldc, 0, nth)) {
        fprintf(stderr, "ggml_amx_gemm failed on thread 0\n");
        exit(1);
    }
    for (auto & t : threads) t.join();
}

template <typename WT>
int8_t weight_value(const WT * row, int64_t kb, int j);

template <>
int8_t weight_value<block_q4_0>(const block_q4_0 * row, int64_t kb, int j) {
    // ik_llama.cpp q4_0 layout: elements 0..15 -> low nibbles, 16..31 -> high nibbles
    const uint8_t b = row[kb].qs[(j < 16) ? j : (j - 16)];
    const uint8_t nib = (j < 16) ? (b & 0xF) : (b >> 4);
    return (int8_t)(nib) - 8;
}

template <>
int8_t weight_value<block_q8_0>(const block_q8_0 * row, int64_t kb, int j) {
    return row[kb].qs[j];
}

template <typename WT>
bool check_one(int type, int64_t M, int64_t N, int64_t K, int nth, std::mt19937 & rng, bool bf16, double tol) {
    const int64_t KB = K / 32;
    const size_t w_row = (type == GGML_TYPE_Q4_0) ? sizeof(block_q4_0) * KB : sizeof(block_q8_0) * KB;

    std::vector<float> wf((size_t)N * K);
    std::normal_distribution<float> dist(0.f, 1.f);
    for (auto & v : wf) v = dist(rng);
    std::vector<uint8_t> W((size_t)N * w_row);
    for (int64_t n = 0; n < N; ++n) {
        if (type == GGML_TYPE_Q4_0) quantize_row_q4_0(wf.data() + n * K, W.data() + n * w_row, K);
        else                         quantize_row_q8_0(wf.data() + n * K, W.data() + n * w_row, K);
    }

    std::vector<float> af((size_t)M * K);
    for (auto & v : af) v = dist(rng);
    std::vector<uint8_t> A((size_t)M * sizeof(block_q8_0) * KB);
    for (int64_t m = 0; m < M; ++m) {
        quantize_row_q8_0(af.data() + m * K, A.data() + m * sizeof(block_q8_0) * KB, K);
    }

    std::vector<float> dst((size_t)M * N);
    memset(dst.data(), 0xff, dst.size() * sizeof(float));  // NaN: catches missing overwrite on k == 0

    ggml_amx_set_int8(bf16 ? 0 : 1);
    run_gemm(type, M, N, KB, W.data(), w_row, A.data(), sizeof(block_q8_0) * KB, dst.data(), N, nth);

    const char * wrow = (const char *)W.data();
    const block_q8_0 * Au = (const block_q8_0 *)A.data();

    double max_err = 0.0;
    double max_rel = 0.0;
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            const WT * w_row_p = (const WT *)(wrow + (size_t)n * w_row);
            double ref = 0.0;
            if (bf16) {
                // bf16 kernel semantics: both inputs are rounded to bf16
                // (block scale multiplied in first, like the kernel's pack);
                // products then exact, accumulation in double here vs fp32
                // in the tile - that residual is what tolerance covers
                for (int64_t kb = 0; kb < KB; ++kb) {
                    const double da = fp16_to_fp32(Au[m * KB + kb].d);
                    const double dw = fp16_to_fp32(w_row_p[kb].d);
                    for (int j = 0; j < 32; ++j) {
                        ref += (double)fp32_via_bf16((float)(dw * weight_value<WT>(w_row_p, kb, j))) *
                               (double)fp32_via_bf16((float)(da * Au[m * KB + kb].qs[j]));
                    }
                }
            } else {
                for (int64_t kb = 0; kb < KB; ++kb) {
                    int32_t dot = 0;
                    for (int j = 0; j < 32; ++j) {
                        dot += (int32_t)weight_value<WT>(w_row_p, kb, j) * (int32_t)Au[m * KB + kb].qs[j];
                    }
                    ref += (double)fp16_to_fp32(Au[m * KB + kb].d) *
                           (double)fp16_to_fp32(w_row_p[kb].d) * (double)dot;
                }
            }
            const double got = dst[(size_t)m * N + n];
            const double err = std::fabs(got - ref);
            const double rel = err / std::max(1.0, std::fabs(ref));
            max_err = std::max(max_err, err);
            max_rel = std::max(max_rel, rel);
            if (!(err <= tol * std::max(1.0, std::fabs(ref)))) {
                fprintf(stderr, "FAIL %s %s M=%lld N=%lld K=%lld nth=%d at [%lld,%lld]: got %f ref %f\n",
                        bf16 ? "BF16" : "INT8",
                        type == GGML_TYPE_Q4_0 ? "Q4_0" : "Q8_0",
                        (long long)M, (long long)N, (long long)K, nth,
                        (long long)m, (long long)n, got, ref);
                return 1;
            }
        }
    }
    printf("PASS %s %-5s M=%-4lld N=%-4lld K=%-4lld nth=%d  max_err=%.3g max_rel=%.3g\n",
           bf16 ? "BF16" : "INT8",
           type == GGML_TYPE_Q4_0 ? "Q4_0" : "Q8_0",
           (long long)M, (long long)N, (long long)K, nth, max_err, max_rel);
    return 0;
}

int run_correctness() {
    if (!ggml_amx_available()) {
        printf("AMX is not available on this machine/build, nothing to test\n");
        return 0;
    }

    std::mt19937 rng(1234);
    const shape_t shapes[] = {
        {8,   32,  64}, {9,   17,  32}, {15,  16,  96}, {16,   1,  64},
        {17,  33, 128}, {31,  47,  32}, {32,  32, 256}, {33,  16,  64},
        {47,  63,  96}, {64, 100, 128}, {100, 31,  32}, {512, 512, 512},
    };
    int rc = 0;
    // the default bf16 kernel, then the legacy int8 one; both must match
    // their own reference (bf16 tolerance covers fp32-accumulation drift)
    for (int mode = 0; mode < 2; ++mode) {
        const bool bf16 = (mode == 0);
        const double tol = bf16 ? 1e-1 : 1e-4;
        if (bf16) printf("== correctness: BF16 kernel (default) ==\n");
        else      printf("== correctness: INT8 kernel (GGML_AMX_INT8=1) ==\n");
        for (const auto & s : shapes) {
            rc |= check_one<block_q4_0>(GGML_TYPE_Q4_0, s.m, s.n, s.k, 1, rng, bf16, tol);
            rc |= check_one<block_q8_0>(GGML_TYPE_Q8_0, s.m, s.n, s.k, 1, rng, bf16, tol);
        }
        const shape_t mt_shapes[] = {{9, 17, 32}, {31, 47, 96}, {64, 100, 128}, {512, 512, 512}};
        for (const auto & s : mt_shapes) {
            rc |= check_one<block_q4_0>(GGML_TYPE_Q4_0, s.m, s.n, s.k, 4, rng, bf16, tol);
            rc |= check_one<block_q8_0>(GGML_TYPE_Q8_0, s.m, s.n, s.k, 4, rng, bf16, tol);
        }
    }
    if (rc == 0) printf("all AMX GEMM correctness tests passed\n");
    return rc;
}

void bench_one(int type, int64_t M, int64_t N, int64_t K, int nth, std::mt19937 & rng, bool bf16) {
    const int64_t KB = K / 32;
    const size_t w_row = (type == GGML_TYPE_Q4_0) ? sizeof(block_q4_0) * KB : sizeof(block_q8_0) * KB;

    std::vector<float> wf((size_t)N * K), af((size_t)M * K);
    std::normal_distribution<float> dist(0.f, 1.f);
    for (auto & v : wf) v = dist(rng);
    for (auto & v : af) v = dist(rng);

    std::vector<uint8_t> W((size_t)N * w_row), A((size_t)M * sizeof(block_q8_0) * KB);
    for (int64_t n = 0; n < N; ++n) {
        if (type == GGML_TYPE_Q4_0) quantize_row_q4_0(wf.data() + n * K, W.data() + n * w_row, K);
        else                         quantize_row_q8_0(wf.data() + n * K, W.data() + n * w_row, K);
    }
    for (int64_t m = 0; m < M; ++m) {
        quantize_row_q8_0(af.data() + m * K, A.data() + m * sizeof(block_q8_0) * KB, K);
    }
    std::vector<float> dst((size_t)M * N);

    ggml_amx_set_int8(bf16 ? 0 : 1);
    run_gemm(type, M, N, KB, W.data(), w_row, A.data(), sizeof(block_q8_0) * KB, dst.data(), N, nth);

    const int reps = 3;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
        run_gemm(type, M, N, KB, W.data(), w_row, A.data(), sizeof(block_q8_0) * KB, dst.data(), N, nth);
    }
    auto t1 = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(t1 - t0).count() / reps;
    const double gops = 2.0 * (double)M * N * K / dt / 1e9;
    printf("BENCH %s %-5s M=%-5lld N=%-5lld K=%-5lld nth=%-3d  %8.2f ms  %8.1f GOPS\n",
           bf16 ? "BF16" : "INT8",
           type == GGML_TYPE_Q4_0 ? "Q4_0" : "Q8_0",
           (long long)M, (long long)N, (long long)K, nth, dt * 1e3, gops);
}

int run_bench() {
    if (!ggml_amx_available()) {
        printf("AMX is not available on this machine/build, nothing to bench\n");
        return 0;
    }
    std::mt19937 rng(42);
    unsigned hw = std::thread::hardware_concurrency();
    const int nth = (int)std::max(1u, std::min(hw, 64u));
    printf("bench with %d threads\n", nth);
    for (int mode = 0; mode < 2; ++mode) {
        const bool bf16 = (mode == 0);
        printf(bf16 ? "-- bf16 kernel --\n" : "-- int8 kernel --\n");
        bench_one(GGML_TYPE_Q4_0, 512, 4096, 4096, nth, rng, bf16);
        bench_one(GGML_TYPE_Q8_0, 512, 4096, 4096, nth, rng, bf16);
        bench_one(GGML_TYPE_Q4_0, 512, 4096, 11008, nth, rng, bf16);
        bench_one(GGML_TYPE_Q4_0, 2048, 4096, 4096, nth, rng, bf16);
    }
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    // ggml_init fills ggml_table_f32_f16 used by GGML_FP16_TO_FP32 in the kernel
    struct ggml_init_params ip = {};
    ip.mem_size   = 0;
    ip.mem_buffer = NULL;
    ip.no_alloc   = true;
    struct ggml_context * ctx = ggml_init(ip);

    printf("ggml_amx_available: %s\n", ggml_amx_available() ? "yes" : "no");
    printf("ggml_cpu_has_amx_int8: %d\n", ggml_cpu_has_amx_int8());
    int rc;
    if (argc > 1 && strcmp(argv[1], "diag") == 0) {
        rc = ggml_amx_diag();
    } else if (argc > 1 && strcmp(argv[1], "bench") == 0) {
        rc = run_bench();
    } else {
        rc = run_correctness();
    }
    ggml_free(ctx);
    return rc;
}
