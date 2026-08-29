//
// Copyright (C) 2026 Valeriy Zainullin
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ggml.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// runtime check: compiler support + CPUID AMX-TILE/AMX-INT8 + XTILE enabled
// (requests XTILEDATA permission from the Linux kernel on first call);
// can be disabled with GGML_AMX_DISABLE=1 or GGML_AMX=0 in the environment
GGML_API bool ggml_amx_available(void);

// hardware sanity check (config roundtrip, tile load/store roundtrips,
// a tiny tdpbssd); returns 0 when everything works, a bitmask otherwise
GGML_API int  ggml_amx_diag(void);

// force the older INT8 kernel (per-k-block fp16 rescaling) instead of the
// default BF16 kernel; 1 = int8, 0 = bf16 (the default). Testing hook.
GGML_API void ggml_amx_set_int8(int enable);

// 1 when the default BF16 kernel can actually run on this machine: the
// CPU advertises amx_bf16 AND executes tdpbf16ps (some KVM guests only
// fake the CPUID flag); 0 otherwise
GGML_API int  ggml_amx_has_bf16(void);

// single matrix: dst = Aq8 @ W^T
//   W: N rows of quantized weights, w_stride = bytes between rows
//   A: M rows of block_q8_0, a_stride = bytes between rows
//   dst: M x N floats, ldc = floats between rows
GGML_API bool ggml_amx_gemm(int type,
                            int64_t M, int64_t N, int64_t KB,
                            const void * W, int64_t w_stride,
                            const void * A, int64_t a_stride,
                            float * dst, int64_t ldc, int ith, int nth);

// 4d variant used by ggml.c; B must hold Ny rows of block_q8_0 per batch
GGML_API bool ggml_amx_mul_mat_4d(int typeA,
        long Nx, long Ny, long ne00,
        long ne02, long ne03, long ne12, long ne13,
        long nb02, long nb03, long nbw2, long nbw3, long nb2, long nb3,
        const void * A, long strideA,
        const void * B, long strideB,
        float * C, long stride_C, int ith, int nth);

#ifdef __cplusplus
}
#endif
