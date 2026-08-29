//
// Copyright (C) 2026 Valeriy Zainullin
// MIT license
// SPDX-License-Identifier: MIT
//
// AMX-INT8 GEMM for Q4_0/Q8_0 weights x Q8_0 activations.
//
// A reader's guide, assuming you know SIMD "on fingers" and nothing about
// AMX.
//
// -- What AMX is ----------------------------------------------------------
//
// Classic SIMD works "sideways": a 512-bit register holds 16 floats (or 64
// int8) in one line, and every instruction applies one operation to the
// whole line. Great for element-wise work; awkward for matrix multiply,
// which is the core of every neural network - a row-times-column dot
// product summed over hundreds of elements, performed millions of times.
// SIMD builds each dot product from many shuffle + multiply-add steps.
//
// AMX (Advanced Matrix Extensions) attacks it from the other side. The CPU
// gains eight "tiles" - think of them as small 2D matrix registers, each
// up to 16 rows x 64 BYTES, sitting next to the SIMD registers - plus
// load/store/multiply instructions for them:
//
//     tileloadd  tmm2, (base, row_step)   // copy a 2D image from RAM into a tile
//     tdpbssd    tmm4, tmm2, tmm0         // tmm4 += tmm2 x tmm0, int8 in, int32 out
//     tilestored tmm4, (base, row_step)   // copy the tile back to RAM
//
// One tdpbssd multiplies a 16x32 int8 matrix by a 32x16 one: 8192
// multiply-adds in a single instruction, where a good AVX-512 kernel
// spends dozens of instructions. Matrix multiply IS the neural network
// workload, so AMX turns the CPU into a small matrix accelerator; this
// file is mostly about feeding it.
//
// Three things you must know beyond that:
//  * tiles are dumb until configured. The CPU keeps a per-thread "settings
//    page" (TILECFG) holding the SHAPE of each tile: rows and columns in
//    bytes. LDTILECFG loads the page, and every tile instruction validates
//    its operands against those shapes - usefully so: feed it the wrong
//    tile and it faults instead of computing nonsense (we rely on that to
//    catch A/B swaps early).
//  * tiles multiply int8 and accumulate int32 - precisely the arithmetic
//    of ggml's quantized formats. The float part of the numbers lives in
//    per-block fp16 "scales" (Q8_0: 32 int8 values + 1 fp16 scale per
//    block; Q4_0: the same, but 4-bit). The int tiles produce the dot
//    products; the scales are applied afterwards with plain AVX-512
//    (acc_4tiles) to build the float result.
//  * THERE IS A SECOND MULTIPLY: tdpbf16ps, bf16 in / fp32 accumulate.
//    Same tile machinery, no per-block scales needed - which turns out to
//    matter a lot. The int8 path pays "helper work" on EVERY 32-wide block
//    of K: repack B into VNNI order (~100 uops) and rescale the int32 C
//    tiles into fp32 (~450 uops). Counted over a whole GEMM, the C-tile
//    rescale alone was ~25% of the entire time budget - the mighty
//    multiplier spent most of its life waiting for the cleaning crew.
//    The bf16 path bakes each block scale into the weight ONCE at pack
//    time (dequantize q4/q8 -> bf16), so the fp32 C tile can accumulate
//    the WHOLE K sum inside the tile engine: no rescale per block, and
//    the per-k-block float work disappears. That is the schedule where
//    AMX actually saturates. The price: weights and activations are
//    rounded to bf16 (8-bit mantissa, ~0.2% per value), so results differ
//    from the int8 path in the last decimal - fine for inference, verify
//    with perplexity. This is the default kernel; GGML_AMX_INT8=1 in the
//    environment selects the older int8 path for comparison.
//  * the whole tile state is per-thread, and the OS keeps it disabled
//    until the thread explicitly asks for permission (amx_probe_cpu below).
//
// -- Words used below -----------------------------------------------------
//
//   row step - how many bytes to skip to get from one matrix row to the
//       next: row r+1 begins at "row r plus the row step". Manuals call it
//       "stride"; our parameter names keep that spelling, the prose says
//       "row step". If rows sit back-to-back the row step equals the row
//       length - but ggml's q8_0 activations are 34-byte blocks (2 bytes
//       of fp16 scale + 32 payload), so their row step is 34*KB bytes,
//       deliberately NOT a round multiple: this matters, see quirk 2.
//   VNNI - the k-ordering tdpbssd wants on the B (weights) side: k comes
//       in groups of 4 stored side by side. A picture in pack_b_tile.
//   scale - the fp16 multiplier attached to every quantized block; the
//       int8 dot product times the row/column scales gives the float
//       result.
//
// -- What this file does --------------------------------------------------
//
// tdpbssd wants its inputs in tile-shaped, partly reordered buffers, while
// ggml hands us "rows of bytes". So the file is a shovel:
//
//   pack_b_tile()      - repack 16 weight rows into the VNNI order B wants
//   pack_b_tile_bf16() - dequantize 16 weight rows to a plain bf16 tile
//   unpack_a_tail()    - copy activation rows into a padded 16x32B A image
//   convert_a_bf16()   - dequantize the whole activation matrix to bf16
//   acc_one_tile()     - scale finished int32 tiles into dst (int8 path)
//   amx_gemm_impl()    - int8 loop: multiply + per-block rescale
//   amx_gemm_bf16_impl() - bf16 loop: multiply, rescale-free
//
// -- The quirks that shaped this file -------------------------------------
//
// Evidence for all three: quirks.md in the repo root; the experiments are
// runnable via `test-amx-gemm diag`.
//
//   1. GCC implements the tile intrinsics as bare, unordered asm; we emit
//      our own asm chained through a token (see "the token trick" below).
//   2. A tile loaded straight from the activation matrix (row step 34*KB
//      bytes, not a multiple of 32) MULTIPLIES CORRECTLY, but a plain
//      load->store->memcmp roundtrip of the same load comes back wrong
//      (row 0 fine, later rows differ). So: full A tiles are loaded
//      directly from the activation matrix (verified by dpbssd-based
//      probes), while M-tails go through a zero-padded staging buffer -
//      which they need anyway. And never trust a stored->memcmp check
//      with a non-multiple row step; verify content by multiplying.
//   3. The four C tiles are indexed 0..3 and two code paths must agree on
//      what the index MEANS (row half vs column half). When they disagree
//      the result is not garbage: it is plausible numbers scaled by the
//      wrong half's scales, or exact zeros where the other half was empty.
//      That looked exactly like "AMX lost a tile" and cost us days. The
//      layouts are written out below; keep them in sync.
//
//

#include "ggml-amx.h"

#include <algorithm>
#include <atomic>
#include <csetjmp>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "ggml-impl.h"
#include "ggml-quants.h"

#if defined(__AMX_TILE__) && defined(__AMX_INT8__) && defined(__AMX_BF16__) && defined(__AVX512F__) && defined(__AVX512BW__) && defined(__F16C__) && defined(__FMA__)
#define GGML_AMX_AVAILABLE_COMPILER 1
#else
#define GGML_AMX_AVAILABLE_COMPILER 0
#endif

namespace {

// Tile geometry, in bytes. We compute a 32x32 output block per step:
// B (the weights) is split into two 16-column halves, A (the activations)
// into two 16-row halves, so 2x2 = 4 C tiles cover one 32x32 block
// (quirk 3 from the header: the numbering of those 4 tiles is a trap).
enum {
    TILE_M = 16,   // rows per A tile / rows per C tile
    TILE_N = 16,   // columns per B half = int32 values per C-tile row
    TILE_K = 32,   // int8 values consumed per multiply (32 bytes of A)
    VNNI_BLK = 4,  // dpbssd packs K in groups of 4 (see VNNI below)
};

// Tile indices must be macros, not enum/constexpr: in GCC (through 12) the
// AMX intrinsics are preprocessor macros expanding to inline asm with token
// stringification ("%%tmm"#dst), so the index has to be an integer literal
// after preprocessing; an enum constant produces %tmmTMM0 and breaks the
// assembler (GCC 13+ prints an "i" operand with %c instead, and clang
// implements real builtins that accept any constant expression).
// llama.cpp does the same: #define TMM0..TMM7 in ggml-cpu/amx/common.h.
#define TMM0 0
#define TMM1 1
#define TMM2 2
#define TMM3 3
#define TMM4 4
#define TMM5 5
#define TMM6 6
#define TMM7 7

#if GGML_AMX_AVAILABLE_COMPILER

// The token trick. Why this wall of asm instead of _tile_loadd(...)? (quirk 1)
//
// In plain C++ the compiler tracks every value: it knows a load reads
// memory written just above, and that a multiply must not jump ahead of
// the zeroing of its destination. For AVX-512 it models the registers.
// For AMX it models NOTHING: all current GCC versions (12 through master;
// clang has real builtins) compile the tile intrinsics to bare `asm
// volatile` blobs - tileloadd declares neither a memory operand nor a
// clobber, tdpbssd/tilezero have no operands at all, and the tmm
// registers do not exist for the register allocator. The GCC manual says
// volatile asm may be REORDERED and need not stay consecutive, so the
// optimizer is free to hoist a tile load above the stores that fill its
// buffer (loading stale bytes) or sink a tilezero below the multiply
// (multiplying garbage) - we observed exactly such all-zero tiles with
// GCC 12.
//
// The fix is a chain the optimizer cannot cut: every tile operation
// takes a `unsigned tok` as a read-WRITE operand ("+r"(tok)) and passes
// it to the next one. That is a genuine data dependency, and data
// dependencies are sacred - no scheduler or LICM will break them. loadd/
// stored additionally carry a "memory" clobber so they cannot float above
// the buffer writes (the pack) or below the reads of results. llama.cpp's
// AMX code calls the intrinsics directly and gets away with it on clang;
// on GCC the same hazard applies to their code.
//
// Two more details you will trip over if you edit these:
//  * the wrappers are TWO-LEVEL macros: `#dst` stringifies the argument
//    verbatim ("TMM0"), so the tile number must first be expanded to the
//    literal 0 by an outer macro - the same trick GCC uses in its own
//    _tile_*_internal macros;
//  * operand numbering: tok is the "+r" output (%0), so the base and the
//    row step are %1/%2, and the row step is cast to (size_t) - without
//    it some GCC builds tried to load a 64-bit address register from a
//    32-bit value.
// The {att|intel} braces mean "first spelling for AT&T assembly, second
// for Intel" - GCC picks the dialect it was configured for.
#if defined(__GNUC__) && !defined(__clang__)
#define AMX_TILE_LOADD(tok, dst, base, stride) \
    AMX_TILE_LOADD_I(tok, dst, base, stride)
#define AMX_TILE_LOADD_I(tok, dst, base, stride) \
    __asm__ __volatile__ ("{tileloadd\t(%1,%2,1), %%tmm"#dst"|tileloadd\t%%tmm"#dst", [%1+%2*1]}" \
                          : "+r" (tok) : "r" (base), "r" ((size_t)(stride)) : "memory")
#define AMX_TILE_ZERO(tok, dst) \
    AMX_TILE_ZERO_I(tok, dst)
#define AMX_TILE_ZERO_I(tok, dst) \
    __asm__ __volatile__ ("tilezero\t%%tmm"#dst : "+r" (tok))
#define AMX_TILE_DPBSSD(tok, dst, src1, src2) \
    AMX_TILE_DPBSSD_I(tok, dst, src1, src2)
#define AMX_TILE_DPBSSD_I(tok, dst, src1, src2) \
    __asm__ __volatile__ ("{tdpbssd\t%%tmm"#src2", %%tmm"#src1", %%tmm"#dst"|tdpbssd\t%%tmm"#dst", %%tmm"#src1", %%tmm"#src2"}" \
                          : "+r" (tok))
// tdpbf16ps: same three-operand shape as tdpbssd (dst += src1 x src2), but
// src1/src2 hold bf16 (2 bytes per value) and dst accumulates fp32. The k
// depth is fixed at 16 bf16 values (32 bytes) - no VNNI grouping, so B
// tiles are plain row-major.
#define AMX_TILE_DPBF16PS(tok, dst, src1, src2) \
    AMX_TILE_DPBF16PS_I(tok, dst, src1, src2)
#define AMX_TILE_DPBF16PS_I(tok, dst, src1, src2) \
    __asm__ __volatile__ ("{tdpbf16ps\t%%tmm"#src2", %%tmm"#src1", %%tmm"#dst"|tdpbf16ps\t%%tmm"#dst", %%tmm"#src1", %%tmm"#src2"}" \
                          : "+r" (tok))
#define AMX_TILE_STORED(tok, src, base, stride) \
    AMX_TILE_STORED_I(tok, src, base, stride)
#define AMX_TILE_STORED_I(tok, src, base, stride) \
    __asm__ __volatile__ ("{tilestored\t%%tmm"#src", (%1,%2,1)|tilestored\t[%1+%2*1], %%tmm"#src"}" \
                          : "+r" (tok) : "r" (base), "r" ((size_t)(stride)) : "memory")
#else
#define AMX_TILE_LOADD(tok, dst, base, stride) _tile_loadd(dst, base, stride)
#define AMX_TILE_ZERO(tok, dst)                _tile_zero(dst)
#define AMX_TILE_DPBSSD(tok, dst, src1, src2)  _tile_dpbssd(dst, src1, src2)
#define AMX_TILE_DPBF16PS(tok, dst, src1, src2) _tile_dpbf16ps(dst, src1, src2)
#define AMX_TILE_STORED(tok, src, base, stride) _tile_stored(src, base, stride)
#endif


#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif

bool amx_probe_cpu() {
    // env switches first: the AMX path can be turned off without a rebuild
    // (GGML_AMX_DISABLE=1 or GGML_AMX=0) to fall back to the IQK kernels
    const char * e = getenv("GGML_AMX_DISABLE");
    if (e && e[0] != '\0') return false;
    e = getenv("GGML_AMX");
    if (e && e[0] == '0') return false;

    // CPUID leaf 7 subleaf 0, EDX: bit 22 = AMX-TILE, bit 25 = AMX-INT8
    uint32_t eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
    const bool has_tile = (edx >> 22) & 1;
    const bool has_int8 = (edx >> 25) & 1;
    if (!has_tile || !has_int8) return false;

    // XCR0 must save/restore both XTILECFG (bit 17) and XTILEDATA (bit 18);
    // AMX is an XSAVE feature, unlike AVX512 this cannot be checked via CPUID alone
    uint32_t xcr0_lo, xcr0_hi;
    asm volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
    if (((xcr0_lo >> 17) & 3) != 3) return false;

    // Linux: XTILEDATA defaults to disabled in XFD; executing a tile
    // instruction without requesting permission raises #NM (SIGFPE/SIGILL
    // depending on kernel). The request is per-thread, so probe() must be
    // called once per thread before the first tile op.
#if defined(__linux__)
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0) return false;
#endif
    return true;
}

// The tile settings page. A tile is not just a register: the CPU also
// remembers a SHAPE for each of the 8 tiles (rows, and columns in BYTES),
// and every tile instruction validates its operands against these shapes.
// This 64-byte struct is the image of that internal state; its field
// offsets are fixed by the ISA (not chosen by us): byte 0 palette, byte 1
// start_row, bytes 16..31 colsb[8] (u16), bytes 48..55 rows[8] (u8).
// alignas(64) is required by LDTILECFG. Verified against the hardware via
// an sttilecfg roundtrip in the diag mode.
struct alignas(64) amx_tilecfg_t {
    uint8_t  palette;
    uint8_t  start_row;
    uint8_t  reserved0[14];
    uint16_t colsb[8];
    uint8_t  reserved1[16];
    uint8_t  rows[8];
    uint8_t  reserved2[8];
};

// Tile settings for one kernel flavor. The two flavors share the file's
// 2-2-4 pattern but disagree on tile shapes, and the CPU keeps exactly ONE
// config page per thread - so the last-loaded flavor is remembered and the
// page is reloaded on switches (calling both kernels alternately is legal).
//
//   int8 (tdpbssd):    tmm0/tmm1 :  8 x 64 B - B, VNNI order (see pack_b_tile)
//                      tmm2/tmm3 : 16 x 32 B - A, int8 row-major
//                      tmm4..7   : 16 x 64 B - C, int32 (4 bytes/col)
//   bf16 (tdpbf16ps):  tmm0..3   : 16 x 32 B - B and A, bf16 row-major
//                      tmm4..7   : 16 x 64 B - C, fp32
// (8 tiles x <=1024 B each = the whole palette, exactly, for both.)
void amx_tile_config_init(bool bf16) {
    // Shape is per-thread state, like the tiles themselves: OpenMP worker
    // threads are born mid-loop and never pass through ggml_amx_available()
    // again, so every kernel entry ensures its own thread is configured.
    static thread_local int loaded = -1;   // -1 none, 0 int8, 1 bf16
    const int want = bf16 ? 1 : 0;
    if (loaded == want) return;

    // palette 1 is the only layout current CPUs implement; it caps the
    // grand total at 1 KB of tile storage, and rows are u8 / columns u16
    // BYTES. Our casting:
    amx_tilecfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.palette   = 1;
    cfg.start_row = 0;
    if (!bf16) {
        for (int i = 0; i < 2; ++i) { cfg.rows[i] = 8;  cfg.colsb[i] = 64; }
        for (int i = 2; i < 4; ++i) { cfg.rows[i] = 16; cfg.colsb[i] = 32; }
    } else {
        for (int i = 0; i < 4; ++i) { cfg.rows[i] = 16; cfg.colsb[i] = 32; }
    }
    for (int i = 4; i < 8; ++i) { cfg.rows[i] = 16; cfg.colsb[i] = 64; }
#if defined(__GNUC__) && !defined(__clang__)
    // own asm instead of __builtin_ia32_ldtilecfg: the "memory" clobber
    // guarantees the config stores above cannot be moved below or dropped
    __asm__ __volatile__ ("ldtilecfg (%0)" :: "r" (&cfg) : "memory");
#else
    _tile_loadconfig(&cfg);
#endif
    loaded = want;
}

struct alignas(64) amx_buffers_t {
    // Scratch space, reused by every call on this thread (thread_local, so
    // OpenMP workers each own a private copy and never contend).
    //
    // B tiles as tileloadd sees them: 8 rows x 64 B of VNNI-packed int8 -
    // byte [kg][n*4+b] holds qs(w_n, kg*4+b), see pack_b_tile for the
    // picture. The 16 fp16 weight scales ride along at byte 512 so the
    // cvtph_ps reads below are a single load. 576 = 512 + 64.
    int8_t  b_tiles[2][576];
    // A tiles: 16 rows x 32 B, plain row-major - the staging image used ONLY
    // for M-tails (mr < 16). Full tiles load straight from the activation
    // matrix (row step a_stride = 34*KB): the multiply consumes such loads
    // correctly (quirk 2), so staging would be pure overhead. The zero
    // padding here is the M-tail filler.
    int8_t  a_tiles[2][512];
    // C tiles as tilestored writes them: 16 rows x 16 int32 each, four per
    // 32x32 output block. Two sets so the pipeline can overlap: while the
    // current window multiplies, the previous window's ints are being
    // scaled and accumulated into dst from the other set (see the m-loop).
    // The bf16 kernel reuses them as fp32 storage (same 1024 B) for the
    // rare N/M-tail windows that cannot be stored straight into dst.
    int32_t c_tiles[2][4][256];
    int     flip;
    // bf16 path: the activation matrix converted once per call from q8_0
    // into plain bf16 rows (2*KB*16 bytes per row). Grown on demand; the
    // capacity lives across calls so steady-state inference never mallocs.
    std::unique_ptr<char[]> a_bf16;
    size_t  a_bf16_cap = 0;
};

// static storage duration => zero-initialized (.tbss): flip starts at 0,
// no explicit init needed.
thread_local amx_buffers_t g_amx_buf;

// The next two macros are ordinary AVX-256 shuffling glue for the pack.
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
#define SHUFFLE_EPI32(a, b, mask) \
    _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), mask))

// 8x8 transpose of int32 lanes (unpack/shuffle/permute, the textbook
// three-stage square-matrix transpose). It is the heart of the VNNI pack:
// see pack_b_tile for why "column j of the transpose" IS VNNI row j.
inline void transpose_8x8_32bit(__m256i * v, __m256i * v1) {
    v1[0] = _mm256_unpacklo_epi32(v[0], v[1]);
    v1[1] = _mm256_unpackhi_epi32(v[0], v[1]);
    v1[2] = _mm256_unpacklo_epi32(v[2], v[3]);
    v1[3] = _mm256_unpackhi_epi32(v[2], v[3]);
    v1[4] = _mm256_unpacklo_epi32(v[4], v[5]);
    v1[5] = _mm256_unpackhi_epi32(v[4], v[5]);
    v1[6] = _mm256_unpacklo_epi32(v[6], v[7]);
    v1[7] = _mm256_unpackhi_epi32(v[6], v[7]);

    v[0] = SHUFFLE_EPI32(v1[0], v1[2], 0x44);
    v[1] = SHUFFLE_EPI32(v1[0], v1[2], 0xee);
    v[2] = SHUFFLE_EPI32(v1[4], v1[6], 0x44);
    v[3] = SHUFFLE_EPI32(v1[4], v1[6], 0xee);
    v[4] = SHUFFLE_EPI32(v1[1], v1[3], 0x44);
    v[5] = SHUFFLE_EPI32(v1[1], v1[3], 0xee);
    v[6] = SHUFFLE_EPI32(v1[5], v1[7], 0x44);
    v[7] = SHUFFLE_EPI32(v1[5], v1[7], 0xee);

    v1[0] = _mm256_permute2f128_si256(v[2], v[0], 0x02);
    v1[1] = _mm256_permute2f128_si256(v[3], v[1], 0x02);
    v1[2] = _mm256_permute2f128_si256(v[6], v[4], 0x02);
    v1[3] = _mm256_permute2f128_si256(v[7], v[5], 0x02);
    v1[4] = _mm256_permute2f128_si256(v[2], v[0], 0x13);
    v1[5] = _mm256_permute2f128_si256(v[3], v[1], 0x13);
    v1[6] = _mm256_permute2f128_si256(v[6], v[4], 0x13);
    v1[7] = _mm256_permute2f128_si256(v[7], v[5], 0x13);
}

// Packs one k-block of 16 weight rows into the layout tdpbssd wants
// (VNNI), plus the 16 fp16 block scales appended after the data.
//
// Why pack at all? tdpbssd computes C[r][n] += sum over k of A[r][k]*B[k][n],
// but it walks the k dimension of B in groups of 4 (VNNI_BLK). So B must
// be stored as 8 rows (kg = k-group index), each row holding, for every
// weight column n, the 4 k-bytes of that group side by side:
//
//   row kg of the tile:  [ w0.k3 w0.k2 w0.k1 w0.k0 | w1.k3 w1.k2 ... ]^VNNI
//   byte address:        kg*64 + n*4 + b   ==   qs(w_n, kg*4 + b)
//
//   (b counts within the group; draw it as 16 groups of 4 bytes per row.)
// Feed tdpbssd an A tile row (32 consecutive k-bytes) and this B, and the
// 8 kg-rows contribute the 8 four-term sub-sums of the 32-element dot.
//
// The pack itself is cute: a q8_0 payload row is 32 bytes = 8 int32 lanes,
// so 8 weight rows form an 8x8 int32 matrix; transposing it puts "the same
// byte position j*4..j*4+3 of every row" into row j - which is exactly
// VNNI row j. Two h-halves cover weight columns 0..7 and 8..15 of the
// 16-column tile (a C row holds 16 int32 columns = 64 B = two halves).
//
// q4_0 nibble convention is SPLIT (same in ik_llama.cpp and upstream
// llama.cpp): quantized elements 0..15 live in the low nibbles of
// bytes 0..15, elements 16..31 in the high nibbles of the same bytes -
// hence the single 16-byte load, the srli_epi16 pair and the -8 offset
// (nibbles are unsigned, q4_0 values are value-8). Rows beyond nv are
// zero-filled and their scales zeroed, so tdpbssd simply adds 0 for
// partial N blocks.
template <typename WT>
inline void pack_b_tile(int8_t * tile, const char * rows, int64_t w_stride, int64_t ki, int nv) {
    constexpr bool IS_Q4 = std::is_same<WT, block_q4_0>::value;
    constexpr int   BS    = IS_Q4 ? 18 : 34;
    const __m256i zero = _mm256_setzero_si256();
    __m256i v[8], t[8];

    for (int h = 0; h < 2; ++h) {
        for (int j = 0; j < 8; ++j) {
            const int idx = h * 8 + j;
            if (idx < nv) {
                const char * p = rows + idx * w_stride + ki * BS + 2;
                if (IS_Q4) {
                    const __m128i n = _mm_loadu_si128((const __m128i *)p);
                    const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(n, 4), n);
                    v[j] = _mm256_sub_epi8(_mm256_and_si256(_mm256_set1_epi8(0xF), bytes), _mm256_set1_epi8(8));
                } else {
                    v[j] = _mm256_loadu_si256((const __m256i *)p);
                }
            } else {
                v[j] = zero;
            }
        }
        transpose_8x8_32bit(v, t);
        for (int j = 0; j < 8; ++j) {
            _mm256_storeu_si256((__m256i *)(tile + j * 64 + h * 32), t[j]);
        }
    }

    ggml_fp16_t * d = (ggml_fp16_t *)(tile + 512);
    for (int j = 0; j < 16; ++j) {
        d[j] = (j < nv) ? *(const ggml_fp16_t *)(rows + j * w_stride + ki * BS) : (ggml_fp16_t)0;
    }
}

// Stages mr rows of one k-block of the q8_0 activation matrix into a
// compact 16x32B tile image (rows beyond mr stay zero).
//
// Used ONLY for M-tails (mr < 16): full tiles load straight from the
// activation matrix, but a partial tile needs the rows beyond mr zeroed -
// garbage there would pollute every dot product of rows that do not exist.
// A plain memset+copy also happens to sidestep the strided-readback
// weirdness entirely (quirk 2). The +2 / ki*34 offsets skip the fp16
// scale at the start of each block (block layout: 2 bytes of scale, then
// 32 payload bytes).
inline void unpack_a_tail(int8_t * tile, const char * a_rows, int64_t a_stride, int64_t ki, int mr) {
    memset(tile, 0, 512);
    for (int r = 0; r < mr; ++r) {
        memcpy(tile + r * 32, a_rows + r * a_stride + ki * 34 + 2, 32);
    }
}

// ---------------------------------------------------------------------------
// bf16 path helpers. tdpbf16ps consumes PLAIN row-major bf16 (no VNNI), so
// B tiles are 16 k-rows x 16 weight columns of dequantized weights.
// ---------------------------------------------------------------------------

// 8x8 transpose of 16-bit lanes (__m128i). Quadrant workhorse for the
// 16x16 bf16 transpose below; same three unpack stages as the int32
// version, one level narrower.
inline void transpose_8x8_16bit(__m128i * v) {
    __m128i t[8];
    for (int i = 0; i < 4; ++i) {
        t[2*i+0] = _mm_unpacklo_epi16(v[2*i], v[2*i+1]);
        t[2*i+1] = _mm_unpackhi_epi16(v[2*i], v[2*i+1]);
    }
    __m128i u[8];
    for (int i = 0; i < 2; ++i) {
        u[4*i+0] = _mm_unpacklo_epi32(t[4*i+0], t[4*i+2]);
        u[4*i+1] = _mm_unpackhi_epi32(t[4*i+0], t[4*i+2]);
        u[4*i+2] = _mm_unpacklo_epi32(t[4*i+1], t[4*i+3]);
        u[4*i+3] = _mm_unpackhi_epi32(t[4*i+1], t[4*i+3]);
    }
    for (int i = 0; i < 4; ++i) {
        v[2*i+0] = _mm_unpacklo_epi64(u[2*i], u[2*i+1]);
        v[2*i+1] = _mm_unpackhi_epi64(u[2*i], u[2*i+1]);
    }
}

// Packs one k-chunk (16 values) of 16 weight rows into a 16x16 bf16 tile,
// k-major (tile row k, column n). Values are fully dequantized: the fp16
// block scale is multiplied in HERE, once - that is what frees the C tile
// from per-block rescaling later. Rows beyond nv are zero-filled so
// partial N blocks simply contribute zeros.
//
// The weight matrix stores rows (n) one after another, but the tile wants
// rows (k) - a transpose is unavoidable. Route: dequantize each weight row
// into a staging row (n-major), then transpose the 16x16 with four 8x8
// quadrant transposes. Not the cheapest possible shuffle sequence, but it
// runs once per weight per call and the kernels below are where the time
// goes.
template <typename WT>
inline void pack_b_tile_bf16(int8_t * tile, const char * rows, int64_t w_stride, int64_t ki, int nv) {
    constexpr bool IS_Q4 = std::is_same<WT, block_q4_0>::value;
    constexpr int   BS   = IS_Q4 ? 18 : 34;
    alignas(64) uint16_t stage[16][16];   // n-major: stage[j] = 16 bf16 of weight row j

    for (int j = 0; j < 16; ++j) {
        if (j >= nv) {
            _mm256_storeu_si256((__m256i *)stage[j], _mm256_setzero_si256());
            continue;
        }
        const char * p = rows + j * w_stride + ki * BS;
        const float d = GGML_FP16_TO_FP32(*(const ggml_fp16_t *)p);
        __m512 v;
        if (IS_Q4) {
            // split nibbles (see pack_b_tile): 16 values out of 8 bytes
            const __m128i b   = _mm_loadl_epi64((const __m128i *)(p + 2));
            const __m128i lo  = _mm_cvtepu8_epi16(b);                              // elems 0..7
            const __m128i hi  = _mm_and_si128(_mm_srli_epi16(lo, 4), _mm_set1_epi16(0xF)); // 8..15
            // nibbles are 0..15, q4_0 values are value-8; packs (signed) keeps -8..7
            const __m128i q   = _mm_packs_epi16(_mm_sub_epi16(lo, _mm_set1_epi16(8)),
                                                _mm_sub_epi16(hi, _mm_set1_epi16(8)));
            v = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q));
        } else {
            v = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(p + 2))));
        }
        v = _mm512_mul_ps(v, _mm512_set1_ps(d));
        _mm256_storeu_si256((__m256i *)stage[j], _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
    }

    // 16x16 transpose in four 8x8 quadrants: rows 0..7 x cols 0..7, 8..15
    // and rows 8..15 likewise. Output row k of the tile = [lo part | hi part].
    __m128i q00[8], q01[8], q10[8], q11[8];
    for (int r = 0; r < 8; ++r) {
        q00[r] = _mm_loadu_si128((const __m128i *)(stage[r] + 0));
        q01[r] = _mm_loadu_si128((const __m128i *)(stage[r] + 8));
        q10[r] = _mm_loadu_si128((const __m128i *)(stage[8 + r] + 0));
        q11[r] = _mm_loadu_si128((const __m128i *)(stage[8 + r] + 8));
    }
    transpose_8x8_16bit(q00);
    transpose_8x8_16bit(q01);
    transpose_8x8_16bit(q10);
    transpose_8x8_16bit(q11);
    for (int r = 0; r < 8; ++r) {
        _mm_storeu_si128((__m128i *)(tile + r * 32 +  0), q00[r]);
        _mm_storeu_si128((__m128i *)(tile + r * 32 + 16), q01[r]);
        _mm_storeu_si128((__m128i *)(tile + (8 + r) * 32 +  0), q10[r]);
        _mm_storeu_si128((__m128i *)(tile + (8 + r) * 32 + 16), q11[r]);
    }
}

// Stages mr rows of one k-chunk of the bf16 activation image into a padded
// 16x32B tile (rows beyond mr zero). M-tails only, same story as
// unpack_a_tail: garbage rows would pollute nonexistent outputs.
inline void unpack_a_tail_bf16(int8_t * tile, const char * a_rows, int64_t a_stride, int64_t ki, int mr) {
    memset(tile, 0, 512);
    for (int r = 0; r < mr; ++r) {
        memcpy(tile + r * 32, a_rows + r * a_stride + ki * 32, 32);
    }
}

// Converts the q8_0 activation matrix into plain bf16 rows (scales baked
// in), M rows x K values dense. Returns a thread-local buffer that is
// grown on demand and reused across calls; nullptr on OOM (the caller
// falls back to the int8 kernel rather than dying).
//
// Why from q8_0 and not straight from the f32 source: ggml.c already
// hands us q8_0 rows (the same quantization the int8 path and, within
// rounding, the IQK path consume), so numerics stay comparable between
// kernels at zero extra plumbing.
inline char * convert_a_bf16(amx_buffers_t & buf, const char * a_q8, int64_t a_stride, int64_t M, int64_t KB) {
    const size_t need = (size_t)M * KB * 32;
    if (buf.a_bf16_cap < need) {
        buf.a_bf16.reset(new (std::nothrow) char[need]);
        buf.a_bf16_cap = buf.a_bf16 ? need : 0;
        if (!buf.a_bf16) return nullptr;
    }
    char * out = buf.a_bf16.get();
    for (int64_t m = 0; m < M; ++m) {
        const char * src = a_q8 + m * a_stride;
        uint16_t * drow = (uint16_t *)(out + m * KB * 32);
        for (int64_t kb = 0; kb < KB; ++kb) {
            const __m512 vd = _mm512_set1_ps(GGML_FP16_TO_FP32(*(const ggml_fp16_t *)(src + kb * 34)));
            const __m256i q = _mm256_loadu_si256((const __m256i *)(src + kb * 34 + 2));
            const __m512 f0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_castsi256_si128(q))), vd);
            const __m512 f1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_extracti128_si256(q, 1))), vd);
            _mm256_storeu_si256((__m256i *)(drow + kb * 32),      _mm512_cvtps_ph(f0, _MM_FROUND_TO_NEAREST_INT));
            _mm256_storeu_si256((__m256i *)(drow + kb * 32 + 16), _mm512_cvtps_ph(f1, _MM_FROUND_TO_NEAREST_INT));
        }
    }
    return out;
}

// dst[mb+r][nb+c] += tile[r][c] for the t-th tile of a 2x2 window. bf16
// twin of acc_one_tile with NOTHING to scale: the tiles already hold fp32
// K-sums (scales were baked into the inputs), so this is a plain masked
// add. Only N/M-tail windows take this path - full tiles are tilestored
// straight into dst by the kernel (every element is written exactly once,
// no accumulation into dst is needed). The tile index mapping is the SAME
// 2x2 layout as the int8 kernel: t bit0 = row half, t bit1 = column half -
// quirk 3, the layout that cost us days when it disagreed.
inline void acc_bf16_tile(const int32_t * tile, int t,
                          int64_t m0, int64_t n0, int nv0, int nv1,
                          float * dst, int64_t ldc, int64_t M) {
    const int64_t mb = m0 + 16 * (t & 1);
    const int64_t nb = n0 + 16 * (t >> 1);
    const int nv = (t >> 1) ? nv1 : nv0;
    if (nv <= 0) return;
    const int mr = (int)std::min<int64_t>(16, M - mb);
    if (mr <= 0) return;

    const __mmask16 pmask = nv >= 16 ? (__mmask16)0xffff : (__mmask16)((1u << nv) - 1u);
    const float * trow = (const float *)tile;
    float * drow = dst + mb * ldc + nb;
    for (int r = 0; r < mr; ++r) {
        _mm512_mask_storeu_ps(drow + r * ldc, pmask,
            _mm512_add_ps(_mm512_maskz_loadu_ps(pmask, drow + r * ldc),
                          _mm512_loadu_ps(trow + r * 16)));
    }
}

// dst[mb+r][nb+c] (+)= tile[t][r][c] * d_a[mb+r] * d_w[nb+c]
//
// Scales ONE raw int32 C tile (the t-th of a 2x2 window) into the float
// destination. The tile indices follow the STORED layout of the kernel
// loop (t bit0 = A half = row block, t bit1 = B half = column block):
// cur[0]=A0B0, cur[1]=A1B0, cur[2]=A0B1, cur[3]=A1B1. Getting this
// mapping wrong does NOT produce garbage — it multiplies the wrong halves
// with the other half's scale, which looked exactly like a "lost tile"
// (exact zeros when the other A half was empty) and sent the debugging
// far off course; keep the two layouts identical.
// nv0/nv1 = valid columns in each B half (pmask prevents writing past the
// N tail), M-mb caps the rows. is_acc=false on the first k-block (dst
// holds a NaN sentinel, must be overwritten, not read); later k-blocks
// read-modify-write only the masked columns. Scales: fp16 d_w per column
// (from the pack), fp16 d_a per row/block read straight from the q8_0
// activation blocks.
//
// This is deliberately a ONE-tile function: the kernel calls it between
// multiply/store pairs so the plain AVX-512 work overlaps the (asynchronous)
// execution of the next tdpbssd — see the m-window in amx_gemm_impl.
template <typename WT>
inline void acc_one_tile(int32_t (*cbuf)[256], int t,
                         const __m512 & vdw0, const __m512 & vdw1,
                         const char * a_rows, int64_t a_stride, int64_t ki, bool is_acc,
                         int64_t m0, int64_t n0, int nv0, int nv1,
                         float * dst, int64_t ldc, int64_t M) {
    const int64_t mb = m0 + 16 * (t & 1);
    const int64_t nb = n0 + 16 * (t >> 1);
    const int nv = (t >> 1) ? nv1 : nv0;
    if (nv <= 0) return;
    const int mr = (int)std::min<int64_t>(16, M - mb);
    if (mr <= 0) return;

    // column half is t bit1: scales must follow the SAME bit (mixing this
    // up multiplies the B1 half by B0's fp16 scales — plausible-looking
    // wrong values, or exact zeros when nv1 == 0 and b_tiles[1] is memset)
    const __m512 vdw = (t >> 1) ? vdw1 : vdw0;
    const __mmask16 pmask = nv >= 16 ? (__mmask16)0xffff : (__mmask16)((1u << nv) - 1u);
    const int32_t * tile = cbuf[t];
    const char * arow = a_rows + mb * a_stride;
    float * drow = dst + mb * ldc + nb;

    for (int r = 0; r < mr; ++r) {
        const float da = GGML_FP16_TO_FP32(*(const ggml_fp16_t *)(arow + r * a_stride + ki * 34));
        const __m512 vd  = _mm512_mul_ps(vdw, _mm512_set1_ps(da));
        const __m512 vi  = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i *)(tile + r * 16)));
        __m512 vs = is_acc ? _mm512_maskz_loadu_ps(pmask, drow + r * ldc) : _mm512_set1_ps(0.f);
        vs = _mm512_fmadd_ps(vi, vd, vs);
        _mm512_mask_storeu_ps(drow + r * ldc, pmask, vs);
    }
}

// all four tiles of a window (used for the final window after the m-loop)
template <typename WT>
inline void acc_4tiles(int32_t (*cbuf)[256],
                       const __m512 & vdw0, const __m512 & vdw1,
                       const char * a_rows, int64_t a_stride, int64_t ki, bool is_acc,
                       int64_t m0, int64_t n0, int nv0, int nv1,
                       float * dst, int64_t ldc, int64_t M) {
    for (int t = 0; t < 4; ++t) {
        acc_one_tile<WT>(cbuf, t, vdw0, vdw1, a_rows, a_stride, ki, is_acc, m0, n0, nv0, nv1, dst, ldc, M);
    }
}

template <typename WT>
void amx_gemm_impl(int64_t M, int64_t N, int64_t KB,
                   const char * w, int64_t w_stride,
                   const char * a_q8, int64_t a_stride,
                   float * dst, int64_t ldc,
                   int64_t ith, int64_t nth) {
    amx_tile_config_init(/*bf16=*/false);
    amx_buffers_t & buf = g_amx_buf;
#if defined(__GNUC__) && !defined(__clang__)
    unsigned amx_tok = 0;
#endif

    const int64_t NB = (N + 2 * TILE_N - 1) / (2 * TILE_N);
    const int64_t n_my = (NB + nth - 1) / nth;
    const int64_t nb0 = std::min(n_my * ith, NB);
    const int64_t nb1 = std::min(nb0 + n_my, NB);

    for (int64_t nb = nb0; nb < nb1; ++nb) {
        const int64_t n0 = nb * 2 * TILE_N;
        // valid columns in the two B halves of this 32-column block.
        // nv1 is clamped to >= 0: for the last partial block (e.g. N=33)
        // N-n0-TILE_N is negative, and a negative nv1 would flow into
        // (1u << nv)-1 in acc_4tiles (UB) — keep the clamp even though
        // acc_4tiles also checks nv <= 0.
        const int nv0 = (int)std::min<int64_t>(TILE_N, N - n0);
        const int nv1 = (int)std::max<int64_t>(0, std::min<int64_t>(TILE_N, N - n0 - TILE_N));
        const char * w0 = w + n0 * w_stride;
        const char * w1 = w + (n0 + TILE_N) * w_stride;

        for (int64_t ki = 0; ki < KB; ++ki) {
            pack_b_tile<WT>(buf.b_tiles[0], w0, w_stride, ki, nv0);
            if (nv1 > 0) {
                pack_b_tile<WT>(buf.b_tiles[1], w1, w_stride, ki, nv1);
            } else {
                // b_tiles is thread_local and persists across calls: without
                // the memset a previous call's B data would linger and be
                // multiplied into columns the current matrix does not have
                memset(buf.b_tiles[1], 0, sizeof(buf.b_tiles[1]));
            }
            // Both B tiles go in back-to-back here; the first multiply that
            // consumes them is only a few tile ops later (llama.cpp's amx
            // kernel does the same, and our tile snapshots proved tiles keep
            // their content between ops). The loads carry a "memory" clobber,
            // so they cannot float above the pack stores.
            AMX_TILE_LOADD(amx_tok, TMM0, buf.b_tiles[0], TILE_N * VNNI_BLK);
            AMX_TILE_LOADD(amx_tok, TMM1, buf.b_tiles[1], TILE_N * VNNI_BLK);
            const __m512 vdw0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(buf.b_tiles[0] + 512)));
            const __m512 vdw1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(buf.b_tiles[1] + 512)));

            int32_t (*cbuf)[256] = nullptr;
            int64_t m_prev = 0;

            for (int64_t m0 = 0; m0 < M; m0 += 2 * TILE_M) {
                int32_t (*cur)[256] = buf.c_tiles[buf.flip];
                const int64_t m1 = m0 + TILE_M;
                const int mr0 = (int)std::min<int64_t>(TILE_M, M - m0);
                const int mr1 = (int)std::max<int64_t>(0, std::min<int64_t>(TILE_M, M - m1));
                //
                // A-tile sources. Full 16-row tiles are loaded STRAIGHT from
                // the q8_0 activation matrix with the natural row step
                // (a_stride = 34*KB): our own experiments (diag 7b snapshots,
                // old step-7 t=0, the old kernel's M=16 N=1 case) showed the
                // multiply consumes such loads correctly - the "corrupted
                // stride" was a memcmp-readback lie only (quirk 2). Only
                // M-tails (mr < 16) still go through the padded staging
                // buffer, which doubles as the zero filler.
                //
                // The schedule below is the whole point: tile ops (load,
                // zero, multiply) are queued into the asynchronous tile
                // engine; the CPU only stalls where a result is needed
                // (tilestored, and acc reading stored memory). So after each
                // store we immediately run the PREVIOUS window's float
                // scaling (acc_one_tile) - plain AVX-512 work that overlaps
                // the next multiply. This mirrors llama.cpp's amx/mmq.cpp.
                // The C-tile mapping (bit0 = row half, bit1 = column half)
                // MUST stay in sync with acc_one_tile - quirk 3.
                if (mr0 == TILE_M) {
                    AMX_TILE_LOADD(amx_tok, TMM2, a_q8 + m0 * a_stride + ki * 34 + 2, a_stride);
                } else {
                    unpack_a_tail(buf.a_tiles[0], a_q8 + m0 * a_stride, a_stride, ki, mr0);
                    AMX_TILE_LOADD(amx_tok, TMM2, buf.a_tiles[0], TILE_K);
                }
                AMX_TILE_ZERO(amx_tok, TMM4);

                if (mr1 == TILE_M) {
                    AMX_TILE_LOADD(amx_tok, TMM3, a_q8 + m1 * a_stride + ki * 34 + 2, a_stride);
                } else if (mr1 > 0) {
                    unpack_a_tail(buf.a_tiles[1], a_q8 + m1 * a_stride, a_stride, ki, mr1);
                    AMX_TILE_LOADD(amx_tok, TMM3, buf.a_tiles[1], TILE_K);
                } else {
                    // no rows in the second half: multiplying against a zero
                    // tile yields zeros, which acc then skips anyway
                    AMX_TILE_ZERO(amx_tok, TMM3);
                }
                AMX_TILE_ZERO(amx_tok, TMM5);
                AMX_TILE_ZERO(amx_tok, TMM6);
                AMX_TILE_ZERO(amx_tok, TMM7);

                AMX_TILE_DPBSSD(amx_tok, TMM4, TMM2, TMM0);
                AMX_TILE_STORED(amx_tok, TMM4, cur[0], TILE_N * sizeof(int32_t));
                if (cbuf) {
                    acc_one_tile<WT>(cbuf, 0, vdw0, vdw1, a_q8, a_stride, ki, ki > 0, m_prev, n0, nv0, nv1, dst, ldc, M);
                }

                AMX_TILE_DPBSSD(amx_tok, TMM5, TMM3, TMM0);
                AMX_TILE_STORED(amx_tok, TMM5, cur[1], TILE_N * sizeof(int32_t));
                if (cbuf) {
                    acc_one_tile<WT>(cbuf, 1, vdw0, vdw1, a_q8, a_stride, ki, ki > 0, m_prev, n0, nv0, nv1, dst, ldc, M);
                }

                AMX_TILE_DPBSSD(amx_tok, TMM6, TMM2, TMM1);
                AMX_TILE_STORED(amx_tok, TMM6, cur[2], TILE_N * sizeof(int32_t));
                if (cbuf) {
                    acc_one_tile<WT>(cbuf, 2, vdw0, vdw1, a_q8, a_stride, ki, ki > 0, m_prev, n0, nv0, nv1, dst, ldc, M);
                }

                AMX_TILE_DPBSSD(amx_tok, TMM7, TMM3, TMM1);
                AMX_TILE_STORED(amx_tok, TMM7, cur[3], TILE_N * sizeof(int32_t));
                if (cbuf) {
                    acc_one_tile<WT>(cbuf, 3, vdw0, vdw1, a_q8, a_stride, ki, ki > 0, m_prev, n0, nv0, nv1, dst, ldc, M);
                }

                cbuf   = cur;
                m_prev = m0;
                buf.flip ^= 1;
            }

            if (cbuf) {
                acc_4tiles<WT>(cbuf, vdw0, vdw1, a_q8, a_stride, ki, ki > 0, m_prev, n0, nv0, nv1, dst, ldc, M);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The bf16 kernel (default). Same GEMM, but every input is dequantized to
// bf16 ONCE (scales baked in), so the fp32 C tiles accumulate the WHOLE K
// sum inside the tile engine - no per-k-block rescale, no VNNI repack, no
// int32-to-float bridge. The helper work that dominated the int8 path
// simply does not exist here.
//
// Loop order differs from the int8 kernel on purpose. There the C tiles
// are throwaway per k-block (rescaled into dst after every block), so the
// k-loop lives OUTSIDE the m-loop and the rescale pipeline hides between
// stores. Here the C tiles ARE the answer (they accumulate across all K),
// so the 32x32 output window is the unit of work: walk all of K feeding
// packed B chunks and activation rows to the engine, then drain the four
// finished tiles. The CPU only waits at the tilestored drain - everything
// else (packing the next chunk, converting A once) overlaps the multiplies.
template <typename WT>
bool amx_gemm_bf16_impl(int64_t M, int64_t N, int64_t KB,
                        const char * w, int64_t w_stride,
                        const char * a_q8, int64_t a_stride,
                        float * dst, int64_t ldc,
                        int64_t ith, int64_t nth) {
    amx_tile_config_init(/*bf16=*/true);
    amx_buffers_t & buf = g_amx_buf;
#if defined(__GNUC__) && !defined(__clang__)
    unsigned amx_tok = 0;
#endif

    // One bf16 image of the activations per thread. Threads re-convert the
    // whole matrix redundantly (there is no barrier inside this call) - at
    // ~0.3 uop per value that is noise next to the GEMM itself.
    char * abf = convert_a_bf16(buf, a_q8, a_stride, M, KB);
    if (!abf) return false;
    const int64_t arow = KB * 64;   // bf16 bytes per activation row = 2*K

    const int64_t NB = (N + 2 * TILE_N - 1) / (2 * TILE_N);
    const int64_t n_my = (NB + nth - 1) / nth;
    const int64_t nb0 = std::min(n_my * ith, NB);
    const int64_t nb1 = std::min(nb0 + n_my, NB);

    for (int64_t nb = nb0; nb < nb1; ++nb) {
        const int64_t n0 = nb * 2 * TILE_N;
        // valid columns in the two halves of this 32-column block (clamped
        // like the int8 kernel: a negative nv1 would poison (1u << nv)-1)
        const int nv0 = (int)std::min<int64_t>(TILE_N, N - n0);
        const int nv1 = (int)std::max<int64_t>(0, std::min<int64_t>(TILE_N, N - n0 - TILE_N));
        const char * w0 = w + n0 * w_stride;
        const char * w1 = w + (n0 + TILE_N) * w_stride;

        for (int64_t m0 = 0; m0 < M; m0 += 2 * TILE_M) {
            const int64_t m1 = m0 + TILE_M;
            const int mr0 = (int)std::min<int64_t>(TILE_M, M - m0);
            const int mr1 = (int)std::max<int64_t>(0, std::min<int64_t>(TILE_M, M - m1));

            // the accumulators are zeroed ONCE per window: they must survive
            // all KB chunks (each chunk adds its partial K-sum into them)
            AMX_TILE_ZERO(amx_tok, TMM4);
            AMX_TILE_ZERO(amx_tok, TMM5);
            AMX_TILE_ZERO(amx_tok, TMM6);
            AMX_TILE_ZERO(amx_tok, TMM7);

            for (int64_t ki = 0; ki < KB; ++ki) {
                pack_b_tile_bf16<WT>(buf.b_tiles[0], w0, w_stride, ki, nv0);
                if (nv1 > 0) {
                    pack_b_tile_bf16<WT>(buf.b_tiles[1], w1, w_stride, ki, nv1);
                } else {
                    // b_tiles is thread_local and persists across calls;
                    // a dead half must not leak data from earlier calls
                    memset(buf.b_tiles[1], 0, 512);
                }
                AMX_TILE_LOADD(amx_tok, TMM0, buf.b_tiles[0], 32);
                AMX_TILE_LOADD(amx_tok, TMM1, buf.b_tiles[1], 32);

                // A tiles come straight from the bf16 image (dense rows, row
                // step arow - a multiple of 32, so not even the readback
                // quirk 2 applies); M-tails through the padded staging buffer
                if (mr0 == TILE_M) {
                    AMX_TILE_LOADD(amx_tok, TMM2, abf + m0 * arow + ki * 32, arow);
                } else {
                    unpack_a_tail_bf16(buf.a_tiles[0], abf + m0 * arow, arow, ki, mr0);
                    AMX_TILE_LOADD(amx_tok, TMM2, buf.a_tiles[0], 32);
                }
                if (mr1 == TILE_M) {
                    AMX_TILE_LOADD(amx_tok, TMM3, abf + m1 * arow + ki * 32, arow);
                } else if (mr1 > 0) {
                    unpack_a_tail_bf16(buf.a_tiles[1], abf + m1 * arow, arow, ki, mr1);
                    AMX_TILE_LOADD(amx_tok, TMM3, buf.a_tiles[1], 32);
                } else {
                    // no rows in the second half: a zero tile keeps the
                    // multiply quiet (its results are skipped below)
                    AMX_TILE_ZERO(amx_tok, TMM3);
                }

                // tmm4 = A0*B0, tmm5 = A1*B0, tmm6 = A0*B1, tmm7 = A1*B1
                // (the SAME 2x2 mapping as the int8 kernel - quirk 3)
                AMX_TILE_DPBF16PS(amx_tok, TMM4, TMM2, TMM0);
                AMX_TILE_DPBF16PS(amx_tok, TMM5, TMM3, TMM0);
                AMX_TILE_DPBF16PS(amx_tok, TMM6, TMM2, TMM1);
                AMX_TILE_DPBF16PS(amx_tok, TMM7, TMM3, TMM1);
            }

            // Drain. Full tiles are stored straight into dst - every output
            // element is produced exactly once (the whole K lives in one
            // window), so overwrite, not accumulate. Tail tiles detour
            // through a C buffer and land via masked add in acc_bf16_tile.
            for (int t = 0; t < 4; ++t) {
                const int64_t mb = m0 + 16 * (t & 1);
                const int64_t nc = n0 + 16 * (t >> 1);
                const int nv = (t >> 1) ? nv1 : nv0;
                const int mr = (int)std::min<int64_t>(TILE_M, M - mb);
                if (nv <= 0 || mr <= 0) continue;
                if (nv == TILE_N && mr == TILE_M) {
                    // the tile index must be a literal token for the asm
                    // (#src stringification) - four explicit calls, no TMM4+t
                    switch (t) {
                    case 0: AMX_TILE_STORED(amx_tok, TMM4, dst + mb * ldc + nc, ldc * sizeof(float)); break;
                    case 1: AMX_TILE_STORED(amx_tok, TMM5, dst + mb * ldc + nc, ldc * sizeof(float)); break;
                    case 2: AMX_TILE_STORED(amx_tok, TMM6, dst + mb * ldc + nc, ldc * sizeof(float)); break;
                    case 3: AMX_TILE_STORED(amx_tok, TMM7, dst + mb * ldc + nc, ldc * sizeof(float)); break;
                    }
                } else {
                    switch (t) {
                    case 0: AMX_TILE_STORED(amx_tok, TMM4, buf.c_tiles[0][0], 64); break;
                    case 1: AMX_TILE_STORED(amx_tok, TMM5, buf.c_tiles[0][1], 64); break;
                    case 2: AMX_TILE_STORED(amx_tok, TMM6, buf.c_tiles[0][2], 64); break;
                    case 3: AMX_TILE_STORED(amx_tok, TMM7, buf.c_tiles[0][3], 64); break;
                    }
                    acc_bf16_tile(buf.c_tiles[0][t], t, m0, n0, nv0, nv1, dst, ldc, M);
                }
            }
        }
    }
    return true;
}

// hardware sanity check; returns a bitmask, 0 = everything works:
// bit0: tile config roundtrip mismatch
// bit1: tmm0 (8x64)    load/store roundtrip mismatch
// bit2: tmm2 (16x32)   load/store roundtrip mismatch
// bit3: tmm4 (16x64)   load/store roundtrip mismatch
// bit4: tiny tdpbssd wrong result
int amx_diag_impl() {
    // the diag exercises the INT8 tile shapes (bitmasks documented in
    // quirks.md; the bf16 path is verified by the correctness suite)
    amx_tile_config_init(/*bf16=*/false);
    int rc = 0;
    unsigned tok = 0;

    alignas(64) amx_tilecfg_t cfg2;
    memset(&cfg2, 0, sizeof(cfg2));
#if defined(__GNUC__) && !defined(__clang__)
    __asm__ __volatile__ ("sttilecfg (%0)" :: "r" (&cfg2) : "memory");
#else
    _tile_storeconfig(&cfg2);
#endif
    printf("diag: config roundtrip: palette=%u start_row=%u rows={%u,%u,%u,%u} colsb={%u,%u,%u,%u}\n",
           cfg2.palette, cfg2.start_row,
           cfg2.rows[0], cfg2.rows[2], cfg2.rows[4], cfg2.rows[7],
           cfg2.colsb[0], cfg2.colsb[2], cfg2.colsb[4], cfg2.colsb[7]);
    if (cfg2.palette != 1 || cfg2.rows[0] != 8 || cfg2.colsb[0] != 64 ||
        cfg2.rows[2] != 16 || cfg2.colsb[2] != 32 ||
        cfg2.rows[4] != 16 || cfg2.colsb[4] != 64) {
        printf("diag: CONFIG MISMATCH\n");
        rc |= 1;
    }

    // load/store roundtrip through one tile of each configured shape
    {
        alignas(64) uint8_t pat[1024], out[1024];
        for (int i = 0; i < 1024; ++i) pat[i] = (uint8_t)(i * 7 + 1);

        memset(out, 0, sizeof(out));
        AMX_TILE_LOADD(tok, TMM0, pat, 64);
        AMX_TILE_STORED(tok, TMM0, out, 64);
        const bool ok0 = memcmp(pat, out, 8 * 64) == 0;
        printf("diag: tmm0 (8x64) roundtrip: %s\n", ok0 ? "OK" : "FAIL");
        if (!ok0) rc |= 2;

        memset(out, 0, sizeof(out));
        AMX_TILE_LOADD(tok, TMM2, pat, 32);
        AMX_TILE_STORED(tok, TMM2, out, 32);
        const bool ok2 = memcmp(pat, out, 16 * 32) == 0;
        printf("diag: tmm2 (16x32) roundtrip: %s\n", ok2 ? "OK" : "FAIL");
        if (!ok2) rc |= 4;

        memset(out, 0, sizeof(out));
        AMX_TILE_LOADD(tok, TMM4, pat, 64);
        AMX_TILE_STORED(tok, TMM4, out, 64);
        const bool ok4 = memcmp(pat, out, 16 * 64) == 0;
        printf("diag: tmm4 (16x64) roundtrip: %s\n", ok4 ? "OK" : "FAIL");
        if (!ok4) rc |= 8;

        // roundtrip the remaining tiles (kernel uses tmm1 and tmm3 as well)
        const int rem[5][2] = {{TMM1, 8}, {TMM3, 32}, {TMM5, 64}, {TMM6, 64}, {TMM7, 64}};
        for (int t = 0; t < 5; ++t) {
            const int tmm = rem[t][0];
            const int colsb = rem[t][1];
            const int rows = (colsb == 32) ? 16 : (tmm == TMM1 ? 8 : 16);
            memset(out, 0, sizeof(out));
            switch (tmm) {
            case TMM1: AMX_TILE_LOADD(tok, TMM1, pat, colsb); AMX_TILE_STORED(tok, TMM1, out, colsb); break;
            case TMM3: AMX_TILE_LOADD(tok, TMM3, pat, colsb); AMX_TILE_STORED(tok, TMM3, out, colsb); break;
            case TMM5: AMX_TILE_LOADD(tok, TMM5, pat, colsb); AMX_TILE_STORED(tok, TMM5, out, colsb); break;
            case TMM6: AMX_TILE_LOADD(tok, TMM6, pat, colsb); AMX_TILE_STORED(tok, TMM6, out, colsb); break;
            case TMM7: AMX_TILE_LOADD(tok, TMM7, pat, colsb); AMX_TILE_STORED(tok, TMM7, out, colsb); break;
            }
            const bool ok = memcmp(pat, out, (size_t)rows * colsb) == 0;
            printf("diag: tmm%d (%dx%d) roundtrip: %s\n", tmm, rows, colsb, ok ? "OK" : "FAIL");
            if (!ok) rc |= 128;
        }
    }

    // tdpbssd for every A x B combination (tmm2/tmm3 x tmm0/tmm1)
    {
        alignas(64) uint8_t at[512], bt[512];
        alignas(64) int32_t ct[256];
        const int a_tmm[2] = {TMM2, TMM3};
        const int b_tmm[2] = {TMM0, TMM1};
        for (int ia = 0; ia < 2; ++ia) {
            for (int ib = 0; ib < 2; ++ib) {
                memset(at, 0, sizeof(at));
                memset(bt, 0, sizeof(bt));
                for (int r = 0; r < 16; ++r) at[r * 32 + 0] = (uint8_t)(r + 1);
                for (int n = 0; n < 16; ++n) bt[0 * 64 + n * 4 + 0] = (uint8_t)(n + 1);
                memset(ct, 0, sizeof(ct));
                switch (ia * 2 + ib) {
                case 0: AMX_TILE_LOADD(tok, TMM2, at, 32); AMX_TILE_LOADD(tok, TMM0, bt, 64); AMX_TILE_ZERO(tok, TMM4); AMX_TILE_DPBSSD(tok, TMM4, TMM2, TMM0); AMX_TILE_STORED(tok, TMM4, ct, 64); break;
                case 1: AMX_TILE_LOADD(tok, TMM2, at, 32); AMX_TILE_LOADD(tok, TMM1, bt, 64); AMX_TILE_ZERO(tok, TMM4); AMX_TILE_DPBSSD(tok, TMM4, TMM2, TMM1); AMX_TILE_STORED(tok, TMM4, ct, 64); break;
                case 2: AMX_TILE_LOADD(tok, TMM3, at, 32); AMX_TILE_LOADD(tok, TMM0, bt, 64); AMX_TILE_ZERO(tok, TMM4); AMX_TILE_DPBSSD(tok, TMM4, TMM3, TMM0); AMX_TILE_STORED(tok, TMM4, ct, 64); break;
                case 3: AMX_TILE_LOADD(tok, TMM3, at, 32); AMX_TILE_LOADD(tok, TMM1, bt, 64); AMX_TILE_ZERO(tok, TMM4); AMX_TILE_DPBSSD(tok, TMM4, TMM3, TMM1); AMX_TILE_STORED(tok, TMM4, ct, 64); break;
                }
                int bad = 0;
                for (int r = 0; r < 16 && bad < 2; ++r)
                    for (int n = 0; n < 16 && bad < 2; ++n)
                        if (ct[r * 16 + n] != (r + 1) * (n + 1)) {
                            printf("diag: mul A=tmm%d B=tmm%d C[%d][%d]=%d expect %d\n",
                                   a_tmm[ia], b_tmm[ib], r, n, ct[r * 16 + n], (r + 1) * (n + 1));
                            bad++;
                        }
                printf("diag: tdpbssd A=tmm%d B=tmm%d: %s\n", a_tmm[ia], b_tmm[ib], bad ? "MISMATCH" : "OK");
                if (bad) rc |= 16;
            }
        }
    }

    // 5) pack_b_tile into the kernel's thread-local buffer + multiply with it
    {
        alignas(64) char rows[16 * 34];
        for (int n = 0; n < 16; ++n) {
            block_q8_0 * b = (block_q8_0 *)(rows + n * 34);
            b->d = (ggml_fp16_t)GGML_FP32_TO_FP16((n + 1) / 16.0f);
            for (int j = 0; j < 32; ++j) b->qs[j] = (int8_t)((n + j) % 64 - 32);
        }
        pack_b_tile<block_q8_0>(g_amx_buf.b_tiles[0], rows, 34, 0, 16);
        printf("diag: packed bytes[0..3]=%d %d %d %d (expect %d %d %d %d), scale[0]=%f expect %f\n",
               g_amx_buf.b_tiles[0][0], g_amx_buf.b_tiles[0][1], g_amx_buf.b_tiles[0][2], g_amx_buf.b_tiles[0][3],
               -32, -31, -30, -29,
               GGML_FP16_TO_FP32(*(ggml_fp16_t *)(g_amx_buf.b_tiles[0] + 512)), 1.0f / 16.0f);

        alignas(64) uint8_t at2[512];
        alignas(64) int32_t ct2[256];
        memset(at2, 0, sizeof(at2));
        for (int r = 0; r < 16; ++r) at2[r * 32 + 0] = (uint8_t)(r + 1);
        AMX_TILE_LOADD(tok, TMM2, at2, 32);
        AMX_TILE_LOADD(tok, TMM0, g_amx_buf.b_tiles[0], 64);
        AMX_TILE_ZERO(tok, TMM4);
        AMX_TILE_DPBSSD(tok, TMM4, TMM2, TMM0);
        AMX_TILE_STORED(tok, TMM4, ct2, 64);
        // C[r][n] = (r+1) * (qs[n][0]) = (r+1) * (n - 32)
        int bad2 = 0;
        for (int r = 0; r < 16 && bad2 < 3; ++r)
            for (int n = 0; n < 16 && bad2 < 3; ++n)
                if (ct2[r * 16 + n] != (r + 1) * (n - 32)) {
                    printf("diag: bufmul C[%d][%d]=%d expect %d\n", r, n, ct2[r * 16 + n], (r + 1) * (n - 32));
                    bad2++;
                }
        printf("diag: bufmul %s\n", bad2 ? "MISMATCH" : "OK");
        if (bad2) rc |= 32;
    }

    // 6) the real kernel on synthetic data (full 16-row A tile, one B tile, KB=1)
    {
        const int64_t M = 16, N = 16, KKB = 1;
        alignas(64) char w[(size_t)N * 34], a[(size_t)M * 34];
        std::vector<float> dst((size_t)M * N, 0.f);
        for (int n = 0; n < N; ++n) {
            block_q8_0 * b = (block_q8_0 *)(w + n * 34);
            b->d = (ggml_fp16_t)GGML_FP32_TO_FP16((n + 1) / 16.0f);
            for (int j = 0; j < 32; ++j) b->qs[j] = (int8_t)((n * 3 + j) % 64 - 32);
        }
        for (int m = 0; m < M; ++m) {
            block_q8_0 * b = (block_q8_0 *)(a + m * 34);
            b->d = (ggml_fp16_t)GGML_FP32_TO_FP16((m + 1) / 8.0f);
            for (int j = 0; j < 32; ++j) b->qs[j] = (int8_t)((m * 5 + j) % 32 - 16);
        }
        amx_gemm_impl<block_q8_0>(M, N, KKB, w, 34, a, 34, dst.data(), N, 0, 1);

        const block_q8_0 * wb = (const block_q8_0 *)w;
        const block_q8_0 * ab = (const block_q8_0 *)a;
        int dot = 0;
        for (int j = 0; j < 32; ++j) dot += (int)ab[0].qs[j] * (int)wb[0].qs[j];
        const float ref00 = GGML_FP16_TO_FP32(ab[0].d) * GGML_FP16_TO_FP32(wb[0].d) * (float)dot;
        int bad3 = 0;
        double maxerr = 0;
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                int d2 = 0;
                for (int j = 0; j < 32; ++j) d2 += (int)ab[m].qs[j] * (int)wb[n].qs[j];
                const float r2 = GGML_FP16_TO_FP32(ab[m].d) * GGML_FP16_TO_FP32(wb[n].d) * (float)d2;
                maxerr = std::max(maxerr, (double)std::fabs(dst[m * N + n] - r2));
                if (std::fabs(dst[m * N + n] - r2) > 1e-3 * std::max(1.f, std::fabs(r2))) bad3++;
            }
        printf("diag: kernel run dst[0]=%f ref=%f maxerr=%f bad=%d -> %s\n",
               dst[0], ref00, maxerr, bad3, bad3 ? "MISMATCH" : "OK");
        if (bad3) rc |= 64;
    }

    // 6b) the real kernel on the exact shapes that fail in the correctness test
    {
        const int64_t shapes[][3] = {{8, 32, 64}, {9, 17, 32}, {17, 33, 128}, {33, 16, 64}, {31, 47, 32}};
        for (int si = 0; si < 5; ++si) {
            const int64_t M = shapes[si][0], N = shapes[si][1], K = shapes[si][2];
            const int64_t KKB = K / 32;
            alignas(64) char w[47 * 34 * 4], a[33 * 34 * 4];
            std::vector<float> dst((size_t)M * N, 0.f);
            for (int64_t n = 0; n < N; ++n) {
                block_q8_0 * b = (block_q8_0 *)(w + n * 34 * KKB);
                for (int kb = 0; kb < KKB; ++kb) {
                    b[kb].d = (ggml_fp16_t)GGML_FP32_TO_FP16((float)((n % 13) + 1) / 16.0f);
                    for (int j = 0; j < 32; ++j) b[kb].qs[j] = (int8_t)((n * 3 + j + kb * 7) % 64 - 32);
                }
            }
            for (int64_t m = 0; m < M; ++m) {
                block_q8_0 * b = (block_q8_0 *)(a + m * 34 * KKB);
                for (int kb = 0; kb < KKB; ++kb) {
                    b[kb].d = (ggml_fp16_t)GGML_FP32_TO_FP16((float)((m % 7) + 1) / 8.0f);
                    for (int j = 0; j < 32; ++j) b[kb].qs[j] = (int8_t)((m * 5 + j + kb * 3) % 32 - 16);
                }
            }
            amx_gemm_impl<block_q8_0>(M, N, KKB, w, (int64_t)34 * KKB, a, (int64_t)34 * KKB, dst.data(), N, 0, 1);

            int bad4 = 0;
            int64_t fm = 0, fn = 0;
            for (int64_t m = 0; m < M && bad4 == 0; ++m)
                for (int64_t n = 0; n < N && bad4 == 0; ++n) {
                    double r2 = 0;
                    for (int kb = 0; kb < KKB; ++kb) {
                        int d2 = 0;
                        for (int j = 0; j < 32; ++j) {
                            d2 += (int)((block_q8_0 *)(a + m * 34 * KKB))[kb].qs[j] *
                                  (int)((block_q8_0 *)(w + n * 34 * KKB))[kb].qs[j];
                        }
                        r2 += (double)GGML_FP16_TO_FP32(((block_q8_0 *)(a + m * 34 * KKB))[kb].d) *
                              (double)GGML_FP16_TO_FP32(((block_q8_0 *)(w + n * 34 * KKB))[kb].d) * (double)d2;
                    }
                    if (std::fabs(dst[m * N + n] - r2) > 1e-3 * std::max(1.0, std::fabs(r2))) {
                        bad4 = 1;
                        fm = m;
                        fn = n;
                    }
                }
            printf("diag: kernel M=%d N=%d K=%d -> %s", (int)M, (int)N, (int)K, bad4 ? "FAIL" : "OK");
            if (bad4) {
                printf(" at [%d,%d] got=%f\n  row got:", (int)fm, (int)fn, dst[fm * N + fn]);
                for (int64_t n = 0; n < N && n < 36; ++n) printf(" %8.3f", dst[0 * N + n]);
                printf("\n  row ref:");
                for (int64_t n = 0; n < N && n < 36; ++n) {
                    double r2 = 0;
                    for (int kb = 0; kb < KKB; ++kb) {
                        int d2 = 0;
                        for (int j = 0; j < 32; ++j) {
                            d2 += (int)((block_q8_0 *)a)[kb].qs[j] * (int)((block_q8_0 *)(w + n * 34 * KKB))[kb].qs[j];
                        }
                        r2 += (double)GGML_FP16_TO_FP32(((block_q8_0 *)a)[kb].d) *
                              (double)GGML_FP16_TO_FP32(((block_q8_0 *)(w + n * 34 * KKB))[kb].d) * (double)d2;
                    }
                    printf(" %8.3f", r2);
                }
                printf("\n");
                rc |= 256;
            } else {
                printf("\n");
            }
        }
    }

    // 7) verbatim replica of the CURRENT kernel sequence (staged A, zero-all,
    //    dpbssd pair, stored pair, A1, dpbssd pair, stored pair) with tmm0/tmm1
    //    snapshots at several points to catch WHERE a tile loses its content
    {
        const int64_t NR = 32;
        alignas(64) char w[(size_t)NR * 34], a[(size_t)NR * 34];
        for (int64_t n = 0; n < NR; ++n) {
            block_q8_0 * b = (block_q8_0 *)(w + n * 34);
            b->d = (ggml_fp16_t)GGML_FP32_TO_FP16((float)(n + 1) / 16.0f);
            for (int j = 0; j < 32; ++j) b->qs[j] = (int8_t)((n * 3 + j) % 64 - 32);
        }
        for (int64_t m = 0; m < NR; ++m) {
            block_q8_0 * b = (block_q8_0 *)(a + m * 34);
            b->d = (ggml_fp16_t)GGML_FP32_TO_FP16((float)(m + 1) / 8.0f);
            for (int j = 0; j < 32; ++j) b->qs[j] = (int8_t)((m * 5 + j) % 32 - 16);
        }
        pack_b_tile<block_q8_0>(g_amx_buf.b_tiles[0], w, 34, 0, 16);
        pack_b_tile<block_q8_0>(g_amx_buf.b_tiles[1], w + 16 * 34, 34, 0, 16);

        // 7a) identity-A probe of the tmm1 path alone: A row r selects k=r,
        //     so C[r][n] must equal qs(w_{16+n}, r) exactly (int check)
        {
            unpack_a_tail(g_amx_buf.a_tiles[0], a, 34, 0, 0);  // zero-fill
            for (int r = 0; r < 16; ++r) g_amx_buf.a_tiles[0][r * 32 + r] = 1;
            AMX_TILE_LOADD(tok, TMM2, g_amx_buf.a_tiles[0], 32);
            AMX_TILE_LOADD(tok, TMM1, g_amx_buf.b_tiles[1], 64);
            AMX_TILE_ZERO(tok, TMM4);
            AMX_TILE_DPBSSD(tok, TMM4, TMM2, TMM1);
            alignas(64) int32_t cid[256];
            AMX_TILE_STORED(tok, TMM4, cid, 64);
            int badid = 0;
            for (int r = 0; r < 16 && badid < 3; ++r)
                for (int n = 0; n < 16 && badid < 3; ++n) {
                    const int32_t expv = (int8_t)(((16 + n) * 3 + r) % 64 - 32);
                    if (cid[r * 16 + n] != expv) {
                        printf("diag: 7a identity tmm1 [%d][%d] got=%d exp=%d\n", r, n, cid[r * 16 + n], expv);
                        badid++;
                    }
                }
            printf("diag: 7a identity-A x tmm1 (packed B tile 1): %s\n", badid ? "MISMATCH" : "OK");
            if (badid) rc |= 4096;
        }

        // 7b) full replica with tmm0/tmm1 snapshots at each stage
        AMX_TILE_LOADD(tok, TMM0, g_amx_buf.b_tiles[0], 64);
        AMX_TILE_LOADD(tok, TMM1, g_amx_buf.b_tiles[1], 64);
        alignas(64) uint8_t snap[6][1024];
        AMX_TILE_STORED(tok, TMM0, snap[0], 64);
        AMX_TILE_STORED(tok, TMM1, snap[1], 64);

        unpack_a_tail(g_amx_buf.a_tiles[0], a, 34, 0, 16);
        AMX_TILE_LOADD(tok, TMM2, g_amx_buf.a_tiles[0], 32);
        AMX_TILE_STORED(tok, TMM1, snap[2], 64);

        AMX_TILE_ZERO(tok, TMM4);
        AMX_TILE_ZERO(tok, TMM5);
        AMX_TILE_ZERO(tok, TMM6);
        AMX_TILE_ZERO(tok, TMM7);
        AMX_TILE_STORED(tok, TMM1, snap[3], 64);

        AMX_TILE_DPBSSD(tok, TMM4, TMM2, TMM0);
        AMX_TILE_STORED(tok, TMM1, snap[4], 64);

        AMX_TILE_DPBSSD(tok, TMM6, TMM2, TMM1);
        AMX_TILE_STORED(tok, TMM1, snap[5], 64);

        AMX_TILE_STORED(tok, TMM4, g_amx_buf.c_tiles[0][0], 64);
        AMX_TILE_STORED(tok, TMM6, g_amx_buf.c_tiles[0][2], 64);

        unpack_a_tail(g_amx_buf.a_tiles[1], a + 16 * 34, 34, 0, 16);
        AMX_TILE_LOADD(tok, TMM3, g_amx_buf.a_tiles[1], 32);
        AMX_TILE_DPBSSD(tok, TMM5, TMM3, TMM0);
        AMX_TILE_DPBSSD(tok, TMM7, TMM3, TMM1);
        AMX_TILE_STORED(tok, TMM5, g_amx_buf.c_tiles[0][1], 64);
        AMX_TILE_STORED(tok, TMM7, g_amx_buf.c_tiles[0][3], 64);

        const bool s0ok = memcmp(snap[0], g_amx_buf.b_tiles[0], 8 * 64) == 0;
        bool s1ok = true;
        for (int i = 1; i < 6; ++i) {
            const bool ok = memcmp(snap[i], g_amx_buf.b_tiles[1], 8 * 64) == 0;
            if (!ok) s1ok = false;
            printf("diag: 7b tmm1 snap%d (%s): %s", i,
                   i == 1 ? "after loadd B0/B1" : i == 2 ? "after loadd A0"
                 : i == 3 ? "after zero tmm4..7" : i == 4 ? "after dpbssd(tmm0)"
                 : "after dpbssd(tmm1)", ok ? "OK" : "FAIL");
            if (!ok) {
                printf("\n      got:");
                for (int j = 0; j < 16; ++j) printf(" %02x", snap[i][j]);
                printf("\n      exp:");
                for (int j = 0; j < 16; ++j) printf(" %02x", ((const uint8_t *)g_amx_buf.b_tiles[1])[j]);
            }
            printf("\n");
        }
        printf("diag: 7b tmm0 snapshot: %s, tmm1: %s\n", s0ok ? "OK" : "FAIL", s1ok ? "OK" : "FAIL");
        if (!s0ok || !s1ok) rc |= 512;

        int bad5 = 0;
        for (int t = 0; t < 4 && bad5 < 2; ++t) {
            // same mapping as acc_4tiles: t bit0 = A half, bit1 = B half
            const int64_t mb = 16 * (t & 1);
            const int64_t nb = 16 * (t >> 1);
            for (int r = 0; r < 16 && bad5 < 2; ++r)
                for (int n = 0; n < 16 && bad5 < 2; ++n) {
                    int d2 = 0;
                    for (int j = 0; j < 32; ++j) {
                        d2 += (int)((block_q8_0 *)(a + (mb + r) * 34))->qs[j] *
                              (int)((block_q8_0 *)(w + (nb + n) * 34))->qs[j];
                    }
                    const int32_t gotv = g_amx_buf.c_tiles[0][t][r * 16 + n];
                    if (gotv != d2) {
                        printf("diag: 7b 4-tile block t=%d [%d][%d] got=%d exp=%d\n", t, r, n, gotv, d2);
                        bad5++;
                    }
                }
        }
        printf("diag: 7b 4-tile dpbssd block (raw ints): %s\n", bad5 ? "MISMATCH" : "OK");
        if (bad5) rc |= 1024;
    }

    // 8) minimal load->store roundtrips isolating row-step/alignment/sequence
    //    factors: (a) solo tmm2 with odd base (+2) and row step 34, (b) same
    //    with aligned base, (c) row step 68, (d) pair tmm2+tmm3, (e)
    //    interleaved load/store, (f) quad with all-aligned bases and 64
    //    (the printed labels below say "stride" - same thing, ISA spelling)
    {
        alignas(64) char a[(size_t)32 * 34 + 64];
        alignas(64) uint8_t pat[2048];
        alignas(64) uint8_t s0[1024], s2[1024], s3[1024];
        for (int i = 0; i < 2048; ++i) pat[i] = (uint8_t)(i * 7 + 1);
        memcpy(a, pat, 32 * 34);
        // note: a+2 pattern == pat+2, a+32 pattern == pat+32
        const uint8_t * e2 = pat + 2;   // expected for base a+2, stride 34, 16 rows x 32 B
        const uint8_t * e32 = pat + 32;
        bool ok;
        int bad6 = 0;

        // (a) solo tmm2 base a+2 row step 34
        memset(s2, 0, sizeof(s2));
        AMX_TILE_LOADD(tok, TMM2, a + 2, 34);
        AMX_TILE_STORED(tok, TMM2, s2, 32);
        ok = memcmp(s2, e2, 16 * 32) == 0;
        printf("diag: 8a solo tmm2 base+2 stride34: %s\n", ok ? "OK" : "FAIL");
        if (!ok) { printf("diag:   got:"); for (int i = 0; i < 32; ++i) printf(" %02x", s2[i]);
                   printf("\n       exp:"); for (int i = 0; i < 32; ++i) printf(" %02x", e2[i]); printf("\n"); bad6 = 1; }

        // (b) solo tmm2 base a+32 (64-aligned) row step 34
        memset(s2, 0, sizeof(s2));
        AMX_TILE_LOADD(tok, TMM2, a + 32, 34);
        AMX_TILE_STORED(tok, TMM2, s2, 32);
        ok = memcmp(s2, e32, 16 * 32) == 0;
        printf("diag: 8b solo tmm2 base+32 stride34: %s\n", ok ? "OK" : "FAIL");
        if (!ok) { printf("diag:   got:"); for (int i = 0; i < 32; ++i) printf(" %02x", s2[i]);
                   printf("\n       exp:"); for (int i = 0; i < 32; ++i) printf(" %02x", e32[i]); printf("\n"); bad6 = 1; }

        // (c) solo tmm2 base a+2 row step 68
        memset(s2, 0, sizeof(s2));
        AMX_TILE_LOADD(tok, TMM2, a + 2, 68);
        AMX_TILE_STORED(tok, TMM2, s2, 32);
        ok = memcmp(s2, e2, 16 * 32) == 0;
        printf("diag: 8c solo tmm2 base+2 stride68: %s\n", ok ? "OK" : "FAIL");
        if (!ok) bad6 = 1;

        // (d) pair tmm2+tmm3 base a+2 row step 34
        memset(s2, 0, sizeof(s2));
        memset(s3, 0, sizeof(s3));
        AMX_TILE_LOADD(tok, TMM2, a + 2, 34);
        AMX_TILE_LOADD(tok, TMM3, a + 2, 34);
        AMX_TILE_STORED(tok, TMM2, s2, 32);
        AMX_TILE_STORED(tok, TMM3, s3, 32);
        ok = memcmp(s2, e2, 16 * 32) == 0 && memcmp(s3, e2, 16 * 32) == 0;
        printf("diag: 8d pair tmm2+tmm3 stride34: %s\n", ok ? "OK" : "FAIL");
        if (!ok) bad6 = 1;

        // (e) interleaved loadd/stored tmm2,tmm3
        memset(s2, 0, sizeof(s2));
        memset(s3, 0, sizeof(s3));
        AMX_TILE_LOADD(tok, TMM2, a + 2, 34);
        AMX_TILE_STORED(tok, TMM2, s2, 32);
        AMX_TILE_LOADD(tok, TMM3, a + 2, 34);
        AMX_TILE_STORED(tok, TMM3, s3, 32);
        ok = memcmp(s2, e2, 16 * 32) == 0 && memcmp(s3, e2, 16 * 32) == 0;
        printf("diag: 8e interleaved tmm2/tmm3 stride34: %s\n", ok ? "OK" : "FAIL");
        if (!ok) bad6 = 1;

        // (f) quad tmm0..tmm3, all bases 64-aligned, stride 64 (8-row shapes)
        //     tmm0/tmm1 are 8x64; use their config for a like-for-like quad
        memset(s0, 0, sizeof(s0));
        memset(s2, 0, sizeof(s2));
        memset(s3, 0, sizeof(s3));
        AMX_TILE_LOADD(tok, TMM0, pat, 64);
        AMX_TILE_LOADD(tok, TMM1, pat + 512, 64);
        AMX_TILE_LOADD(tok, TMM2, pat, 32);
        AMX_TILE_LOADD(tok, TMM3, pat + 512, 32);
        AMX_TILE_STORED(tok, TMM0, s0, 64);
        AMX_TILE_STORED(tok, TMM2, s2, 32);
        AMX_TILE_STORED(tok, TMM3, s3, 32);
        ok = memcmp(s0, pat, 8 * 64) == 0 && memcmp(s2, pat, 16 * 32) == 0 && memcmp(s3, pat + 512, 16 * 32) == 0;
        printf("diag: 8f quad aligned stride64/32: %s\n", ok ? "OK" : "FAIL");
        if (!ok) bad6 = 1;

        printf("diag: step8 %s\n", bad6 ? "MISMATCH" : "OK");
        if (bad6) rc |= 2048;
    }
    return rc;
}
// Does THIS machine actually execute tdpbf16ps? CPUID can lie: our test
// server is a KVM guest whose vCPU advertises the amx_bf16 flag while the
// instruction raises #UD (SIGILL) - the flag is copied from the host, the
// execution path is not (see quirks.md, section 11). tileload/store/zero
// and tdpbssd from the same feature set DO run there, so the only honest
// test is to run the instruction once and survive the answer.
//
// Probe recipe: request XTILEDATA (amx_probe_cpu), load the minimal legal
// bf16 shape set, then zero/load/multiply/store on zero-filled buffers
// under a temporary SIGILL handler. sigsetjmp/siglongjmp jumps out of the
// handler if the CPU says no. The verdict is a hardware/VM property, so a
// process-global cached int is fine; concurrent probes compute the same
// answer.
static sigjmp_buf g_bf16_probe_jmp;
extern "C" void amx_bf16_sigill(int) { siglongjmp(g_bf16_probe_jmp, 1); }

bool amx_bf16_executes() {
    static int cached = -1;
    if (cached >= 0) return cached != 0;
    cached = 0;
    if (!amx_probe_cpu()) return false;

    // minimal legal bf16 shapes: tmm0/tmm1 16x32 (B and A, 16x16 bf16),
    // tmm2 16x64 (C, 16x16 fp32). Shapes validated against the SDM rules
    // for tdpbf16ps (dst colsb = 2 * src2 colsb; src1 colsb = 2 * rows(src2)).
    amx_tilecfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.palette = 1;
    for (int i = 0; i < 2; ++i) { cfg.rows[i] = 16; cfg.colsb[i] = 32; }
    cfg.rows[2] = 16; cfg.colsb[2] = 64;

    alignas(64) char zero_buf[1024];
    memset(zero_buf, 0, sizeof(zero_buf));

    struct sigaction sa, old;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = amx_bf16_sigill;
    sigaction(SIGILL, &sa, &old);
    bool ok = false;
    if (sigsetjmp(g_bf16_probe_jmp, 1) == 0) {
#if defined(__GNUC__) && !defined(__clang__)
        unsigned tok = 0;
        __asm__ __volatile__ ("ldtilecfg (%0)" :: "r" (&cfg) : "memory");
        AMX_TILE_ZERO(tok, TMM2);
        AMX_TILE_LOADD(tok, TMM0, zero_buf, 32);
        AMX_TILE_LOADD(tok, TMM1, zero_buf, 32);
        AMX_TILE_DPBF16PS(tok, TMM2, TMM1, TMM0);
        AMX_TILE_STORED(tok, TMM2, zero_buf, 64);
#else
        _tile_loadconfig(&cfg);
        _tile_zero(TMM2);
        _tile_loadd(TMM0, zero_buf, 32);
        _tile_loadd(TMM1, zero_buf, 32);
        _tile_dpbf16ps(TMM2, TMM1, TMM0);
        _tile_stored(TMM2, zero_buf, 64);
#endif
        ok = true;
    }
    sigaction(SIGILL, &old, nullptr);
    cached = ok ? 1 : 0;
    return ok;
}

// bf16 is the default kernel (see the header: no per-k-block rescale) -
// but ONLY where tdpbf16ps actually executes; on VMs that advertise the
// flag without wiring the instruction we fall back to the int8 kernel
// automatically. GGML_AMX_INT8=1 - or the setter below - forces int8.
int g_amx_int8 = -1;   // -1 = not decided yet

bool amx_use_int8() {
    if (g_amx_int8 < 0) {
        const char * e = getenv("GGML_AMX_INT8");
        g_amx_int8 = (e && e[0] == '1') ? 1 : 0;
    }
    if (g_amx_int8 == 0 && !amx_bf16_executes()) return true;   // silent fallback
    return g_amx_int8 != 0;
}

bool amx_run_gemm(int type, int64_t M, int64_t N, int64_t KB,
                  const void * W, int64_t w_stride,
                  const void * A, int64_t a_stride,
                  float * dst, int64_t ldc, int ith, int nth) {
    if (type == GGML_TYPE_Q4_0) {
        if (amx_use_int8()) {
            amx_gemm_impl<block_q4_0>(M, N, KB, (const char *)W, w_stride, (const char *)A, a_stride, dst, ldc, ith, nth);
        } else if (!amx_gemm_bf16_impl<block_q4_0>(M, N, KB, (const char *)W, w_stride, (const char *)A, a_stride, dst, ldc, ith, nth)) {
            return false;
        }
        return true;
    }
    if (type == GGML_TYPE_Q8_0) {
        if (amx_use_int8()) {
            amx_gemm_impl<block_q8_0>(M, N, KB, (const char *)W, w_stride, (const char *)A, a_stride, dst, ldc, ith, nth);
        } else if (!amx_gemm_bf16_impl<block_q8_0>(M, N, KB, (const char *)W, w_stride, (const char *)A, a_stride, dst, ldc, ith, nth)) {
            return false;
        }
        return true;
    }
    return false;
}

#else  // GGML_AMX_AVAILABLE_COMPILER

bool amx_probe_cpu() { return false; }
bool amx_bf16_executes() { return false; }
bool amx_run_gemm(int, int64_t, int64_t, int64_t, const void *, int64_t, const void *, int64_t, float *, int64_t, int, int) { return false; }
int amx_diag_impl() { return -1; }

#endif  // GGML_AMX_AVAILABLE_COMPILER

}  // namespace

extern "C" bool ggml_amx_available(void) {
    static std::atomic<int> state{0};
    int s = state.load(std::memory_order_relaxed);
    if (s == 0) {
        s = amx_probe_cpu() ? 1 : -1;
        state.store(s, std::memory_order_relaxed);
    }
    return s == 1;
}

extern "C" int ggml_cpu_has_amx_int8(void) {
#if GGML_AMX_AVAILABLE_COMPILER
    return ggml_amx_available() ? 1 : 0;
#else
    return 0;
#endif
}

extern "C" void ggml_amx_set_int8(int enable) {
#if GGML_AMX_AVAILABLE_COMPILER
    g_amx_int8 = enable ? 1 : 0;
#else
    (void)enable;
#endif
}

// 1 when the default bf16 kernel can actually run here (CPU advertises
// AND executes tdpbf16ps); 0 on stub builds, non-AMX CPUs, or VMs that
// only fake the CPUID flag (quirks.md section 11)
extern "C" int ggml_amx_has_bf16(void) {
#if GGML_AMX_AVAILABLE_COMPILER
    return ggml_amx_available() && amx_bf16_executes() ? 1 : 0;
#else
    return 0;
#endif
}

extern "C" bool ggml_amx_gemm(int type, int64_t M, int64_t N, int64_t KB,
                              const void * W, int64_t w_stride,
                              const void * A, int64_t a_stride,
                              float * dst, int64_t ldc, int ith, int nth) {
    if (!ggml_amx_available()) return false;
    if (M < 1 || N < 1 || KB < 1 || ith < 0 || nth < 1 || ith >= nth) return false;
    return amx_run_gemm(type, M, N, KB, W, w_stride, A, a_stride, dst, ldc, ith, nth);
}

extern "C" bool ggml_amx_mul_mat_4d(int typeA,
        long Nx, long Ny, long ne00,
        long ne02, long ne03, long ne12, long ne13,
        long nb02, long nb03, long nbw2, long nbw3, long nb2, long nb3,
        const void * A, long strideA,
        const void * B, long strideB,
        float * C, long stride_C, int ith, int nth) {
    if (!ggml_amx_available()) return false;
    if (typeA != GGML_TYPE_Q4_0 && typeA != GGML_TYPE_Q8_0) return false;
    if (Nx < 1 || Ny < 1 || ne00 <= 0 || (ne00 % TILE_K) != 0) return false;
    if (ne12 < ne02 || ne13 < ne03) return false;

    const int64_t KB = ne00 / TILE_K;
    const int64_t M  = Ny;
    const int64_t N  = Nx;

    for (long i13 = 0; i13 < ne13; ++i13) {
        for (long i12 = 0; i12 < ne12; ++i12) {
            const char * w = (const char *)A + i13 * nb03 + i12 * nb02;
            const char * a = (const char *)B + i13 * nbw3 + i12 * nbw2;
            float * c = (float *)((char *)C + i13 * nb3 + i12 * nb2);
            if (!amx_run_gemm(typeA, M, N, KB, w, strideA, a, strideB, c, stride_C, ith, nth)) {
                return false;
            }
        }
    }
    return true;
}

extern "C" int ggml_amx_diag(void) {
#if GGML_AMX_AVAILABLE_COMPILER
    if (!ggml_amx_available()) {
        printf("diag: AMX not available\n");
        return -1;
    }
    return amx_diag_impl();
#else
    printf("diag: compiled without AMX support\n");
    return -1;
#endif
}
