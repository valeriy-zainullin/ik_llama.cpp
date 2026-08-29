# Working agreements for this fork (ik_llama.cpp, branch amx-gemm)

## AMX quirks log (MANDATORY)

Every newly discovered hardware/toolchain quirk related to Intel AMX
(instruction-set assumptions, GCC intrinsics behavior, tile state, strides,
CPU errata-like behavior) MUST be recorded in `quirks.md` in the repo root —
even if the conversation context is later summarized or compacted. When in
doubt whether something is worth recording: record it.

Also keep code comments in `ggml/src/ggml-amx.cpp` explaining WHY (the hidden
AMX/ISA/Intel-CPU assumptions), not just WHAT; quirks.md and the code
comments should stay consistent with each other.

Already documented (do not re-derive, read quirks.md first):
1. GCC tile intrinsics are unmodeled asm -> own `AMX_TILE_*` wrappers with a
   `"+r"(tok)` dependency chain and "memory" clobbers.
2. tileloadd readback anomaly: with a stride not a multiple of 32 the
   stored->memcmp check FAILS, but dpbssd consumes the same content
   correctly -> A is staged through a buffer (stride 32); do not treat
   readback FAIL as content corruption.
3. C-tile index layout must be IDENTICAL in the stored block and in
   acc_4tiles (t bit0 = row half, bit1 = column half); a mismatch produces
   exact zeros, indistinguishable from a "lost tile" (this was the real
   correctness bug; tiles never lose content - 7b snapshots).
4. Tile indices must be literal tokens (`#define TMM0..7`), two-level macros.
5. TILECFG layout/palette, XCR0 bits 17/18, `ARCH_REQ_XCOMP_PERM(18)`.
6. ik_llama.cpp q4_0 nibbles are split (0..15 low, 16..31 high), unlike
   upstream llama.cpp.
7. Debug AMX results by comparing RAW int32 tile products with an EXPLICIT
   tile layout first (diag step 7b / identity-A probe 7a); only then blame
   hardware. Numerical cross-check caught the real bug that three hardware
   hypotheses missed.

## Build / test habits

- Local machine has NO AVX512/AMX: only compile + objdump locally
  (`/tmp/opencode/build-amx`); runtime verification happens on the user's
  AMX server via `./scripts/benchmark-amx.sh` and
  `./build-amx/bin/test-amx-gemm diag`.
- Always check the exit code of cmake builds (`echo "exit=$?"`): grepping
  the log for "error" misses Russian "Ошибка" messages.
- Workflow: single amend commit on `amx-gemm`, push with
  `--force-with-lease`; user pulls with `git reset --hard origin/amx-gemm`.
- Local GCC 16 codegen does not guarantee server GCC 14 behavior; when in
  doubt ask for `objdump -d build-amx/bin/test-amx-gemm` output from the
  server.
