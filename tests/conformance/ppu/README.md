# PPU conformance suite

Differential test of every PPU instruction: RPCS3's PPU interpreter (the oracle)
against ps3recomp (ppu_loader → ppu_lifter → runtime), on the same bytes.

```
python3 tests/conformance/ppu/run_ppu_conform.py --work /tmp/ppuconf            # everything
python3 tests/conformance/ppu/run_ppu_conform.py --work /tmp/ppuconf --only ADD,ADDO,BC
python3 tests/conformance/ppu/run_ppu_conform.py --work /tmp/ppuconf --skip-oracle   # reuse oracle.txt
```

Needs a local RPCS3 build (`--rpcs3 PATH`; default `../rpcs3/build/bin/...`).
The oracle runs headless with its own config (`oracle_config.yml`, written into
the work dir): PPU interpreter, saturation bit, accurate non-Java mode, accurate
vector NaNs, FPCC bits, accurate DFMA. Your normal RPCS3 config is not touched.

## How it works

`gen_ppu_conform.py` writes a bare lv2 ELF (no imports, raw syscalls). For each
case it loads a random full state — GPRs, FPRs, VRs, CR, XER, LR, CTR, FPSCR,
VSCR, VRSAVE — runs the instruction(s) under test, saves the state over the same
slot and prints it as one hex line via `sys_tty_write` (plus the 1 KB scratch
buffer for memory ops). `ppu_ops.py` encodes every op RPCS3's decoder knows
(`rpcs3/Emu/Cell/PPUOpcodes.h`) from its fields, so every operand field is
randomized; `ppu_isa.py` draws the values (specials: 0, ±1, INT_MIN/MAX, NaNs,
SNaNs, denormals, ±inf, rounding boundaries, ...). `compare.py` diffs the two
transcripts field by field and prints the failing fields per op.

r1 (stack) and r31 (the slot pointer) are never operands. Branch tests use the
shapes real code uses (conditional returns, tail calls through CTR, calls
through LR/CTR, the `bl $+4` get-PC idiom): a static recompiler models `blr` as
a host return, so branching through LR/CTR to an arbitrary mid-function address
is not expressible and is not tested.

## Known oracle limitations (masked or normalized in compare.py)

- **FPSCR**: RPCS3 does not model it (mtfsf drops RN and the sticky bits; it
  keeps FPCC). Ignored by default (`--ignore r1,fpscr`); FP status needs an
  independent IEEE reference.
- **VSCR**: RPCS3 reads/writes it in big-endian word 0 of the vector; the ISA
  says word 3 (VRB bits 96:127). The harness feeds both words and compares
  RPCS3's word 0 with ps3recomp's word 3.
- Undefined results (lve*x non-addressed lanes, mftb) are masked per case.
