# Multicore conformance suite

Differential test of SPU thread groups and PPU<->SPU interaction: RPCS3
(headless, same oracle config as `../ppu`) against ps3recomp, on the same ELF.

```
python3 tests/conformance/mc/run_mc_conform.py --work /tmp/mcconf
python3 tests/conformance/mc/run_mc_conform.py --work /tmp/mcconf --skip-oracle   # reuse oracle.txt
```

`gen_mc_conform.py` writes a bare lv2 ELF (raw syscalls, no imports) whose PPU
side creates thread groups from three SPU images assembled by `spu_asm.py`:

- **dma**: GET / PUT of 20 sizes and alignments, plus a GETL list; data dumped.
- **atomic**: 6 SPUs and the PPU thread each add 1 to shared words 400 times
  through GETLLAR/PUTLLC and lwarx/stwcx. on one 128-byte line; final counts
  dumped. Each SPU exits with `0xA000 | index` (sys_spu_thread_exit: status
  in SPU_WrOutMbox, then `stop 0x102`).
- **mbox**: the PPU writes four inbound-mailbox words and both signal
  notification registers; the SPU sums them and exits with the sum.

The transcript also carries every join cause/status and per-thread exit status.
The runner requires the MCCONF BEGIN..END blocks to be identical.

ps3recomp runs the SPU images on its interpreter as resident, concurrent
threads with blocking channel reads (`RD_SPU_INTERP=1 RD_SPU_INTERP_ASYNC=1
SPU_CH_BLOCK=1`). Without those, an SPU image with no lifted registration does
not run at all and its group "completes" with status 0.

SPU thread arguments follow LV2/RPCS3: each u64 in the register's preferred
doubleword (`v128::from64(0, arg)`), so the programs `rotqbyi 4` to reach the
low word.
