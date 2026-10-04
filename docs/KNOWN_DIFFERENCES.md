# Known Differences from the Console

Places where ps3recomp knowingly behaves differently from a PS3 (and from RPCS3, the
oracle the conformance suites compare against). Each entry says what differs, what it
would look like if a title ran into it, and what fixing it involves. Add an entry when a
difference is found and left in; remove it when it is fixed.

---

## libgcm_sys and libfs run as HLE

**What.** RPCS3 runs Sony's own `libgcm_sys.sprx` and `libfs.sprx` (LLE) by default. In
ps3recomp they are answered by the HLE libraries (`libs/video/cellGcmSys.c`, `ppu_fs.cpp`)
unless lifted into the build with `tools/lift_firmware_module.py`. libsysmodule's default
module list loads both at process start; lv2's PRX loader (`runtime/syscalls/lv2_prx.c`)
returns an id with nothing behind it for a module that is not in the build.

**What it would look like.** A difference between our cellGcm*/cellFs* and Sony's: a
return code, a struct field, an ordering of RSX commands or file operations. The SPURS
suites don't call either library, so none of them would show it.

**Fixing it.** Lift both and run them on our lv2 as liblv2/libsre already do. libgcm_sys
then needs lv2's RSX syscalls (sys_rsx_*, 666-677) to be real, and libfs needs the
sys_fs_* syscalls (801-). Both are a larger surface than the PRX loader was.

## A lifted module has one fixed load address

**What.** A module is relocated and lifted for one base address
(`lift_firmware_module.py --base`). lv2 places each load of a module wherever it has
room, so a process can load a module, unload it and load it again elsewhere, or (rarely)
hold two copies. `_sys_prx_load_module` here refuses a second load while the first is
still loaded (CELL_PRX_ERROR_ERROR). Unloading then loading again works, at the same
address.

**What it would look like.** A title that loads the same PRX twice gets an error where
the console succeeds. Anything that compares a module's address against an expected one
sees ours.

**Fixing it.** Lift position-independently: keep the relocation table and apply it at
load time to a base lv2 picks, with the lifted code reading addresses through a per-
instance base instead of baking them in. Not worth doing until a title needs it.

## Joining an SPU thread group that is not running returns at once

**What.** On lv2, `sys_spu_thread_group_join` on an initialized group whose last run has
already been joined (or that was never started) blocks until some other thread starts
the group and that run ends. Here it returns immediately with the group's previous cause
and status (`runtime/syscalls/lv2_register.c`, group_join handler, marked KNOWN
DIVERGENCE).

**What it would look like.** Any design where the thread that waits for a group is not
the thread that starts it, and the waiter can get there first:

- A dedicated join thread: `for (;;) { join(g); consume(results); }` while the main
  thread calls `start(g)` once per frame. On the console the join parks until the next
  run finishes. Here every join after the first returns at once with the last run's
  cause, so the loop spins, consuming the same results again and again, or reading output
  buffers while the SPUs are still writing them.
- A waiter created before the start: thread A initializes the group, creates thread B to
  wait on it, then starts it. If B is scheduled first, the console blocks B until the run
  completes; here B returns ALL_THREADS_EXIT/0 before the SPUs ran a single instruction,
  and whatever B does next (reading results, destroying the group) happens too early.

libsre does not do either: its handler thread starts and joins the kernel group itself.

**Fixing it.** Give the group a "run finished, not yet joined" flag and a wait queue:
join consumes the flag if it is set, otherwise sleeps until a run ends. Start clears it.

## Raw SPU count is not part of the SPU limits

**What.** RPCS3's group-create limit check counts raw SPUs in use (`g_raw_spu_ctr`)
against `max_raw`; ours counts only thread groups.

**What it would look like.** A title that creates raw SPUs and then more thread groups
than the remaining SPUs allow gets a group where the console returns EBUSY.

**Fixing it.** Count live raw SPUs from `runtime/spu/spu_raw.c` in `spu_limits_busy`.

## Memory containers are not charged

**What.** SPU thread groups of type MEMORY_FROM_CONTAINER, and PPU thread stacks, take
their memory from a container on lv2 and fail with ENOMEM when it is exhausted. Here they
are not charged to any container.

**What it would look like.** Only a title that relies on running out (or on the exact
amount left) behaves differently.

**Fixing it.** Model lv2 memory containers (sys_memory_container_*) and charge them.
