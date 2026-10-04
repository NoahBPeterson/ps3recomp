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

## Terminating an SPU thread group does not stop its SPU code

**What.** `sys_spu_thread_group_terminate` ends the run (cause TERMINATED, the group back
to INITIALIZED, a joiner woken), but an SPU thread still executing on its host thread is
not stopped; it runs until it stops by itself. lv2 stops every thread of the group.
`sys_spu_thread_group_exit` from one SPU thread likewise ends the run without stopping
the group's other threads.

**What it would look like.** After a terminate or group exit, a still-running SPU thread
keeps issuing DMA and channel operations: memory changes after the PPU believes the group
stopped, or events arrive on a connected queue. A restart of the group while the old host
thread runs puts two copies of the thread on the same local store.

**Fixing it.** A stop request the SPU interpreter and the lifted SPU code check (at
channel operations and branches), so the host thread unwinds out of the SPU program.

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

## lv2 object ids are not in lv2's format

**What.** lv2 gives every kernel object an id whose top byte is the object type and whose
low bits count up in steps (event queues 0x8D00xxxx, event ports 0x0E00xxxx, mutexes
0x85..., lwcond 0x97..., and so on; RPCS3 reproduces this). Most of ours are table slots
plus one (`sys_event.c`: an event queue is `slot + 1`); PRX modules (0x23...) and SPU
images (0x22...) already use the typed form.

**What it would look like.** A title or library that looks inside an id: tests its type
byte, uses it as an index, or compares it with a constant. Nothing seen does, but libsre
stores the ids in its structures (an LFQueue after attach, an event flag), so a dump of
them differs from the console; the LFQueue and event-flag suites mask those words.

**Fixing it.** One id allocator shared by the lv2 object tables that hands out
`type_base | index << step` and maps an id back to its slot, with the type checked on
lookup (a wrong-type id is ESRCH, as on lv2).
