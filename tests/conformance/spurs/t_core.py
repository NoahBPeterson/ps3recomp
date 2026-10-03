"""SPURS instance API: initialize/finalize, GetInfo, SPU thread queries, lv2
event-queue attachment, calls on workloads that do not exist, and argument
validation (null / misaligned / out of range)."""
from spurs_lib import SDK_VERSION, SYS, P

INFO_IDS = ((0x24, 0x24), (0x48, 0x10))     # thread group id, spu thread ids, handler thread ids


def body(T):
    spurs = T.spurs_init(nspus=2, spu_prio=100, ppu_prio=1000)
    info = T.alloc("info", 280, 8)
    out = T.alloc("out", 64, 8)
    T.call("cellSpursGetInfo", spurs, info)
    T.record_mem("GetInfo (2 spus)", info, 280, ignore=INFO_IDS)

    T.call("cellSpursGetNumSpuThread", spurs, out)
    T.record_mem("GetNumSpuThread out", out, 4)
    ids = T.alloc("ids", 64, 8)
    T.load(3, 8); T.emit(P.li32(4, out), P.stw(3, 4, 0))
    T.call("cellSpursGetSpuThreadId", spurs, ids, out, rc="GetSpuThreadId (room 8)")
    T.record_mem("GetSpuThreadId count", out, 4)
    T.load(3, 1); T.emit(P.li32(4, out), P.stw(3, 4, 0))
    T.call("cellSpursGetSpuThreadId", spurs, ids, out, rc="GetSpuThreadId (room 1)")
    T.record_mem("GetSpuThreadId count (room 1)", out, 4)
    T.call("cellSpursGetSpuThreadGroupId", spurs, out)

    # lv2 event queue attachment
    qattr = T.alloc("qattr", 16, 16, bytes.fromhex("0000000100000001") + b"spursq\0\0")
    qid = T.alloc("qid", 16)
    T.syscall(SYS["equeue_create"], qid, qattr, 0, 32, rc="equeue_create")
    port = T.alloc("port", 16)
    T.call("cellSpursAttachLv2EventQueue", spurs, ("mem", qid), port, 1, rc="AttachLv2EventQueue dynamic")
    T.record_mem("dynamic port", port, 4)
    T.call("cellSpursAttachLv2EventQueue", spurs, ("mem", qid), port + 8, 1,
           rc="AttachLv2EventQueue same queue again")
    T.call("cellSpursDetachLv2EventQueue", spurs, ("u8", port), rc="DetachLv2EventQueue dynamic port")
    T.call("cellSpursDetachLv2EventQueue", spurs, 63, rc="DetachLv2EventQueue unattached port 63")
    T.emit(P.li32(3, 0x05000000), P.li32(4, port), P.stw(3, 4, 0))        # static port 5
    T.call("cellSpursAttachLv2EventQueue", spurs, ("mem", qid), port, 0, rc="AttachLv2EventQueue static 5")
    T.record_mem("static port", port, 4)
    T.call("cellSpursDetachLv2EventQueue", spurs, 5, rc="DetachLv2EventQueue 5")
    T.call("cellSpursAttachLv2EventQueue", spurs, 0x7FFFFFF0, port, 1, rc="AttachLv2EventQueue bad queue id")
    T.call("cellSpursAttachLv2EventQueue", spurs, ("mem", qid), 0, 1, rc="AttachLv2EventQueue null port")

    # calls on a workload id that was never added
    prio = T.alloc("prio", 16, 16, bytes([1] * 8))
    winfo = T.alloc("winfo", 48, 8)
    T.call("cellSpursReadyCountStore", spurs, 0, 1, rc="ReadyCountStore absent wid 0")
    T.call("cellSpursReadyCountAdd", spurs, 0, out, 1, rc="ReadyCountAdd absent wid 0")
    T.call("cellSpursReadyCountSwap", spurs, 0, out, 1, rc="ReadyCountSwap absent wid 0")
    T.call("cellSpursReadyCountCompareAndSwap", spurs, 0, out, 0, 1, rc="ReadyCountCompareAndSwap absent wid 0")
    T.call("cellSpursSendWorkloadSignal", spurs, 0, rc="SendWorkloadSignal absent wid 0")
    T.call("cellSpursGetWorkloadInfo", spurs, 0, winfo, rc="GetWorkloadInfo absent wid 0")
    T.call("cellSpursGetWorkloadData", spurs, out, 0, rc="GetWorkloadData absent wid 0")
    T.call("cellSpursSetMaxContention", spurs, 0, 1, rc="SetMaxContention absent wid 0")
    T.call("cellSpursSetPriorities", spurs, 0, prio, rc="SetPriorities absent wid 0")
    T.call("cellSpursRequestIdleSpu", spurs, 0, 1, rc="RequestIdleSpu absent wid 0")
    T.call("cellSpursShutdownWorkload", spurs, 0, rc="ShutdownWorkload absent wid 0")
    T.call("cellSpursWaitForWorkloadShutdown", spurs, 0, rc="WaitForWorkloadShutdown absent wid 0")
    T.call("cellSpursRemoveWorkload", spurs, 0, rc="RemoveWorkload absent wid 0")
    T.call("_cellSpursWorkloadFlagReceiver", spurs, 0, 1, rc="WorkloadFlagReceiver absent wid 0")
    # out-of-range workload ids
    T.call("cellSpursReadyCountStore", spurs, 32, 1, rc="ReadyCountStore wid 32")
    T.call("cellSpursReadyCountStore", spurs, 16, 1, rc="ReadyCountStore wid 16")
    T.call("cellSpursGetWorkloadInfo", spurs, 99, winfo, rc="GetWorkloadInfo wid 99")
    T.call("cellSpursShutdownWorkload", spurs, 99, rc="ShutdownWorkload wid 99")
    # flag and wake-up
    T.call("cellSpursGetWorkloadFlag", spurs, out, rc="GetWorkloadFlag")
    T.record_mem("GetWorkloadFlag pointer (relative to instance below)", out, 4)
    T.call("cellSpursWakeUp", spurs, rc="WakeUp")

    # argument validation
    T.call("cellSpursGetInfo", spurs, 0, rc="GetInfo null info")
    T.call("cellSpursGetInfo", 0, info, rc="GetInfo null spurs")
    T.call("cellSpursGetInfo", spurs + 8, info, rc="GetInfo misaligned spurs")
    T.call("cellSpursGetNumSpuThread", spurs, 0, rc="GetNumSpuThread null")
    T.call("cellSpursGetSpuThreadGroupId", 0, out, rc="GetSpuThreadGroupId null spurs")
    T.call("cellSpursReadyCountStore", 0, 0, 1, rc="ReadyCountStore null spurs")
    T.call("cellSpursReadyCountStore", spurs + 8, 0, 1, rc="ReadyCountStore misaligned spurs")
    T.call("cellSpursWakeUp", 0, rc="WakeUp null")

    T.call("cellSpursFinalize", spurs, rc="Finalize")
    T.call("cellSpursFinalize", spurs, rc="Finalize again")
    T.call("cellSpursGetInfo", spurs, info, rc="GetInfo after finalize")

    # second instance: other attributes and a name prefix
    spurs2 = T.alloc("spurs2", 4096, 128)
    attr2 = T.alloc("attr2", 512, 8)
    name = T.alloc("name", 16, 16, b"ConfTest\0")
    T.call("_cellSpursAttributeInitialize", attr2, 2, SDK_VERSION, 1, 200, 2000, 1, rc="attr2 init")
    T.call("cellSpursAttributeSetNamePrefix", attr2, name, 8, rc="SetNamePrefix")
    T.record_mem("attribute after SetNamePrefix", attr2, 512)
    T.call("cellSpursInitializeWithAttribute", spurs2, attr2, rc="Initialize 1 spu exitIfNoWork")
    T.call("cellSpursGetInfo", spurs2, info, rc="GetInfo (1 spu)")
    T.record_mem("GetInfo (1 spu, prefix)", info, 280, ignore=INFO_IDS)
    T.call("cellSpursFinalize", spurs2, rc="Finalize instance 2")

    # attribute / initialize validation
    T.call("_cellSpursAttributeInitialize", attr2 + 4, 2, SDK_VERSION, 1, 200, 2000, 0, rc="AttributeInitialize misaligned")
    T.call("_cellSpursAttributeInitialize", 0, 2, SDK_VERSION, 1, 200, 2000, 0, rc="AttributeInitialize null")
    for n in (0, 7):
        T.call("_cellSpursAttributeInitialize", attr2, 2, SDK_VERSION, n, 200, 2000, 0, rc=False)
        T.call("cellSpursInitializeWithAttribute", spurs2, attr2, rc="Initialize nSpus=%d" % n)
    T.call("_cellSpursAttributeInitialize", attr2, 2, SDK_VERSION, 1, 300, 2000, 0, rc=False)
    T.call("cellSpursInitializeWithAttribute", spurs2, attr2, rc="Initialize spuPriority=300")
    T.call("_cellSpursAttributeInitialize", attr2, 2, SDK_VERSION, 1, 200, 2000, 0, rc=False)
    T.call("cellSpursInitializeWithAttribute", spurs2 + 64, attr2, rc="Initialize misaligned spurs")
    T.call("cellSpursInitializeWithAttribute", 0, attr2, rc="Initialize null spurs")
    T.call("cellSpursInitializeWithAttribute", spurs2, 0, rc="Initialize null attr")
