"""Bounded index sizing with diagnostic strings, never training corpus data."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
from training.production_dedup import DedupIndex, POLICY
from training.production_corpus import atomic_json, identity

def peak_rss():
    if sys.platform=='win32':
        from ctypes import wintypes
        class Memory(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p
        psapi=ctypes.WinDLL('psapi',use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Memory),wintypes.DWORD]
        m=Memory();m.cb=ctypes.sizeof(m)
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(m),m.cb):return m.PeakWorkingSetSize
    return None

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--documents',type=int,default=1000)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    db=sqlite3.connect(args.output/'index.sqlite');db.execute('PRAGMA cache_size=-16384');db.execute('PRAGMA temp_store=FILE');index=DedupIndex(db)
    snapshots=[];start=time.perf_counter();total_bytes=0
    for i in range(args.documents):
        text=' '.join('d'+str(i)+'word'+str(j) for j in range(256));total_bytes+=len(text.encode())
        with db:index.add(f'item-{i:08d}',text,'prose','','train',{'diagnostic':True,'item':i})
        if i+1 in (100,500,1000,2000,args.documents):
            snapshots.append(dict(documents=i+1,bytes=total_bytes,wall_seconds=time.perf_counter()-start,
                peak_process_rss=peak_rss(),database_bytes=(args.output/'index.sqlite').stat().st_size,
                anchors=db.execute('SELECT count(*) FROM anchors').fetchone()[0]))
    # A duplicate-heavy suffix probes whether exact copies amplify the index.
    repeated=' '.join('d0word'+str(j) for j in range(256));duplicate_start=time.perf_counter()
    for i in range(200):
        with db:index.add(f'copy-{i:08d}',repeated,'prose','','train',{'diagnostic':True})
    near_start=time.perf_counter()
    for i in range(200):
        near=repeated.replace('d0word128','edited'+str(i))
        with db:index.add(f'near-{i:08d}',near,'prose','','train',{'diagnostic':True})
    near_seconds=time.perf_counter()-near_start
    index.verify();db.commit()
    outcomes=[(r[0],index.assignment(r[0])) for r in db.execute('SELECT item FROM dd ORDER BY item')]
    result=dict(policy=POLICY,diagnostic_only=True,training_tokens_created=0,snapshots=snapshots,
        exact_copy_count=200,copy_phase_including_near_and_verification_seconds=time.perf_counter()-duplicate_start,near_copy_count=200,near_copy_seconds=near_seconds,
        verified_edges=db.execute('SELECT count(*) FROM edges').fetchone()[0],
        final_anchor_count=db.execute('SELECT count(*) FROM anchors').fetchone()[0],
        final_database_bytes=(args.output/'index.sqlite').stat().st_size,outcome_identity=identity(outcomes),
        limitations='Unique diagnostic lexical strings plus exact copies; not representative corpus throughput or a 2B runtime guarantee.')
    db.close();atomic_json(args.output/'result.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
