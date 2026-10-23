"""Offline helpers extracted from the historical transfer probe; no acquisition."""
from pathlib import Path
import hashlib,json,re
R=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads((R/p).read_text(encoding="utf-8"))

def shingles(t):
 w=re.findall(r'\w+',t.casefold());return {tuple(w[i:i+5]) for i in range(len(w)-4)}
def longlines(t):return {s.strip() for s in t.splitlines() if len(s.strip())>=40}
def overlap(rows,refs):
 out=[];refs=[(r,shingles(r['text']),longlines(r['text'])) for r in refs]
 for r in rows:
  a=shingles(r['text']);lines=longlines(r['text']);flags=[];shared=set();exact=[]
  for ref,b,bl in refs:
   n=len(a&b);contain=n/max(1,min(len(a),len(b)));jac=n/max(1,len(a|b));shared|=lines&bl
   if contain>=.2 or jac>=.5:flags.append(dict(reference=ref.get('document_id',ref.get('id')),containment=contain,jaccard=jac))
   if ' '.join(r['text'].casefold().split())==' '.join(ref['text'].casefold().split()):exact.append(ref.get('document_id',ref.get('id')))
  out.append(dict(id=r['id'],exact=exact,flagged_pairs=flags,matching_lines=sorted(shared)))
 return out

def verify_payload():
 m=read("manifest.json");payload=(R/"documents.jsonl").read_bytes();assert sha(payload)==m["documents_sha256"]
 return payload
