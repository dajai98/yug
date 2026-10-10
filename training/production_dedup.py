"""Deterministic disk-backed near dedup; no embeddings or corpus-sized RAM set.

128 bottom-hash anchors propose candidates; FULL 128-bit shingle sets verify
Jaccard >= .5 or containment >= .8. This is a lexical heuristic, not semantic
clone detection or a guarantee of recall. Candidate overflow fails closed.
Cluster labels are the smallest item identity in each connected component;
labels settle after ingestion and resolve through unions before accounting.
"""
import hashlib
import heapq
import json
import re
import sqlite3
from training.production_corpus import canonical, identity
from training.production_spec import need

POLICY = dict(version='lexical-bottom128-v1', shingles=5, anchors=128,
              jaccard=.5, containment=.8, candidate_limit=10000,
              max_document_bytes=1_048_576)
_LEX = re.compile(r'(?P<string>"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')|(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)|(?P<word>\w+)|(?P<punct>[^\w\s])')

def features(text, content_type, language=''):
    need(len(text.encode('utf-8')) <= POLICY['max_document_bytes'], 'Dedup document byte limit exceeded')
    if content_type=='code':
        if language=='python':
            import io, tokenize, token
            try:
                ignored={tokenize.COMMENT,tokenize.NL,token.NEWLINE,token.INDENT,token.DEDENT,token.ENDMARKER,token.ENCODING}
                words=[t.string for t in tokenize.generate_tokens(io.StringIO(text).readline) if t.type not in ignored]
            except (tokenize.TokenError,IndentationError,SyntaxError):
                words=[m.group() for m in _LEX.finditer(text) if m.lastgroup!='comment']
        else:
            words=[m.group() for m in _LEX.finditer(text) if m.lastgroup!='comment']
    else:
        words=re.findall(r'\w+',text.casefold())
    if not words:
        return []
    n=min(POLICY['shingles'],len(words))
    # Hash lexical tokens with unambiguous framing; actual training text untouched.
    return sorted({hashlib.sha256(canonical(words[i:i+n])).digest()[:16] for i in range(len(words)-n+1)})

def unpack(blob):
    need(len(blob)%16==0,'Corrupt shingle blob')
    return [blob[i:i+16] for i in range(0,len(blob),16)]

def similarity(a,b):
    i=j=intersection=0
    while i<len(a) and j<len(b):
        if a[i]==b[j]: intersection+=1;i+=1;j+=1
        elif a[i]<b[j]: i+=1
        else: j+=1
    union=len(a)+len(b)-intersection
    return intersection, intersection/union if union else 0., intersection/min(len(a),len(b)) if a and b else 0.

class DedupIndex:
    """Transaction-compatible shared SQLite index. Caller commits item stages."""
    def __init__(self, db):
        self.db=db
        db.executescript('''
        CREATE TABLE IF NOT EXISTS dedup_policy (id INTEGER PRIMARY KEY, identity TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS dd (
          id INTEGER PRIMARY KEY, item TEXT UNIQUE NOT NULL, exact TEXT NOT NULL,
          shingles BLOB NOT NULL, checksum TEXT NOT NULL, split TEXT NOT NULL,
          provenance TEXT NOT NULL, parent INTEGER, size INTEGER NOT NULL,
          label TEXT NOT NULL, winner TEXT NOT NULL, winner_split TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS dd_exact ON dd(exact);
        CREATE TABLE IF NOT EXISTS anchors (anchor BLOB NOT NULL, doc INTEGER NOT NULL,
          PRIMARY KEY(anchor,doc)) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS anchors_doc ON anchors(doc,anchor);
        CREATE TABLE IF NOT EXISTS edges (left_item TEXT, right_item TEXT, reason TEXT,
          intersection_count INTEGER, jaccard REAL, containment REAL,
          PRIMARY KEY(left_item,right_item));
        ''')
        found=db.execute('SELECT identity FROM dedup_policy WHERE id=1').fetchone()
        if found: need(found[0]==identity(POLICY),'Dedup policy changed on resume')
        else: db.execute('INSERT INTO dedup_policy VALUES (1,?)',(identity(POLICY),));db.commit()

    def root(self, node):
        trail=[]
        while True:
            row=self.db.execute('SELECT parent FROM dd WHERE id=?',(node,)).fetchone()
            need(row is not None,'Missing dedup parent')
            if row[0]==node: break
            need(len(trail)<128,'Corrupt dedup parent cycle/depth')
            trail.append(node);node=row[0]
        for old in trail: self.db.execute('UPDATE dd SET parent=? WHERE id=?',(node,old))
        return node

    def union(self,a,b):
        a,b=self.root(a),self.root(b)
        if a==b:return
        x=self.db.execute('SELECT size,label,winner,winner_split FROM dd WHERE id=?',(a,)).fetchone()
        y=self.db.execute('SELECT size,label,winner,winner_split FROM dd WHERE id=?',(b,)).fetchone()
        if (x[0],-a)<(y[0],-b): a,b,x,y=b,a,y,x
        # Validation wins a cross-split duplicate: never migrate held-out text to train.
        winner=min(((x[2],x[3]),(y[2],y[3])),key=lambda w:(w[1]!='validation',w[0]))
        self.db.execute('UPDATE dd SET parent=? WHERE id=?',(a,b))
        self.db.execute('UPDATE dd SET size=?,label=?,winner=?,winner_split=? WHERE id=?',
                        (x[0]+y[0],min(x[1],y[1]),winner[0],winner[1],a))

    def add(self,item,text,content_type,language,split,provenance):
        need(split in ('train','validation'),'Invalid dedup split')
        exact='sha256:'+hashlib.sha256(text.encode('utf-8')).hexdigest()
        encoded=canonical(provenance).decode()
        previous=self.db.execute('SELECT id,exact,split,provenance FROM dd WHERE item=?',(item,)).fetchone()
        if previous:
            need(previous[1:]==(exact,split,encoded),'Dedup item changed on resume')
            return previous[0]
        exact_previous=self.db.execute('SELECT id,item FROM dd WHERE exact=? ORDER BY id LIMIT 1',(exact,)).fetchone()
        if exact_previous:
            # Exact duplicates do not multiply anchors/candidate work or repeat
            # shingle blobs. Their provenance and exact edge remain durable.
            blob=b''
            cur=self.db.execute('INSERT INTO dd(item,exact,shingles,checksum,split,provenance,size,label,winner,winner_split) VALUES (?,?,?,?,?,?,1,?,?,?)',
                (item,exact,blob,hashlib.sha256(blob).hexdigest(),split,encoded,item,item,split))
            node=cur.lastrowid;self.db.execute('UPDATE dd SET parent=? WHERE id=?',(node,node))
            left,right=sorted((item,exact_previous[1]))
            self.db.execute('INSERT INTO edges VALUES (?,?,?,?,?,?)',(left,right,'exact',None,1.,1.))
            self.union(node,exact_previous[0]);return node
        hashes=features(text,content_type,language);blob=b''.join(hashes)
        # Disk temporary table avoids a corpus-sized candidate Python set.
        self.db.execute('CREATE TEMP TABLE IF NOT EXISTS candidates(id INTEGER PRIMARY KEY)')
        self.db.execute('DELETE FROM candidates')
        self.db.execute('INSERT OR IGNORE INTO candidates SELECT id FROM dd WHERE exact=?',(exact,))
        for anchor in heapq.nsmallest(POLICY['anchors'],hashes):
            self.db.execute('INSERT OR IGNORE INTO candidates SELECT doc FROM anchors WHERE anchor=? LIMIT ?',(anchor,POLICY['candidate_limit']+1))
            need(self.db.execute('SELECT count(*) FROM candidates').fetchone()[0]<=POLICY['candidate_limit'],'Near-dedup candidate overflow; requires review, never silently accepted')
        count=self.db.execute('SELECT count(*) FROM candidates').fetchone()[0]
        need(count<=POLICY['candidate_limit'],'Near-dedup candidate overflow; requires review, never silently accepted')
        cur=self.db.execute('INSERT INTO dd(item,exact,shingles,checksum,split,provenance,size,label,winner,winner_split) VALUES (?,?,?,?,?,?,1,?,?,?)',
            (item,exact,blob,hashlib.sha256(blob).hexdigest(),split,encoded,item,item,split))
        node=cur.lastrowid
        self.db.execute('UPDATE dd SET parent=? WHERE id=?',(node,node))
        for old,olditem,oldexact,other,checksum in self.db.execute('SELECT d.id,d.item,d.exact,d.shingles,d.checksum FROM dd d JOIN candidates c ON d.id=c.id ORDER BY d.item'):
            if self.root(node)==self.root(old):
                continue  # Existing component already proves connectivity; avoid quadratic redundant edges.
            need(hashlib.sha256(other).hexdigest()==checksum,'Corrupt dedup features')
            inter,jaccard,contain=similarity(hashes,unpack(other))
            reason='exact' if oldexact==exact else 'lexical_shingles'
            if oldexact==exact or jaccard>=POLICY['jaccard'] or contain>=POLICY['containment']:
                left,right=sorted((item,olditem))
                self.db.execute('INSERT INTO edges VALUES (?,?,?,?,?,?)',(left,right,reason,inter,jaccard,contain))
                self.union(node,old)
        self.db.executemany('INSERT INTO anchors VALUES (?,?)',((h,node) for h in heapq.nsmallest(POLICY['anchors'],hashes)))
        return node

    def assignment(self,item):
        row=self.db.execute('SELECT id FROM dd WHERE item=?',(item,)).fetchone()
        need(row is not None,'Unindexed item')
        root=self.root(row[0])
        label,winner,split=self.db.execute('SELECT label,winner,winner_split FROM dd WHERE id=?',(root,)).fetchone()
        return dict(cluster_id=identity(dict(policy=identity(POLICY),minimum_member=label)),canonical_item=winner,split=split)

    def verify(self):
        need(self.db.execute('PRAGMA quick_check').fetchone()[0]=='ok','Corrupt dedup database')
        for node,blob,checksum in self.db.execute('SELECT id,shingles,checksum FROM dd'):
            need(hashlib.sha256(blob).hexdigest()==checksum,'Corrupt dedup features')
            expected=unpack(blob)[:POLICY['anchors']]
            found=[row[0] for row in self.db.execute('SELECT anchor FROM anchors WHERE doc=? ORDER BY anchor',(node,))]
            need(found==expected,'Corrupt dedup anchor index')
            self.root(node)
