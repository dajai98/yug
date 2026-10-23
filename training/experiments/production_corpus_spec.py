"""Deterministically build the v1 specification using retained evidence; no network."""
import json
import re
from pathlib import Path
from training.production_spec import QUOTAS, TOKENIZER, CHECKPOINTS, seal, validate
from training.production_corpus import file_digest
from training.experiments.rep_corpus_policy import GLOBAL_EXCLUDED_PATTERNS, GENERATED_FILE_MARKERS

# Owner-requested discovery names, NOT independently verified licenses or URLs.
CANDIDATES = {
'code': 'CPython selected core|Django|Go core and toolchain|Rust core std compiler|LLVM Clang libc++|TypeScript|Node.js core|React|Godot core|glTF-Blender-IO|OpenSubdiv|meshoptimizer|Assimp code docs',
'csharp_gamedev': 'O3DE core|SDL|GameNetworkingSockets|ENet|Unreal compatible open C++|engine architecture game AI editor networking|Godot 3D animation skeleton docs|individually cleared Unity open repos',
'graphics': 'Khronos Vulkan-Docs|Khronos Vulkan-Samples|SPIRV-Tools|SPIRV-Headers|GLSL|glslang|wgpu|Dawn Tint|Filament|bgfx|ANGLE|Vulkan Validation Layers|Vulkan Memory Allocator|The Forge|raylib shaders|Mesa selected files',
'algorithms': 'TheAlgorithms Python|TheAlgorithms C++ code|williamfiset Algorithms|keon algorithms|javascript-algorithms|cp-algorithms|Jeff Erickson Algorithms|NetworkX algorithms|Boost algorithms graph containers|petgraph|Apache Commons Collections|LLVM libc++ algorithms',
'math_science': 'GLM|JOML|Eigen|nalgebra|glam|NumPy math|SciPy numerical|SymPy|libigl|Box2D|Bullet Physics core|Box2D-Lite',
'linux_systems': 'Docker CLI|Podman|CMake|Meson|Ninja|systemd|Linux kernel selected docs',
'security': 'MITRE CWE|OpenSSF secure development|OWASP Cheat Sheet Series|OWASP WSTG|Google OSS-Fuzz|pip-audit|OpenSSF supply chain|safe cryptography examples|defensive static analysis',
'instruction': 'OASST1|Dolly 15K|Natural Instructions cleared components|FLAN cleared components',
'reasoning': 'MATH|GSM8K|selected reasoning tasks',
'conversation': 'OASST1 conversation subset|YUG-owned conversations',
'tool_use': 'ToolBench cleared components|BFCL benchmark only|YUG-owned game-dev debugging tool-use',
'general_english': 'FineWeb filtered|FineWeb-Edu high score|curated explanatory text',
'educational_reference': 'Wikibooks|Wikiversity|NIST works|NASA government works|FineWeb-Edu instructional|open reference handbooks',
'nepali': 'AI4Bharat Sangraha Nepali|IndicCorp v2 Nepali|IndicTrans2 BPCC cleared components|Titung Nepali-English|Sangraha romanized Nepali|YUG-owned technical Nepali-English',
'software_docs': 'OpenTelemetry docs|Prometheus docs|open standards explicit reuse|Blender Manual|Blender Developer Docs|Blender Python API|Khronos glTF specification|OpenAssetIO',
'software_engineering': 'curated project tests debugging build reviews|YUG-owned debugging explanations',
}
REJECTED = 'proprietary Unreal Engine bulk source|unknown-license scraped data|random social-media scraping|exploit malware payload collections|non-commercial-only sources|AI-training-prohibited sources|uncleared vendored third-party content|MIT OCW default|OpenStax default|CK-12 default|Microsoft Learn broad generic-terms scrape'
CONDITIONAL_WORDS = ('blender','llvm','eigen','libigl','mesa','systemd','linux kernel','owasp','cp-algorithms','erickson','nist','nasa','wikibooks','wikiversity','fineweb','sangraha','indiccorp','bpcc','toolbench','bfcl')
TOPICS = [('modeling_workflows','csharp_gamedev',20),('topology_mesh_retopology','math_science',15),
 ('rigging_skeleton_skin_weights','csharp_gamedev',15),('animation_ik_fk_retarget','csharp_gamedev',15),
 ('uv_material_pbr','graphics',10),('blender_python_procedural','code',10),
 ('asset_interchange','software_docs',10),('lod_optimization','graphics',5)]
SUBTOPICS = {
'code':['python','c_cpp','csharp','rust','go','javascript_typescript','blender_python_procedural'],
'csharp_gamedev':['unity_ecosystem','csharp','engine_architecture','networking','build_debug','modeling_workflows','rigging_skeleton_skin_weights','animation_ik_fk_retarget'],
'software_engineering':['debugging','testing','build_integration','serialization_state','performance_budgets','specification_verification'],
'general_english':['explanatory','long_form','reference_prose'],
'educational_reference':['instructional_prose','handbooks','reference'],
'nepali':['native_nepali','parallel','technical_transfer','natural_codeswitch','romanized_limited'],
'software_docs':['api','specifications','build','asset_interchange'],
'graphics':['shaders','gpu','uv_material_pbr','lod_optimization'],
'math_science':['math','geometry','physics','topology_mesh_retopology'],
'algorithms':['algorithms','data_structures','complexity'],
'linux_systems':['linux','build_tools','containers','cli'],
'security':['secure_development','supply_chain','defensive_analysis'],
'instruction':['natural_instructions','qa'], 'reasoning':['math_reasoning','debug_reasoning'],
'tool_use':['owned_tool_traces','game_dev_tools'], 'conversation':['dialogue','clarification']}

def build(root):
    root=Path(root)
    oldpath='training/resources/phase1/rep_corpus_10m_v1/01_source_policy.json'
    old=json.loads((root/oldpath).read_text())
    deny=list(GLOBAL_EXCLUDED_PATTERNS)+['**/.git/**','**/dist/**','**/build/**','**/*.dll','**/*.exe','**/*.bin','**/*.fbx','**/*.blend','**/*.png','**/*.jpg','**/*.glb','**/*.obj']
    sources=[]
    prior={s['source_id']:s for s in json.loads((root/'training/resources/phase1/tokenizer_fit_sample_gate_v1/source_registry.json').read_text())}
    for s in old['source_policy']:
        cat='nepali' if s['category']=='nepali_english_codeswitch' else s['category']
        facts=s['facts']; status={'BLOCKED':'REJECT','UNKNOWN':'REVIEW'}.get(s['stage1c_status'],s['stage1c_status'])
        lic=facts.get('license_used_for_eligibility') or facts.get('license_expression')
        if status=='ACCEPT' and lic and ('CC-BY-SA' in lic or lic == 'MS-PL'):
            status='CONDITIONAL'
        source=dict(source_id=s['source_id'],name=facts['project'],category=cat,
            subcategory=s['subcategory'] or 'curated_subset',status=status,prior_status=s['stage1c_status'],
            status_reason=s['reason'],status_authority='retained Stage 1C; CC BY-SA and MS-PL conditional under production request',
            license_id=lic,license_evidence=(facts.get('evidence_ref') or '').replace('artifacts/rep_corpus_10m_v1/', 'training/resources/phase1/rep_corpus_10m_v1/') or None,license_text_sha256=facts.get('license_text_sha256'),
            revision=facts.get('commit') or prior.get(s['source_id'],{}).get('revision'),locator=prior.get(s['source_id'],{}).get('canonical_id'),
            target_cap_tokens=0 if status=='REJECT' else QUOTAS[cat]//10,
            cap_basis='v1 planning ceiling; not a yield estimate',
            repository_group=s['source_id'],domain_group=s['source_id'],split_group=s['source_id'],
            split='unassigned',allow_paths=[],deny_paths=sorted(set(deny+s['proposal']['excluded_path_patterns'])),
            clearance_record=None,file_manifest=None,provenance=dict(legacy_policy=oldpath,legacy_source_id=s['source_id']),
            tags=[s['subcategory']],training_allowed_after_clearance=status!='REJECT')
        sources.append(source)
    for cat,names in CANDIDATES.items():
        for name in names.split('|'):
            sid='candidate-'+re.sub('[^a-z0-9]+','-',name.lower()).strip('-')
            sources.append(dict(source_id=sid,name=name,category=cat,subcategory='curated_subset',
                status='CONDITIONAL' if any(w in name.lower() for w in CONDITIONAL_WORDS) else 'REVIEW', prior_status=None,
                status_reason='Owner-requested candidate; pin, component rights and file scope not yet verified',
                status_authority='production milestone request; no new license verification',
                license_id=None,license_evidence=None,license_text_sha256=None,revision=None,locator=None,
                target_cap_tokens=QUOTAS[cat]//10,cap_basis='v1 planning ceiling; not a yield estimate',
                repository_group=sid,domain_group=sid,split_group=sid,split='unassigned',
                allow_paths=[],deny_paths=deny,clearance_record=None,file_manifest=None,
                provenance={'discovery_name':name},tags=['candidate'],training_allowed_after_clearance='BFCL' not in name))
    for name in REJECTED.split('|'):
        sid='excluded-'+re.sub('[^a-z0-9]+','-',name.lower()).strip('-')
        sources.append(dict(source_id=sid,name=name,category='general_english',subcategory='excluded_policy_class',
            status='REJECT',prior_status=None,status_reason='Explicit owner default exclusion; not a blanket license finding',
            status_authority='production milestone request',license_id=None,license_evidence=None,
            license_text_sha256=None,revision=None,locator=None,target_cap_tokens=0,cap_basis='excluded',
            repository_group=sid,domain_group=sid,split_group=sid,split='unassigned',allow_paths=[],deny_paths=['**'],
            clearance_record=None,file_manifest=None,provenance={'exclusion_rule':name},tags=['policy_exclusion'],training_allowed_after_clearance=False))
    aliases=[['candidate-llvm-clang-libc','candidate-llvm-libc-algorithms'],
        ['candidate-oasst1','candidate-oasst1-conversation-subset'],
        ['candidate-fineweb-edu-high-score','candidate-fineweb-edu-instructional','candidate-fineweb-filtered'],
        ['candidate-godot-core','candidate-godot-3d-animation-skeleton-docs'],
        ['candidate-blender-manual','candidate-blender-developer-docs','candidate-blender-python-api'],
        ['candidate-ai4bharat-sangraha-nepali','candidate-sangraha-romanized-nepali'],
        ['candidate-cpython-selected-core','python-docs-cpython-doc']]
    byid={s['source_id']:s for s in sources}
    for group in aliases:
        for sid in group:
            if sid in byid:
                for field in ('repository_group','split_group','domain_group'):
                    byid[sid][field]=group[0]
    for s in sources:
        if 'wikimedia' in s['source_id'] or 'wikipedia' in s['source_id'] or s['name'] in ('Wikibooks','Wikiversity'):
            s['domain_group']='wikimedia-origin'
    for source in sources:
        if source['source_id']=='candidate-yug-owned-game-dev-debugging-tool-use':
            source['target_cap_tokens']=QUOTAS['tool_use']
            source['cap_basis']='Owner requires YUG-owned tool-use dominance; ceiling permits up to full tool-use quota, not guaranteed yield.'
        allowed={source['category']}
        if source['category'] in ('code','csharp_gamedev','software_docs'):
            allowed.add('software_engineering')
        if any(word in source['name'].lower() for word in ('blender','gltf','libigl','opensubdiv','meshoptimizer','assimp','openassetio','godot')):
            allowed.update(('code','csharp_gamedev','software_docs','graphics','math_science'))
        source['allowed_categories']=sorted(allowed)
        source['category_caps_tokens']={c:0 if source['status']=='REJECT' else QUOTAS[c]//10 for c in sorted(allowed)}
        if source['source_id']=='candidate-yug-owned-game-dev-debugging-tool-use':
            source['category_caps_tokens']['tool_use']=QUOTAS['tool_use']
        source['origin_verified']=False
        source['historical_scope_note']='Retained sample/source evidence only; production file scope and canonical origin clearance pending.'
    registry=seal(dict(schema='yug-production-source-registry-v1',sources=sorted(sources,key=lambda s:s['source_id'])))
    evidence_paths=[oldpath,'training/resources/phase1/rep_corpus_10m_v1/00_source_license_evidence.json',
        'training/resources/phase1/tokenizer_fit_sample_gate_v1/source_registry.json','training/resources/phase1/tokenizer_fit_sample_gate_v1/acquisition.json',
        'training/resources/phase1/model_initialization_gate_v1/model_config.json','training/resources/phase1/model_initialization_gate_v1/freeze.json',
        'training/resources/phase1/bpe_fitter_optimization_gate_v1/comparison/run-4096-0/tokenizer.json',
        'training/production_corpus.py','training/token_shards.py','training/corpus_prep.py',
        'training/experiments/representative_mini_corpus.py','training/experiments/rep_corpus_policy.py',
        'training/experiments/rep_corpus_sources.py','training/checkpoint.py','training/loss.py',
        'training/loop.py','training/evaluate.py','yug/native/model.py','yug/native/config.py','yug/tokenizer/bpe.py']
    caps={field:{} for field in ('repository_group','domain_group')}
    for s in sources:
        caps['repository_group'][s['repository_group']]=max(caps['repository_group'].get(s['repository_group'],0),s['target_cap_tokens'])
        caps['domain_group'][s['domain_group']]=max(caps['domain_group'].get(s['domain_group'],0),0 if s['status']=='REJECT' else QUOTAS[s['category']]//4)
    caps['domain_group']['candidate-yug-owned-game-dev-debugging-tool-use']=QUOTAS['tool_use']
    spec=seal(dict(schema='yug-production-corpus-spec-v1',registry_identity=registry['identity'],
        tokenizer_identity=TOKENIZER,vocab_size=4096,total_clean_train_tokens=2_000_000_000,
        routing='Source category is a planning default. Every cleared manifest file fixes exactly one owner among allowed_categories; tags never add quota credit.',
        categories=[dict(id=c,target_tokens=q,subcategories=sorted(set(SUBTOPICS[c]+[s['subcategory'] for s in sources if s['category']==c and s['status']!='REJECT'])),non_3d_target_tokens=q-sum(n*1_000_000 for _,owner,n in TOPICS if owner==c)) for c,q in QUOTAS.items()],
        checkpoints=CHECKPOINTS,group_caps=caps,
        cap_policy='Source 10% of owner target; repository max member source cap; domain max member 25% owner target. YUG-owned tool-use has a documented full-bucket cap exception to permit required majority. Planning ceilings, not yield. Resolve canonical origins before clearance.',
        three_d=dict(total_tokens=100_000_000,topics=[dict(id=t,owner=c,target_tokens=n*1_000_000) for t,c,n in TOPICS],
            form_targets=dict(authoritative=30_000_000,implementation=25_000_000,workflow=20_000_000,troubleshooting=10_000_000,procedural_tasks=10_000_000,instruction_qa=5_000_000),
            accounting='One primary topic and form per document; two views of the same 100M, never additive.',
            purpose='education reasoning scripting troubleshooting future tool-control; not native 3D generation'),
        accounting=dict(one_budget_owner=True,validation_counts_toward_2b=False,
            unit='frozen-tokenizer clean accepted unique training tokens; raw downloads, validation and repeated epochs do not fill quotas',
            tags=['domain','language','format','difficulty','3d_topic','3d_form'],
            boundary='whole documents; no silent truncation or duplication to force exact quota; residual deficits must be reported',
            tokenization='per-document-concat-no-separators-v1'),
        fallback=dict(cross_category_transfer='forbidden',within_category='Only cleared sources of same owner and coverage need, under all caps; journal before acquisition.',
            unavailable='Keep deficit; no repeated documents, synthetic filler, auto-promotion or cap expansion.'),
        source_policy=dict(preferred=['MIT','Apache-2.0','BSD','CC0','CC BY'],
            conditional=['CC BY-SA','MPL','LLVM exception','mixed licenses','government works','other historically allowed licenses'],
            default='REVIEW; status alone is not acquisition clearance',exclusion_rules=REJECTED.split('|'),
            tool_use_ownership='YUG-owned game-dev/debugging/tool examples must be a strict majority of clean tool-use training tokens at release',benchmark_contamination='BFCL validation-only by default; benchmark/task upstream groups excluded from training'),
        filters=dict(deny_paths=deny,generated_markers=list(GENERATED_FILE_MARKERS),
            manual_quality=['boilerplate','generated enum dumps','SEO','AI-generated junk','outdated workflow','irrelevant assets','duplicate examples'],
            quality_proof='Retain reviewed samples and reasons; heuristics do not prove authorship or detect all junk.'),
        dedup=dict(exact='Global normalized UTF-8 SHA256 across sources/categories/splits; one canonical owner.',
            near='Global lexical 5-word shingles; flag Jaccard >=0.5 or containment >=0.8; review clusters and shared boilerplate before release.',
            heldout='Source origins and duplicate clusters cannot cross splits; exclude prior held-out fit/transfer/evaluation material from train.'),
        split=dict(unit='canonical source group plus duplicate clusters',seed=1234,validation_fraction=None,
            state='Explicit assignment before acquisition; owner has not specified production validation budget. Do not silently copy mini-corpus 20%.'),
        resume=dict(acquisition='Immutable request + cursor + completed payload SHA256/size; verify before retry; no mutable HEAD.',
            preparation='Existing prepare_stream is not resumable on incomplete directories.',
            tokenization='Existing build_token_shards verifies request identity, committed spool and manifest-last publication.'),
        checkpoint_metrics=['accepted_clean_train_tokens','raw_tokens','rejected_tokens','category_progress','source_progress','duplication_rate','license_status','filter_counts','source_diversity','tokenizer_statistics','validation_split_status','remaining_deficits'],
        runtime_obligations=[
            'Clear immutable source/file manifests, canonical origins and per-path rights; sample acceptance is not production clearance.',
            'Assign source-disjoint validation groups and separate budget; prepare_stream currently splits documents.',
            'Implement resumable acquisition/preparation coordinator with global disk-backed ownership/dedup and quota enforcement; token-shard resume exists.',
            'Implement and size global near-duplicate/boilerplate index before production release; current prepare_stream does exact dedup only.',
            'Route expanded categories through the existing structure-preserving normalizer explicitly; legacy normalize_document recognizes legacy categories only.',
            'Prove cleared natural 3D workflow/rigging/animation/troubleshooting yield, not documentation volume alone.'
        ],evidence=[dict(path=p,sha256=file_digest(root/p)) for p in evidence_paths],
        training_readiness='CLOSED per latest owner instruction; external T4 evidence not re-adjudicated here',
        acquisition_started=False,training_authorized=False))
    validate(spec,registry,root)
    return spec,registry

def main():
    spec,registry=build(Path('.'))
    out=Path('training/specifications'); out.mkdir(exist_ok=True)
    for name,data in [('production_corpus_v1.json',spec),('source_registry_v1.json',registry)]:
        (out/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Wrote',len(registry['sources']),'sources; quota',sum(QUOTAS.values()))

if __name__=='__main__': main()
