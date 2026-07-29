# Architecture Review — AUT KG Extraction Pipeline

## Current implementation checkpoint (2026-07-16)

Work in `backend/schemas/process_knowledge/text.py` currently has the following behavior:

- `Instruction.objects` maps each segment ID to that segment's `list[ObservedObject]`.
- `Instruction.actors` maps each segment ID to the ordered, de-duplicated actor names extracted from `ObservedAction.actor`; text `Segment` therefore no longer requires a separate `actors` field.
- `VideoSegment` retains its directly observed `actors: list[ObservedActor]` field.
- During `VideoSegment` validation, repeated object and actor names are automatically numbered in list order: two `cup` objects become `cup_1` and `cup_2`, and two `worker` actors become `worker_1` and `worker_2`. Unique or already distinct names remain unchanged.
- The normalization currently changes only entries in `VideoSegment.objects` and `VideoSegment.actors`. References stored as strings in `ObservedAction.actor`, `.object`, `.instrument`, or `.target` are not rewritten because a duplicate unnumbered reference cannot be assigned unambiguously to one instance. This should be addressed when entity identity/reference resolution is designed.
- Focused coverage is in `tests/test_text_schemas.py`. Current verification: 3 tests pass and Ruff reports no findings for the two changed files.

*Basis: single-commit repo (`3ce191f`), all statements below are grounded in file contents cited inline.*

## Inferred intended architecture

Reading the README pipeline diagram (`backend/{schemas,extraction,llm,graph,training,evaluation,pipeline}`), the module docstrings, and the notebook's narrative, the intended flow is exactly the 9-stage pipeline given in the prompt: raw transcript → cleaned/chunked text → a graph of chunk dependencies (likely for coreference/context across chunks) → decomposition into extraction units → per-unit LLM extraction → per-unit local KGs → merge/dedup into one graph → evaluation against gold data. Today the repo implements a **linear demo slice**: `Transcript(=raw string) → LLM Extraction → (implicit) Local Knowledge Graph(=Pydantic object graph, never persisted) → Evaluation`. Preprocessing, Dependency Graph, Decomposition, and Graph Merge do not exist in any form.

## Stage-by-stage assessment

**1. Transcript (input handling)**
- Status: **Not implemented.** No ingestion, file-loading, or transcript-representation code anywhere in `backend/`.
- Relevant files: none. Input is hardcoded strings in `backend/schemas/process_knowledge/examples.py` (`LEVEL_1_TEXT` … `LEVEL_5_TEXT`) and notebook cells.
- Existing APIs: none.
- Missing abstractions: a `Transcript`/`Document` type carrying raw text + metadata (speaker turns, timestamps) — `KGEntity`/`ExtractionResult` in `backend/schemas/base.py:33-44` has `source_chunk: Optional[str]` but nothing produces it.
- Technical debt: none yet (nothing to accrue debt on).
- Blockers: real transcripts (format, source system) are undefined; no sample data beyond synthetic German examples.
- Recommended order: **implement early**, since every downstream stage currently substitutes hardcoded strings for it.

**2. Preprocessing**
- Status: **Not implemented.** No cleaning, normalization, or chunking code. `ExperimentConfig.chunking_strategy: Optional[str]` (`backend/evaluation/experiment.py:29`) is a bare metadata field with comment `# "none", "fixed_500", "semantic"` — a label for a strategy that is never executed anywhere.
- Relevant files: `backend/evaluation/experiment.py:29` (field only).
- Existing APIs: none.
- Missing abstractions: a `preprocessing` module/function turning `Transcript → List[TextChunk]`.
- Technical debt: the `chunking_strategy` field is speculative/dead — it documents an intention with no implementation, which is misleading if left as-is.
- Blockers: depends on Transcript stage existing first; chunking strategy choice depends on what "Dependency Graph" stage needs as input granularity.
- Recommended order: after Transcript, before Dependency Graph.

**3. Dependency Graph**
- Status: **Not implemented.** No graph library dependency (no `networkx` or similar in `pyproject.toml`), no code referencing cross-chunk dependencies, coreference, or ordering constraints.
- Relevant files: none.
- Existing APIs: none.
- Missing abstractions: entirely undefined — the repo gives no signal of what "dependency" means here (temporal step order? coreference chains? document structure?). This is the least-specified stage in the whole intended architecture relative to the codebase.
- Technical debt: n/a.
- Blockers: **conceptually blocking** — without a definition of what this graph represents, Decomposition (stage 4) can't be scoped either, since decomposition presumably slices work along dependency-graph boundaries.
- Recommended order: needs a design spike/decision before any implementation; do not build Decomposition against an assumed shape.

**4. Decomposition**
- Status: **Not implemented.** No code splits a document into extraction units for the LLM stage.
- Relevant files: none directly, but `backend/extraction/extractor.py` implicitly assumes decomposition already happened — `Extractor.extract`/`extract_list` (`extractor.py:53-117`) take one flat `text: str` and one target schema, i.e., the "decomposed unit → schema" mapping is the caller's responsibility today, done manually per notebook cell (`LEVEL_1_TEXT`, `Tool` / `LEVEL_2_TEXT`, `Procedure` etc.).
- Existing APIs: `Extractor.extract(text, response_model)` / `extract_list(text, item_model)` are the seam decomposition output would need to plug into.
- Missing abstractions: a `Decomposer`/`ExtractionUnit` type pairing a chunk with the schema(s) to extract, and orchestration to run `Extractor` over each unit and collect `ExtractionResult`s (`base.py:33-44`, defined but never constructed by any code).
- Technical debt: none yet, since nothing exists; but note `ExtractionResult` is a genuine dead abstraction — a fully-specified container (`entities`, `relations`, `source_chunk`, `extraction_model`, `extraction_timestamp`) that no code instantiates or returns. `Extractor` never fills it.
- Blockers: depends on Preprocessing/Dependency Graph producing units; the schema-per-unit selection strategy (single schema per call today) may not generalize to "extract everything found in this chunk."
- Recommended order: after Preprocessing; can be prototyped in parallel with Dependency Graph once chunk shape is fixed, but real implementation should wire into `ExtractionResult`.

**5. LLM Extraction**
- Status: **Implemented and working** (verified: `uv run pytest` passes; extraction requires local Ollama, not exercised by tests).
- Relevant files: `backend/llm/client.py` (`get_openai_client`, `get_client`, `extract_structured`, lines 34-83), `backend/extraction/extractor.py` (`Extractor` class, full file).
- Existing APIs: `extract_structured(text, response_model, model, system_prompt, temperature, max_retries)`; `Extractor.extract(text, response_model, system_prompt=None)`; `Extractor.extract_list(text, item_model, system_prompt=None)`.
- Missing abstractions: no batching/async support (one `OpenAI`/`instructor` client constructed per call in `get_client()`, `client.py:42-47` — no reuse/pooling); no combined "extract everything relevant" call — only single-schema extraction, so a chunk containing entities *and* relations needs multiple sequential calls with no coordination layer.
- Technical debt: `get_client()` is called fresh inside `extract_structured` every invocation (`client.py:72`), rebuilding the OpenAI+instructor client per call — wasteful once used at scale; no error handling/retry-exhaustion path exposed to callers (Instructor's `max_retries` is passed through but failures aren't caught, as seen in the notebook's own `try/except Exception` workaround for `StepOrder` extraction).
- Blockers: none technical; quality is model-dependent (notebook explicitly documents Level 4/5 struggling with small models).
- Recommended order: already done; next work is integration into a decomposition-driven loop, not the extraction call itself.

**6. Local Knowledge Graphs**
- Status: **Not implemented as a distinct concept.** The closest artifact is the in-memory Pydantic object graph returned by a single `Extractor.extract`/`extract_list` call (e.g., a `StepExecution` object nesting `Step`, `Worker`, `Tool` — `backend/schemas/process_knowledge/relations.py:74-93`). There is no per-chunk graph object, no graph data structure (nodes/edges), and no persistence.
- Relevant files: `backend/schemas/base.py` (`ExtractionResult`, unused), `backend/schemas/process_knowledge/{entities,relations}.py` (the node/edge *types*, well developed).
- Existing APIs: none for graph construction — only the Pydantic model constructors themselves.
- Missing abstractions: a `LocalKnowledgeGraph` structure that collects the various entities/relations extracted from one chunk into a single addressable graph (nodes with IDs, edges as reified relation records) — needed as the unit that stage 7 (Merge) operates over.
- Technical debt: entity identity is currently only `name: str` (`base.py:15`) with no stable ID — merge/dedup downstream will need this resolved first.
- Blockers: depends on Decomposition existing to know what "one local graph" scopes over.
- Recommended order: right after Decomposition + LLM Extraction are wired together; this is the natural checkpoint to introduce a real graph structure instead of loose Pydantic objects.

**7. Graph Merge**
- Status: **Not implemented.** No dedup/entity-resolution/merge code exists. `KGRelation`'s docstring (`base.py:20-29`) explains reification theory for Neo4j/LadybugDB/TypeDB but implements nothing. `backend/schemas/base.py:38` says "deduplication happen[s] downstream" — an explicit acknowledgment in-code that this stage is deferred, not yet started.
- Relevant files: `backend/schemas/base.py:20-38` (design intent only).
- Existing APIs: none. `backend/evaluation/matching.py`'s `compare_values`/`match_token_overlap`/`match_normalized` (`matching.py:23-142`) are the closest reusable primitives (fuzzy string matching) but are wired only into evaluation, not merge.
- Missing abstractions: entity resolution/dedup logic, a merge algorithm reconciling multiple `LocalKnowledgeGraph`s into one, and a decision on target backend (Neo4j/LadybugDB property-graph reification vs. TypeDB native n-ary relations — both floated in docstrings, neither chosen).
- Technical debt: none accrued (nothing built), but the backend choice is an open decision blocking everything downstream of it.
- Blockers: **backend choice** (property graph vs. TypeDB) must be made before merge logic can be written, since reification strategy differs; also blocked on stable entity IDs (see stage 6).
- Recommended order: after Local Knowledge Graphs exist as real objects; matching primitives from `backend/evaluation/matching.py` should likely be extracted/reused here rather than duplicated.

**8. Knowledge Graph (persisted/queryable store)**
- Status: **Not implemented.** No database client, driver, or storage code (`pyproject.toml` dependencies: `instructor`, `numpy`, `openai`, `pydantic`, `python-dotenv`, `sentence-transformers` — no `neo4j`, no `typedb-client`, no graph DB driver at all).
- Relevant files: none. `backend/graph/` referenced in README's tree diagram does not exist on disk.
- Existing APIs: none.
- Missing abstractions: insertion/lookup/query API (`backend/graph/` per README comment "insert, lookup, dedup") — entirely a stub in documentation only.
- Technical debt: README's project-structure section documents `backend/graph/`, `backend/training/`, `backend/pipeline/` as if they exist — this is **aspirational documentation drift**: a new contributor following the README will `cd` into directories that don't exist.
- Blockers: backend selection (Neo4j vs. TypeDB vs. LadybugDB — three candidates named, zero chosen) is a hard prerequisite.
- Recommended order: after Graph Merge design is settled; this is a good candidate for an early spike/prototype given it blocks the most downstream work.

**9. Evaluation**
- Status: **Implemented and tested** — the most mature stage in the repo (7 passing tests in `tests/test_eval.py`).
- Relevant files: `backend/evaluation/matching.py`, `evaluators.py`, `metrics.py`, `experiment.py`; gold data in `backend/schemas/process_knowledge/examples.py`.
- Existing APIs: `compare_values(expected, extracted, strategy, threshold)`; `evaluate_entity_detection`, `evaluate_attributes`, `evaluate_relation_detection`, `evaluate_extraction_level` (`evaluators.py`); `compute_prf1`, `aggregate_prf1`, `format_prf1`, `format_prf1_table` (`metrics.py`); `ExperimentConfig`/`ExperimentResult` (`experiment.py`).
- Missing abstractions: evaluation currently operates only on **pre-extraction** Pydantic objects (entity/attribute/relation level) — there is no evaluator for the *merged, persisted* Knowledge Graph (stage 8) or for graph-level structural quality (e.g., precision/recall over the final merged graph vs. a gold graph). `ExperimentResult` fields (`entity_metrics`, `attribute_metrics`, `relation_metrics`, `structure_metrics`) anticipate this but nothing populates `structure_metrics` (`experiment.py:40`).
- Technical debt: `evaluate_entity_detection`/`evaluate_relation_detection` use **greedy best-match**, not optimal bipartite matching (`evaluators.py:42-72`, `195-241`) — can under- or over-count TP/FP/FN when multiple candidates have similar scores; a known-but-undocumented approximation. Also `evaluators.py:122` triggers a Pydantic deprecation warning (`model_fields` accessed on instance, not class) — confirmed via `uv run pytest -v` output.
- Blockers: none for current scope; extending to graph-level evaluation is blocked on Graph Merge/Knowledge Graph stages existing.
- Recommended order: keep evaluation logic, but plan to extend it once stages 6–8 exist; fix the greedy-matching and deprecation issues opportunistically now since they're cheap and isolated.

## 1. Dependency diagram (current, as implemented)

```
main.py  (standalone "Hello!" — not wired to anything)

notebooks/01_extraction_basics.ipynb
        │
        ▼
backend/schemas/process_knowledge/examples.py ───────┐
        │  (imports)                                 │ (gold data)
        ▼                                             ▼
backend/schemas/process_knowledge/{entities,relations}.py
        │  (imports KGEntity/KGRelation)
        ▼
backend/schemas/base.py   (KGEntity, KGRelation, ExtractionResult[unused])
        ▲
        │ (imports schema types as response_model / item_model)
backend/extraction/extractor.py  (Extractor)
        │  (imports)
        ▼
backend/llm/client.py  (extract_structured, get_client) ──► Ollama (external, localhost:11434)

backend/evaluation/matching.py  (MatchStrategy, compare_values)
        ▲
        │ (imports)
backend/evaluation/metrics.py  (PRF1, compute_prf1, format_prf1)
        ▲
        │ (imports both)
backend/evaluation/evaluators.py  (evaluate_entity_detection, evaluate_attributes, evaluate_relation_detection, evaluate_extraction_level)
        ▲
        │ (imports evaluators + matching + metrics)
backend/evaluation/experiment.py  (ExperimentConfig, ExperimentResult — currently unused by evaluators.py)

tests/test_eval.py  ───► imports entities/relations, matching, metrics, experiment, evaluators (exercises everything above except llm/client.py and extractor.py)
```

Missing nodes (named in README/docstrings, absent from disk): `backend/schemas/organizational_knowledge/`, `backend/graph/`, `backend/training/`, `backend/pipeline/`. No module currently implements Preprocessing, Dependency Graph, or Decomposition, so there is no code-level edge between "raw text" and `Extractor` other than manual copy-paste of example strings.

## 2. Prioritized implementation roadmap

1. **Fix documentation/reality mismatch** — README's project-structure tree lists `graph/`, `training/`, `pipeline/`, `organizational_knowledge/` as if present; either scaffold empty modules with clear "not yet implemented" docstrings or mark them explicitly as planned in the README, so `backend.schemas.organizational_knowledge` (referenced in the notebook's "Next Steps" cell) doesn't 404 for a new contributor.
2. **Transcript + Preprocessing** — define a `Transcript`/chunk representation and a chunking function; this unblocks everything downstream and lets `examples.py` strings be replaced by a real ingestion path.
3. **Decomposition + wire into `Extractor`** — build the orchestration loop that takes chunks, decides which schema(s) to run, calls `Extractor`, and actually populates `ExtractionResult` (currently dead code) instead of returning bare Pydantic objects per call.
4. **Local Knowledge Graph structure** — introduce a real graph object (stable entity IDs, node/edge collection) as `Extractor`'s output target, replacing loose per-call objects.
5. **Entity ID + resolution scheme** — before merge, decide identity (e.g., add a stable `id` field to `KGEntity`) since `name`-only matching (used today only in evaluation) is not a sound identity key for merge.
6. **Graph backend decision** (Neo4j vs. TypeDB vs. LadybugDB) — a spike/decision doc; this single decision unblocks both Graph Merge and Knowledge Graph stages simultaneously.
7. **Graph Merge** — implement dedup/merge, reusing `backend/evaluation/matching.py`'s comparison primitives rather than re-implementing fuzzy matching.
8. **Knowledge Graph persistence layer** (`backend/graph/`) — insert/lookup APIs per README's stated intent.
9. **Dependency Graph stage** — deliberately last because its purpose is currently underspecified in the codebase; don't build it until a concrete need (e.g., cross-chunk coreference) is demonstrated by stages 2–4 in practice.
10. **Extend Evaluation to graph-level metrics** — populate `ExperimentResult.structure_metrics`, add a graph-vs-gold-graph evaluator once stage 8 exists.

(Steps 2–4 can proceed in parallel with the ruff cleanup already identified in `examples.py`, which is unrelated but trivial.)

## 3. Five highest-risk architectural decisions

1. **Graph backend choice (Neo4j/property-graph vs. TypeDB vs. LadybugDB)** — named as open in two separate docstrings (`base.py:24-28`, `relations.py:10-11`) with no dependency committed in `pyproject.toml`. This single choice determines the reification strategy for every `KGRelation`, the Merge algorithm's shape, and the persistence API — reversing it later means rewriting stages 6–8 entirely.
2. **Entity identity model** — `KGEntity` has only `name: str` as an identifying field (`base.py:13-17`). Merge/dedup across chunks cannot work correctly on free-text names alone (as evaluation's own token-overlap fuzzy matching demonstrates is necessary even for read-only comparison). Deciding on IDs late risks reprocessing all extracted data.
3. **Single-schema-per-call extraction model** — `Extractor.extract`/`extract_list` (`extractor.py:53-117`) require the caller to already know which schema to target. Scaling to full-transcript extraction (where a chunk may contain entities *and* several relation types simultaneously) requires either many sequential calls per chunk (cost/latency risk) or a redesign to a "extract everything" call — this affects Decomposition design and LLM cost model significantly.
4. **Chunking/decomposition granularity** — `ExperimentConfig.chunking_strategy` (`experiment.py:29`) enumerates `"none" | "fixed_500" | "semantic"` as a label with no implementation or evidence which is viable for this domain's German industrial-procedure text; wrong granularity risks breaking the nested-relation extraction quality already shown to degrade at Level 4–5 in the notebook.
5. **Matching/dedup threshold reuse across evaluation vs. merge** — `compare_values`/`MatchStrategy` (`matching.py`) is tuned today for scoring extraction quality against gold data (thresholds like `0.6` picked ad hoc per call site in `evaluators.py`/`tests/test_eval.py`). If Graph Merge reuses the same thresholds for real-world entity resolution (a different, high-stakes decision — silently merging two distinct tools is worse than a scoring error), those thresholds need independent validation, not inherited defaults.

## 4. Concrete refactoring recommendations before additional features

- **Populate and actually use `ExtractionResult`** (`base.py:33-44`): make `Extractor` return/assemble this container instead of bare model instances, so downstream stages have one consistent per-chunk output type to build on.
- **Add stable entity IDs to `KGEntity`** now (e.g., a deterministic uuid or content hash), before any merge code is written, since retrofitting IDs after data/tests exist is more costly.
- **Cache/reuse the LLM client**: `extract_structured` calls `get_client()` fresh every invocation (`client.py:72`); refactor `Extractor` to hold a client instance instead of constructing one per `extract()`/`extract_list()` call, ahead of adding a decomposition loop that will call this in a tight loop.
- **Replace greedy matching with optimal assignment** (e.g., Hungarian algorithm) in `evaluate_entity_detection`/`evaluate_relation_detection` (`evaluators.py:42-91`, `167-255`) before reusing this matching code for Graph Merge, where mismatches have larger consequences than in evaluation.
- **Reconcile README with actual file tree**: remove or clearly flag `graph/`, `training/`, `pipeline/`, `organizational_knowledge/` as "planned, not implemented" to prevent broken imports (the notebook already references `backend.schemas.organizational_knowledge`, which doesn't exist).
- **Fix the two live ruff/lint findings** in `backend/schemas/process_knowledge/examples.py` (unused imports `PPE`, `QualificationRecord`) and the `model_fields`-on-instance deprecation in `evaluators.py:122` — both are cheap, isolated fixes surfaced by existing tooling (`uv run ruff check`, `uv run pytest -v` warnings) and should be cleared before the codebase grows further.
- **Add package markers**: no `__init__.py` exists anywhere under `backend/`; the project currently relies on implicit namespace packages plus `[tool.hatch.build.targets.wheel] packages = ["backend"]`. Decide deliberately whether to keep namespace packages or add explicit `__init__.py` files as the module count grows (adding `graph/`, `training/`, `pipeline/`, `organizational_knowledge/` next).
