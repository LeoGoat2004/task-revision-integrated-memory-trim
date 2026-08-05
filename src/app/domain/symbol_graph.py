"""Symbol graph + personalized PageRank (Aider repomap style).

Builds a symbol/file co-occurrence graph from `SymbolRef`s extracted by the
tree-sitter adapter, then runs personalized PageRank seeded by a retrieved
memory's `files_symbols` to surface RELATED files for cross-file localization
(the SWEContextBench hard cases that pure vector retrieval misses).

Edges (undirected, weighted):
  1. Same-file cohesion: defs in the same file are connected (they form a unit).
  2. Cross-file: a symbol name defined in file A and referenced (imported) in
     file B connects A and B.
  3. Import edges: a file's import targets connect to files defining those names.

Personalized PageRank: seed nodes (from the card's files_symbols) get an
`edited_boost` multiplier on their initial weight (Aider uses ×50 for edited
files). Returns ranked files for retrieval expansion.

Pure domain: takes SymbolRefs (plain dicts) so it never imports tree-sitter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .params import DomainParams, DEFAULT_PARAMS, RANDOM_SEED


@dataclass
class SymbolGraph:
    """Adjacency graph over files (and the symbols they define)."""

    # file → set of symbol names defined there
    file_symbols: dict[str, set[str]] = field(default_factory=dict)
    # symbol name → set of files defining it (for cross-file edges)
    symbol_files: dict[str, set[str]] = field(default_factory=dict)
    # file → set of files it imports/depends on
    file_imports: dict[str, set[str]] = field(default_factory=dict)
    # adjacency: file → {neighbour_file: weight}
    adj: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def files(self) -> list[str]:
        return list(self.file_symbols.keys())


def build_graph(refs: Iterable[dict]) -> SymbolGraph:
    """Build a SymbolGraph from SymbolRef-like dicts.

    Each ref dict has: file, name, kind ("def"|"ref"), node_type (optional).
    """
    g = SymbolGraph()
    for r in refs:
        file = r.get("file", "")
        name = r.get("name", "")
        kind = r.get("kind", "")
        if not file or not name:
            continue
        if kind == "def":
            g.file_symbols.setdefault(file, set()).add(name)
            g.symbol_files.setdefault(name, set()).add(file)
        elif kind == "ref" and r.get("node_type") == "import":
            # import target — defer edge creation until we can map to a file.
            g.file_imports.setdefault(file, set()).add(name)

    # Build adjacency.
    # 1. Same-file cohesion is implicit (a file is one node).
    # 2. Cross-file: symbol defined in multiple files → connect those files.
    for name, files in g.symbol_files.items():
        flist = list(files)
        for i in range(len(flist)):
            for j in range(i + 1, len(flist)):
                _add_edge(g.adj, flist[i], flist[j], 1.0)
    # 3. Import edges: file F imports module M; connect F to files whose path
    #    or defined symbols relate to M (best-effort suffix match).
    for importer, targets in g.file_imports.items():
        for tgt in targets:
            # Match by tail of the import path against file paths.
            tgt_tail = tgt.split(".")[-1].lower()
            for candidate in g.file_symbols:
                if candidate == importer:
                    continue
                cand_tail = candidate.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                cand_stem = cand_tail.rsplit(".", 1)[0].lower()
                if cand_stem and (cand_stem == tgt_tail or tgt_tail in cand_stem or cand_stem in tgt_tail):
                    _add_edge(g.adj, importer, candidate, 0.5)
    return g


def _add_edge(adj: dict[str, dict[str, float]], a: str, b: str, w: float) -> None:
    if a == b:
        return
    adj.setdefault(a, {})[b] = adj.setdefault(a, {}).get(b, 0.0) + w
    adj.setdefault(b, {})[a] = adj.setdefault(b, {}).get(a, 0.0) + w


@dataclass
class RankedFile:
    file: str
    score: float
    symbols: list[str] = field(default_factory=list)


def pagerank(
    graph: SymbolGraph,
    seed: Iterable[str],
    params: DomainParams = DEFAULT_PARAMS,
) -> list[RankedFile]:
    """Personalized PageRank over the file graph, seeded by `seed`.

    `seed` is a collection of file paths and/or symbol names (typically the
    `files_symbols` of a retrieved memory's card). Seed files get an
    `edited_boost` multiplier on their initial weight (Aider ×50). Returns
    files ranked by PageRank score, excluding the seed files themselves when
    `pagerank_top_n` limits the result.

    Deterministic: uses RANDOM_SEED for the reset distribution so runs are
    reproducible.
    """
    import random
    rng = random.Random(RANDOM_SEED)

    files = graph.files
    if not files:
        return []
    n = len(files)
    idx = {f: i for i, f in enumerate(files)}

    # Build seed weight vector.
    seed_set_files: set[str] = set()
    for s in seed:
        if not s:
            continue
        if s in idx:
            seed_set_files.add(s)
        else:
            # Treat as a symbol name → seed the files that define it.
            for f in graph.symbol_files.get(s, set()):
                seed_set_files.add(f)

    personalization = [0.0] * n
    for f in seed_set_files:
        personalization[idx[f]] = params.pagerank_edited_boost
    total_p = sum(personalization)
    if total_p <= 0:
        # No seed matched — uniform personalization (degraded but functional).
        personalization = [1.0] * n
        total_p = float(n)
    personalization = [p / total_p for p in personalization]

    # Build transition matrix rows (sparse).
    out_weight = [0.0] * n
    transitions: list[dict[int, float]] = [dict() for _ in range(n)]
    for f, neighbours in graph.adj.items():
        i = idx[f]
        row_sum = 0.0
        for nb, w in neighbours.items():
            j = idx.get(nb)
            if j is not None:
                transitions[i][j] = w
                row_sum += w
        out_weight[i] = row_sum

    # Dangling nodes (no out-edges) redistribute uniformly.
    for i in range(n):
        if out_weight[i] <= 0:
            transitions[i] = {j: 1.0 / n for j in range(n) if j != i}
            out_weight[i] = (n - 1) / n

    # Power iteration (standard personalized PageRank).
    rank = [1.0 / n] * n
    damping = params.pagerank_damping
    for _ in range(params.pagerank_iterations):
        new_rank = [(1.0 - damping) * personalization[i] for i in range(n)]
        for i in range(n):
            if out_weight[i] <= 0:
                continue
            share = damping * rank[i] / out_weight[i]
            for j, w in transitions[i].items():
                new_rank[j] += share * w
        # Normalize (guard against drift).
        s = sum(new_rank)
        if s > 0:
            new_rank = [r / s for r in new_rank]
        rank = new_rank

    # Rank files, exclude pure seed files from the expansion result.
    ranked = []
    for f in files:
        score = rank[idx[f]]
        if f in seed_set_files:
            continue
        ranked.append(RankedFile(
            file=f, score=score,
            symbols=sorted(graph.file_symbols.get(f, set())),
        ))
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[: params.pagerank_top_n]


def expand_files(
    graph: SymbolGraph,
    seed: Iterable[str],
    params: DomainParams = DEFAULT_PARAMS,
) -> list[RankedFile]:
    """Convenience alias for `pagerank` — the retrieval-expansion entry point."""
    return pagerank(graph, seed, params)
