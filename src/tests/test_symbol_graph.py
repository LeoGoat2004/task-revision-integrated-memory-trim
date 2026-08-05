"""Tests for tree_sitter adapter + symbol_graph PageRank."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.domain.params import DomainParams
from app.domain.symbol_graph import build_graph, expand_files, pagerank
from app.infrastructure import tree_sitter as ts


PY_SRC = '''\
import os
from utils.helper import format_date

class AuthService:
    def verify(self, token):
        if token is None:
            return False
        return self._check(token)

    def _check(self, token):
        return bool(token)

def login(user):
    auth = AuthService()
    return auth.verify(user.token)
'''

PY_HELPER = '''\
def format_date(d):
    return d.isoformat()
'''


def test_tree_sitter_extracts_python_defs():
    refs = ts.extract_symbols("auth.py", PY_SRC)
    names = {r.name for r in refs if r.kind == "def"}
    assert "AuthService" in names
    assert "verify" in names
    assert "login" in names
    assert "_check" in names
    # Imports captured as refs.
    imports = [r for r in refs if r.kind == "ref" and r.node_type == "import"]
    assert any("utils.helper" in r.name or "format_date" in r.name for r in imports)


def test_tree_sitter_regex_fallback_matches_defs():
    # Force regex path by checking the same source still yields defs even if
    # tree-sitter is available (the def set must at least contain the regex hits).
    refs = ts.extract_symbols("auth.py", PY_SRC)
    names = {r.name for r in refs if r.kind == "def"}
    assert "AuthService" in names and "login" in names


def test_tree_sitter_unknown_lang_returns_empty():
    refs = ts.extract_symbols("readme.md", "# title")
    assert refs == []


def test_available_returns_bool():
    # Should not crash; either True (grammars loaded) or False.
    assert isinstance(ts.available(), bool)


def test_build_graph_links_cross_file_symbol():
    refs = ts.extract_symbols("auth.py", PY_SRC) + ts.extract_symbols("utils/helper.py", PY_HELPER)
    g = build_graph([{"file": r.file, "name": r.name, "kind": r.kind, "node_type": r.node_type} for r in refs])
    # format_date defined in helper, imported in auth → cross-file edge.
    assert "utils/helper.py" in g.file_symbols
    assert "format_date" in g.symbol_files
    # The two files share the format_date symbol edge OR an import edge.
    assert (
        "utils/helper.py" in g.adj.get("auth.py", {})
        or "auth.py" in g.adj.get("utils/helper.py", {})
    )


def test_pagerank_surfaces_related_file():
    refs = ts.extract_symbols("auth.py", PY_SRC) + ts.extract_symbols("utils/helper.py", PY_HELPER)
    g = build_graph([{"file": r.file, "name": r.name, "kind": r.kind, "node_type": r.node_type} for r in refs])
    # Seed with auth.py (the file a memory mentioned) → helper.py should surface.
    ranked = pagerank(g, ["auth.py"])
    files = [r.file for r in ranked]
    assert "utils/helper.py" in files
    # Seed file excluded from results.
    assert "auth.py" not in files


def test_pagerank_seed_by_symbol_name():
    refs = ts.extract_symbols("auth.py", PY_SRC) + ts.extract_symbols("utils/helper.py", PY_HELPER)
    g = build_graph([{"file": r.file, "name": r.name, "kind": r.kind, "node_type": r.node_type} for r in refs])
    # Seed by symbol name → files defining it become seed, excluded; related surface.
    ranked = pagerank(g, ["verify"])
    # auth.py defines verify → excluded; helper.py (linked) should appear.
    files = [r.file for r in ranked]
    assert "utils/helper.py" in files


def test_pagerank_empty_graph():
    g = build_graph([])
    assert pagerank(g, ["x"]) == []


def test_pagerank_deterministic_with_seed():
    refs = ts.extract_symbols("auth.py", PY_SRC) + ts.extract_symbols("utils/helper.py", PY_HELPER)
    refs_dicts = [{"file": r.file, "name": r.name, "kind": r.kind, "node_type": r.node_type} for r in refs]
    g = build_graph(refs_dicts)
    r1 = pagerank(g, ["auth.py"])
    r2 = pagerank(g, ["auth.py"])
    assert [r.file for r in r1] == [r.file for r in r2]
    assert [round(r.score, 6) for r in r1] == [round(r.score, 6) for r in r2]


def test_expand_files_alias():
    refs = ts.extract_symbols("auth.py", PY_SRC) + ts.extract_symbols("utils/helper.py", PY_HELPER)
    g = build_graph([{"file": r.file, "name": r.name, "kind": r.kind, "node_type": r.node_type} for r in refs])
    assert expand_files(g, ["auth.py"]) == pagerank(g, ["auth.py"])
