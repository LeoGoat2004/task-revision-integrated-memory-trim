"""tree-sitter adapter: extract symbol definitions + references from source.

Multi-language (python/js/ts/java/go/c/cpp). Used by `domain.symbol_graph` to
build the def→ref graph that powers Aider-style personalized PageRank for
cross-file localization.

Degradation: if tree-sitter or a language grammar is unavailable, or parsing
fails, the adapter falls back to a regex-based extractor (definitions +
imports) so the pipeline never crashes. `available()` reflects whether the
real parser loaded.

The adapter is the ONLY place that imports tree-sitter; the domain layer sees
only `SymbolRef` dataclasses.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SymbolRef:
    """A symbol occurrence extracted from source."""

    file: str
    name: str
    kind: str  # "def" | "ref"
    line: int
    lang: str
    node_type: str = ""  # e.g. function_definition, class_declaration
    container: str = ""  # enclosing class/module for methods


# ---------------------------------------------------------------------------
# Language registry
# ---------------------------------------------------------------------------

_EXT_LANG = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".go": "go",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
}

# Node types that define a named symbol, per language.
_DEF_NODE_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition"},
    "typescript": {"function_declaration", "class_declaration", "method_definition"},
    "java": {"method_declaration", "class_declaration", "interface_declaration"},
    "go": {"function_declaration", "type_declaration", "method_declaration"},
    "c": {"function_definition", "struct_specifier"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
}

# Map a def node type to a coarse symbol category.
_DEF_CATEGORY = {
    "function_definition": "function",
    "function_declaration": "function",
    "method_definition": "method",
    "method_declaration": "method",
    "class_definition": "class",
    "class_declaration": "class",
    "class_specifier": "class",
    "struct_specifier": "struct",
    "interface_declaration": "interface",
    "type_declaration": "type",
}


# ---------------------------------------------------------------------------
# Parser registry (lazy)
# ---------------------------------------------------------------------------

_parsers: dict[str, Any] = {}
_registry_ok: bool | None = None


def _load_registry() -> bool:
    """Try to load all available language parsers. Returns True if at least one loaded."""
    global _registry_ok
    if _registry_ok is not None:
        return _registry_ok
    try:
        from tree_sitter import Language, Parser
    except Exception as exc:  # noqa: BLE001
        logger.warning("tree-sitter unavailable, falling back to regex: %s", exc)
        _registry_ok = False
        return False

    loaders = {
        "python": "_language" if False else None,  # placeholder, filled below
    }
    # Each tree-sitter-<lang> package exposes a `language()` callable.
    try:
        import tree_sitter_python as tsp
        _parsers["python"] = _build(Parser, Language, tsp.language())
    except Exception as exc:  # noqa: BLE001
        logger.debug("python grammar not loaded: %s", exc)
    for mod_name, lang_key in (
        ("tree_sitter_javascript", "javascript"),
        ("tree_sitter_typescript", "typescript"),
        ("tree_sitter_java", "java"),
        ("tree_sitter_go", "go"),
        ("tree_sitter_c", "c"),
        ("tree_sitter_cpp", "cpp"),
    ):
        try:
            mod = __import__(mod_name)
            # typescript package exposes .language_typescript() / .language_tsx()
            if mod_name == "tree_sitter_typescript":
                lang_fn = getattr(mod, "language_typescript", None) or mod.language_typescript
                _parsers[lang_key] = _build(Parser, Language, lang_fn())
            else:
                lang_fn = getattr(mod, "language", None)
                _parsers[lang_key] = _build(Parser, Language, lang_fn())
        except Exception as exc:  # noqa: BLE001
            logger.debug("%s grammar not loaded: %s", mod_name, exc)

    _registry_ok = bool(_parsers)
    if _registry_ok:
        logger.info("tree-sitter loaded languages: %s", list(_parsers))
    else:
        logger.warning("tree-sitter loaded no grammars; using regex fallback")
    return _registry_ok


def _build(ParserCls, LanguageCls, lang_ptr) -> Any:
    """Build a Parser for a language pointer (tree-sitter 0.26 API)."""
    lang = LanguageCls(lang_ptr)
    return ParserCls(lang)


def available() -> bool:
    """Whether the real tree-sitter parser loaded at least one language."""
    return _load_registry()


def lang_for_file(file: str) -> str | None:
    ext = Path(file).suffix.lower()
    return _EXT_LANG.get(ext)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _name_of(node: Any) -> str:
    """The declared name of a definition node (the `name` field child)."""
    try:
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return name_node.text.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    # Fallback: first identifier child.
    try:
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    return ""


def _walk_defs(node: Any, def_types: set[str], file: str, lang: str) -> list[SymbolRef]:
    refs: list[SymbolRef] = []
    stack = [node]
    while stack:
        n = stack.pop()
        try:
            ntype = n.type
        except Exception:  # noqa: BLE001
            continue
        if ntype in def_types:
            name = _name_of(n)
            if name:
                refs.append(SymbolRef(
                    file=file, name=name, kind="def",
                    line=(n.start_point[0] + 1) if hasattr(n, "start_point") else 0,
                    lang=lang, node_type=ntype,
                ))
        try:
            children = n.children
        except Exception:  # noqa: BLE001
            children = []
        for c in children:
            stack.append(c)
    return refs


def _extract_imports_regex(content: str, lang: str) -> list[str]:
    """Cheap import-target extraction for the reference graph."""
    targets: list[str] = []
    patterns: list[re.Pattern[str]] = []
    if lang == "python":
        patterns.append(re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.MULTILINE))
    elif lang in ("javascript", "typescript"):
        patterns.append(re.compile(r"""(?:import\s+.*?\s+from\s+['"]([^'"]+)['"]|require\(\s*['"]([^'"]+)['"]\s*\))"""))
    elif lang == "java":
        patterns.append(re.compile(r"^\s*import\s+([\w.]+);", re.MULTILINE))
    elif lang == "go":
        patterns.append(re.compile(r"""^\s*import\s+(?:\(\s*)?['"]([^'"]+)['"]""", re.MULTILINE))
    for pat in patterns:
        for m in pat.finditer(content):
            for g in m.groups():
                if g:
                    targets.append(g)
    return targets


def _regex_defs(content: str, lang: str, file: str) -> list[SymbolRef]:
    """Regex fallback definition extractor."""
    refs: list[SymbolRef] = []
    patterns: list[tuple[str, re.Pattern[str]]] = []
    if lang == "python":
        patterns = [
            ("function_definition", re.compile(r"^\s*def\s+(\w+)\s*\(", re.MULTILINE)),
            ("class_definition", re.compile(r"^\s*class\s+(\w+)\b", re.MULTILINE)),
        ]
    elif lang in ("javascript", "typescript"):
        patterns = [
            ("function_declaration", re.compile(r"\bfunction\s+(\w+)\s*\(")),
            ("class_declaration", re.compile(r"\bclass\s+(\w+)\b")),
            ("method_definition", re.compile(r"^\s*(\w+)\s*\([^)]*\)\s*\{", re.MULTILINE)),
        ]
    elif lang == "java":
        patterns = [
            ("class_declaration", re.compile(r"\bclass\s+(\w+)\b")),
            ("method_declaration", re.compile(r"\b(?:public|private|protected|static|\s)+\s+\w+\s+(\w+)\s*\([^)]*\)\s*\{")),
        ]
    elif lang == "go":
        patterns = [
            ("function_declaration", re.compile(r"^func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(", re.MULTILINE)),
            ("type_declaration", re.compile(r"^type\s+(\w+)\b", re.MULTILINE)),
        ]
    for ntype, pat in patterns:
        for m in pat.finditer(content):
            line = content.count("\n", 0, m.start()) + 1
            refs.append(SymbolRef(
                file=file, name=m.group(1), kind="def",
                line=line, lang=lang, node_type=ntype,
            ))
    return refs


def extract_symbols(file: str, content: str) -> list[SymbolRef]:
    """Extract def + ref SymbolRefs from a source file.

    Uses tree-sitter when available for the file's language; otherwise the
    regex fallback. Imports are always extracted via regex (cheap and uniform).
    """
    lang = lang_for_file(file)
    if lang is None:
        return []
    refs: list[SymbolRef] = []

    if _load_registry():
        parser = _parsers.get(lang)
        if parser is not None:
            try:
                tree = parser.parse(content.encode("utf-8"))
                refs = _walk_defs(tree.root_node, _DEF_NODE_TYPES.get(lang, set()), file, lang)
            except Exception as exc:  # noqa: BLE001
                logger.debug("tree-sitter parse failed for %s: %s", file, exc)
                refs = _regex_defs(content, lang, file)
        else:
            refs = _regex_defs(content, lang, file)
    else:
        refs = _regex_defs(content, lang, file)

    # Add imports as reference edges (file → imported module/symbol).
    for tgt in _extract_imports_regex(content, lang):
        refs.append(SymbolRef(
            file=file, name=tgt, kind="ref", line=0, lang=lang, node_type="import",
        ))
    return refs


def extract_symbols_from_files(files: dict[str, str]) -> list[SymbolRef]:
    """Convenience: extract across {file_path: content} mapping."""
    out: list[SymbolRef] = []
    for f, content in files.items():
        out.extend(extract_symbols(f, content))
    return out
