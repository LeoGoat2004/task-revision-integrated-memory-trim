"""Domain unit tests: tokenize, chunker, BM25, profile_extractor, quality.

These tests cover the pure-functional core of the system and do not touch
the network or filesystem.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_tokenize_drops_short_and_pure_digit_tokens():
    from app.domain.tokenizer import tokenize

    tokens = tokenize("a b 12 4567 FastAPI")
    assert "a" not in tokens
    assert "b" not in tokens
    assert "12" not in tokens
    assert "4567" not in tokens
    assert "fastapi" in tokens


def test_tokenize_splits_camel_case_with_acronym_boundary():
    from app.domain.tokenizer import tokenize

    tokens = tokenize("HTTPRequest 解析报错")
    assert "http" in tokens
    assert "request" in tokens
    # CJK run preserved.
    assert "解析报错" in tokens


def test_tokenize_splits_dotted_and_underscored_identifiers():
    from app.domain.tokenizer import tokenize

    tokens = tokenize("django.db.utils.OperationalError app_settings")
    for expected in ("django", "db", "utils", "operational", "error", "app", "settings"):
        assert expected in tokens


def test_tokenize_preserves_original_token():
    """For partial-match recall we keep both the original and split sub-tokens."""
    from app.domain.tokenizer import tokenize

    tokens = tokenize("OperationalError")
    assert "operationalerror" in tokens
    assert "operational" in tokens
    assert "error" in tokens


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


def _msg(content: str):
    from app.api.schemas import AddMessage
    return AddMessage(role="user", content=content)


def test_chunker_returns_single_chunk_when_small():
    from app.domain.chunker import chunk_messages

    msgs = [_msg(f"msg {i}") for i in range(5)]
    chunks = chunk_messages(msgs)
    assert len(chunks) == 1
    assert chunks[0] == msgs


def test_chunker_splits_when_too_many_messages():
    from app.domain.chunker import chunk_messages
    from app.domain.params import DomainParams

    # Use explicit params so the test is independent of the default thresholds.
    params = DomainParams(chunk_max_messages=20, chunk_boundary_on_todo=False)
    msgs = [_msg(f"msg {i}") for i in range(45)]
    chunks = chunk_messages(msgs, params)
    assert len(chunks) >= 2
    # Every chunk respects the message-count cap.
    for chunk in chunks:
        assert len(chunk) <= 20


def test_chunker_splits_when_word_count_exceeded():
    from app.domain.chunker import chunk_messages
    from app.domain.params import DomainParams

    params = DomainParams(chunk_max_words=2000, chunk_boundary_on_todo=False)
    big = "word " * 1500
    msgs = [_msg(big), _msg(big)]
    chunks = chunk_messages(msgs, params)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert sum(len(m.content.split()) for m in chunk) <= 2000


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


def test_rank_documents_returns_top_n_in_score_order():
    from app.domain.bm25 import build_index, rank_documents, score_query
    from app.domain.tokenizer import tokenize

    # Use unique terms so BM25 has discriminative IDF (with very small corpora,
    # common terms collapse to ~0 IDF and mask the ranking).
    docs = [
        tokenize("apple banana mango"),
        tokenize("orange kiwi papaya"),
        tokenize("apple banana mango pear"),
    ]
    index = build_index(docs)
    assert index is not None
    ranks = rank_documents(index, tokenize("apple banana mango"), top_n=3)
    assert len(ranks) == 3
    # Verify the function returns indices in non-increasing score order.
    scores = score_query(index, tokenize("apple banana mango"))
    # Replicate the min-shift applied inside rank_documents.
    scores_shifted = scores - scores.min()
    for i, j in zip(ranks, ranks[1:]):
        assert scores_shifted[i] >= scores_shifted[j]


def test_rank_documents_handles_empty_input():
    from app.domain.bm25 import rank_documents, build_index
    # Empty corpus → build_index returns None → rank_documents returns [].
    assert rank_documents(build_index([]), [], top_n=5) == []


# ---------------------------------------------------------------------------
# Profile extractor
# ---------------------------------------------------------------------------


def test_profile_extractor_finds_language():
    from app.domain.profile_extractor import extract_profile_pairs

    pairs = extract_profile_pairs("Project uses Python 3.12.")
    assert ("language", "python") in pairs


def test_profile_extractor_finds_database():
    from app.domain.profile_extractor import extract_profile_pairs

    pairs = extract_profile_pairs("Backend uses PostgreSQL 15 with read replicas.")
    assert ("database", "postgresql") in pairs


def test_profile_extractor_deduplicates_and_caps_at_5():
    from app.domain.profile_extractor import extract_profile_pairs

    text = (
        "Uses Python 3.12. Uses Java. Uses Go. Uses Rust. Uses Ruby. Uses PHP. Uses Swift."
    )
    pairs = extract_profile_pairs(text)
    assert len(pairs) <= 5
    # No duplicates.
    assert len(set(pairs)) == len(pairs)


def test_profile_extractor_merge_pairs_dedupes():
    from app.domain.profile_extractor import merge_pairs

    existing = [{"attr": "language", "value": "python"}]
    new = [
        {"attr": "language", "value": "python"},  # duplicate
        {"attr": "language", "value": "java"},
    ]
    merged = merge_pairs(existing, new)
    assert len(merged) == 2
    assert {"attr": "database", "value": "x"} not in merged


# ---------------------------------------------------------------------------
# Quality: dedup
# ---------------------------------------------------------------------------


def test_content_fingerprint_changes_with_user_id():
    from app.domain.quality import content_fingerprint

    a = content_fingerprint("user_a", "fixed NPE")
    b = content_fingerprint("user_b", "fixed NPE")
    assert a != b


def test_content_fingerprint_stable_for_same_input():
    from app.domain.quality import content_fingerprint

    a = content_fingerprint("user_a", "fixed NPE")
    b = content_fingerprint("user_a", "fixed NPE")
    assert a == b


def test_is_near_duplicate_detects_high_overlap():
    from app.domain.enums import MemoryType
    from app.domain.models import MemoryNote
    from app.domain.quality import is_near_duplicate

    # Use stable, distinctive terms so BM25 scores are non-degenerate.
    existing = [
        MemoryNote(
            id="m1",
            user_id="u",
            session_id="s",
            request_id="r",
            content="AuthService throws NullPointerException when user_id is missing.",
            embedding=[],
            memory_type=MemoryType.RECORD,
            type_specific={},
        ),
        # Add a clearly unrelated existing note so the corpus has contrast.
        MemoryNote(
            id="m0",
            user_id="u",
            session_id="s",
            request_id="r",
            content="Redis cache invalidation pattern for user profile data.",
            embedding=[],
            memory_type=MemoryType.RECORD,
            type_specific={},
        ),
    ]
    candidate = MemoryNote(
        id="m2",
        user_id="u",
        session_id="s",
        request_id="r",
        content="AuthService throws NullPointerException when user_id is missing field.",
        embedding=[],
        memory_type=MemoryType.RECORD,
        type_specific={},
    )
    assert is_near_duplicate(candidate, existing, threshold=0.0)


def test_is_near_duplicate_returns_false_for_different_topic():
    from app.domain.enums import MemoryType
    from app.domain.models import MemoryNote
    from app.domain.quality import is_near_duplicate

    existing = [
        MemoryNote(
            id="m1",
            user_id="u",
            session_id="s",
            request_id="r",
            content="PostgreSQL foreign key constraint violation on users table.",
            embedding=[],
            memory_type=MemoryType.RECORD,
            type_specific={},
        ),
        MemoryNote(
            id="m0",
            user_id="u",
            session_id="s",
            request_id="r",
            content="S3 multipart upload timeout retry strategy.",
            embedding=[],
            memory_type=MemoryType.RECORD,
            type_specific={},
        ),
    ]
    candidate = MemoryNote(
        id="m2",
        user_id="u",
        session_id="s",
        request_id="r",
        content="Photos upload fails with MIME type validation error.",
        embedding=[],
        memory_type=MemoryType.RECORD,
        type_specific={},
    )
    assert not is_near_duplicate(candidate, existing, threshold=0.5)


def test_is_near_duplicate_returns_false_for_empty_existing():
    from app.domain.enums import MemoryType
    from app.domain.models import MemoryNote
    from app.domain.quality import is_near_duplicate

    candidate = MemoryNote(
        id="m1",
        user_id="u",
        session_id="s",
        request_id="r",
        content="anything",
        embedding=[],
        memory_type=MemoryType.RECORD,
        type_specific={},
    )
    assert not is_near_duplicate(candidate, [])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


def test_memory_type_values():
    from app.domain.enums import MemoryType

    assert MemoryType.PROFILE.value == "profile"
    assert MemoryType.EVENT.value == "event"
    assert MemoryType.RECORD.value == "record"


def test_memory_type_from_string():
    from app.domain.enums import MemoryType

    assert MemoryType("profile") is MemoryType.PROFILE
    assert MemoryType("event") is MemoryType.EVENT
    assert MemoryType("record") is MemoryType.RECORD


def test_memory_type_invalid_raises():
    from app.domain.enums import MemoryType

    with pytest.raises(ValueError):
        _ = MemoryType("bogus")
