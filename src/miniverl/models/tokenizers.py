"""Tokenizer adapters.

Two implementations satisfy :class:`~miniverl.agent.transcript.TokenizerLike`:

:class:`ToyTokenizer`
    A reversible, offline, ~200-entry greedy-longest-match tokenizer.  Small
    enough that ``exact_full_vocab`` losses are cheap, which is what makes the
    CPU test suite able to check the exact objective end to end.

:class:`HFTokenizerAdapter`
    A thin wrapper over a Hugging Face fast tokenizer.

Both expose a structural identity and a legacy behavioural fingerprint.  The
fingerprint hashes one fixed probe and can detect many, but not all, behavioural
differences.  Trajectories, caches and alignment maps carry tokenizer identity,
so a known mismatch is rejected instead of quietly producing garbage targets.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from miniverl.errors import BackendError, TokenizerMismatchError
from miniverl.utils.lazy import require_transformers

__all__ = [
    "ToyTokenizer",
    "HFTokenizerAdapter",
    "tokenizer_fingerprint",
    "tokenizer_structural_digest",
    "PROBE_TEXT",
    "TOY_SPECIAL_TOKENS",
]

#: Text used to fingerprint a tokenizer by behaviour rather than by metadata.
PROBE_TEXT = (
    "<|im_start|>system\nYou are a tool-using agent.<|im_end|>\n"
    "<|im_start|>user\nCompute 2*(3+4).<|im_end|>\n"
    "<|im_start|>assistant\n"
    '<tool_call>\n{"arguments": {"expression": "2*(3+4)"}, "name": "calculator"}\n</tool_call>'
    '<tool_result>\n{"ok": true, "result": "14"}\n</tool_result>'
    "<final>\n14\n</final>"
)

TOY_SPECIAL_TOKENS: tuple[str, ...] = (
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_result>",
    "</tool_result>",
    "<final>",
    "</final>",
)

_TOY_WORDS: tuple[str, ...] = (
    "answer",
    "arguments",
    "assistant",
    "avg",
    "calculator",
    "call",
    "celsius",
    "column",
    "convert",
    "count",
    "database",
    "error",
    "exactly",
    "expression",
    "fahrenheit",
    "false",
    "feet",
    "final",
    "find",
    "format",
    "from",
    "from_unit",
    "get",
    "grams",
    "integer",
    "invalid",
    "json",
    "keys",
    "kilograms",
    "kilometers",
    "limit",
    "list",
    "max",
    "meters",
    "miles",
    "min",
    "must",
    "name",
    "not",
    "null",
    "number",
    "object",
    "ok",
    "one",
    "only",
    "path",
    "please",
    "pounds",
    "query",
    "respond",
    "result",
    "row",
    "rows",
    "schema",
    "select",
    "sql",
    "step",
    "steps",
    "string",
    "sum",
    "system",
    "table",
    "the",
    "then",
    "this",
    "to_unit",
    "tool",
    "tools",
    "true",
    "turn",
    "unknown",
    "use",
    "user",
    "valid",
    "value",
    "where",
    "with",
    "you",
    "your",
)


def tokenizer_fingerprint(payload: dict[str, Any]) -> str:
    """Stable hex digest of a JSON-serializable tokenizer description."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def tokenizer_structural_digest(tokenizer: Any) -> str:
    """Incrementally hash tokenizer structure without exposing local file paths."""
    digest = hashlib.sha256()

    def update(name: str, value: Any) -> None:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        digest.update(b"\0")

    get_vocab = getattr(tokenizer, "get_vocab", None)
    vocab = get_vocab() if callable(get_vocab) else {}
    update("vocab", sorted((str(token), int(token_id)) for token, token_id in vocab.items()))
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    added = get_added_vocab() if callable(get_added_vocab) else {}
    update("added_vocab", sorted((str(token), int(token_id)) for token, token_id in added.items()))
    update("special_tokens_map", getattr(tokenizer, "special_tokens_map", {}))
    update("tokenizer_class", type(tokenizer).__name__)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_to_str = getattr(backend, "to_str", None)
    backend_description = backend_to_str() if callable(backend_to_str) else None
    if isinstance(backend_description, str):
        try:
            parsed_backend = json.loads(backend_description)
        except json.JSONDecodeError:
            parsed_backend = backend_description
        backend_description = parsed_backend
    update("backend_tokenizer", _canonicalize_tokenizer_structure(backend_description))
    update(
        "tokenizer_config",
        _canonicalize_tokenizer_structure(getattr(tokenizer, "init_kwargs", {})),
    )
    return digest.hexdigest()


_NON_STRUCTURAL_TOKENIZER_KEYS = frozenset(
    {
        "added_tokens_file",
        "cache_dir",
        "chat_template_file",
        "merges_file",
        "name_or_path",
        "special_tokens_map_file",
        "tokenizer_config_file",
        "tokenizer_file",
        "vocab_file",
    }
)


def _canonicalize_tokenizer_structure(value: Any) -> Any:
    """Remove source-location metadata while preserving tokenizer behaviour."""
    if isinstance(value, dict):
        return {
            str(key): _canonicalize_tokenizer_structure(item)
            for key, item in value.items()
            if str(key) not in _NON_STRUCTURAL_TOKENIZER_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_tokenizer_structure(item) for item in value]
    if isinstance(value, set):
        return sorted(_canonicalize_tokenizer_structure(item) for item in value)
    return value


class ToyTokenizer:
    """Reversible greedy-longest-match tokenizer over a tiny fixed vocabulary.

    Guarantees, all covered by tests:

    * ``decode(encode(text)) == text`` for any supported text.
    * Concatenating segment encodings equals encoding the concatenation
      (there are no cross-boundary merges by construction).
    * The vocabulary and therefore the fingerprint are deterministic.

    Only printable ASCII plus ``\\n`` and ``\\t`` are supported.  Anything else
    raises instead of silently substituting an unknown token, because a lossy
    toy tokenizer would break the provenance guarantees the whole project rests
    on.
    """

    def __init__(self) -> None:
        chars = [chr(c) for c in range(32, 127)] + ["\n", "\t", "\r"]
        vocab: list[str] = list(TOY_SPECIAL_TOKENS)
        vocab += [w for w in sorted(set(_TOY_WORDS)) if len(w) > 1]
        vocab += sorted(set(chars))
        # Deduplicate while preserving order.
        seen: set[str] = set()
        self._vocab: list[str] = []
        for piece in vocab:
            if piece not in seen:
                seen.add(piece)
                self._vocab.append(piece)
        self._ids: dict[str, int] = {piece: i for i, piece in enumerate(self._vocab)}
        # Longest first so greedy matching prefers specials, then words, then chars.
        self._ordered: list[str] = sorted(self._vocab, key=len, reverse=True)
        self._by_first: dict[str, list[str]] = {}
        for piece in self._ordered:
            self._by_first.setdefault(piece[0], []).append(piece)

        self.eos_token_id = self._ids["<|im_end|>"]
        self.pad_token_id = self._ids["<|endoftext|>"]
        self.bos_token_id: int | None = None
        self.fingerprint = tokenizer_fingerprint(
            {"kind": "toy", "version": 1, "vocab": self._vocab}
        )
        self.identity = {
            "behavioral_fingerprint_v1": self.fingerprint,
            "structural_digest_v2": tokenizer_fingerprint(
                {"kind": "toy", "version": 2, "vocab": self._vocab}
            ),
            "tokenizer_class": type(self).__name__,
            "length": len(self._vocab),
            "base_vocab_size": len(self._vocab),
            "added_vocab_size": 0,
        }

    # -- TokenizerLike --------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Number of distinct tokens."""
        return len(self._vocab)

    def encode(self, text: str) -> list[int]:
        """Greedy longest-match tokenization."""
        ids: list[int] = []
        i = 0
        n = len(text)
        while i < n:
            candidates = self._by_first.get(text[i])
            if not candidates:
                raise BackendError(
                    f"character {text[i]!r} (U+{ord(text[i]):04X}) is outside the toy "
                    "tokenizer's ASCII vocabulary",
                    hint="the toy backend only handles printable ASCII; use the 'hf' "
                    "backend for arbitrary text",
                )
            for piece in candidates:
                if text.startswith(piece, i):
                    ids.append(self._ids[piece])
                    i += len(piece)
                    break
            else:  # pragma: no cover - single chars always match
                raise BackendError(f"no toy token matches text at offset {i}: {text[i : i + 16]!r}")
        return ids

    def decode(self, token_ids: Sequence[int]) -> str:
        """Concatenate the vocabulary pieces."""
        out: list[str] = []
        for tid in token_ids:
            if not 0 <= int(tid) < len(self._vocab):
                raise BackendError(f"token id {tid} is outside the toy vocabulary")
            out.append(self._vocab[int(tid)])
        return "".join(out)

    def token_piece(self, token_id: int) -> str:
        """Human-readable piece for one id (used by reports)."""
        return self._vocab[int(token_id)]

    @property
    def vocab(self) -> list[str]:
        """Copy of the ordered vocabulary."""
        return list(self._vocab)


class HFTokenizerAdapter:
    """Adapter over a Hugging Face fast tokenizer."""

    def __init__(self, tokenizer: Any, model_id: str, revision: str | None = None) -> None:
        self._tok = tokenizer
        self.model_id = model_id
        self.revision = revision
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is None:
            raise BackendError(
                f"tokenizer for {model_id} has no eos_token_id",
                hint="miniVERL needs an EOS token to bound generation",
            )
        self.eos_token_id = int(eos)
        pad = getattr(tokenizer, "pad_token_id", None)
        self.pad_token_id = int(pad) if pad is not None else self.eos_token_id
        bos = getattr(tokenizer, "bos_token_id", None)
        self.bos_token_id = int(bos) if bos is not None else None
        probe_ids = self.encode(PROBE_TEXT)
        self.fingerprint = tokenizer_fingerprint(
            {
                "kind": "hf",
                "class": type(tokenizer).__name__,
                "vocab_size": len(tokenizer),
                "eos": self.eos_token_id,
                "pad": self.pad_token_id,
                "added": sorted(
                    str(t) for t in getattr(tokenizer, "additional_special_tokens", [])
                ),
                "probe": probe_ids,
            }
        )
        get_vocab = getattr(tokenizer, "get_vocab", None)
        get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
        base_vocab = get_vocab() if callable(get_vocab) else {}
        added_vocab = get_added_vocab() if callable(get_added_vocab) else {}
        self.identity = {
            "behavioral_fingerprint_v1": self.fingerprint,
            "structural_digest_v2": tokenizer_structural_digest(tokenizer),
            "tokenizer_class": type(tokenizer).__name__,
            "length": len(tokenizer),
            "base_vocab_size": len(base_vocab),
            "added_vocab_size": len(added_vocab),
        }

    @classmethod
    def load(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
    ) -> HFTokenizerAdapter:
        """Load a tokenizer from a local path or the Hub."""
        transformers = require_transformers("Loading a Hugging Face tokenizer")
        try:
            tok = transformers.AutoTokenizer.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                use_fast=True,
            )
        except OSError as exc:
            revision_text = f" at revision {revision!r}" if revision else ""
            preload = f"hf download {model_id}"
            if revision:
                preload += f" --revision {revision}"
            offline_hint = (
                f"offline mode found no complete cached tokenizer snapshot; preload it online "
                f"with `{preload}`. "
                if local_files_only
                else ""
            )
            raise BackendError(
                f"could not load tokenizer {model_id!r}{revision_text}",
                hint=offline_hint + f"check the tokenizer id and revision. Original error: {exc}",
            ) from exc
        return cls(tok, model_id=model_id, revision=revision)

    @property
    def vocab_size(self) -> int:
        """Length of the tokenizer including added tokens."""
        return len(self._tok)

    def encode(self, text: str) -> list[int]:
        """Tokenize without implicit special tokens."""
        return list(self._tok(text, add_special_tokens=False)["input_ids"])

    def decode(self, token_ids: Sequence[int]) -> str:
        """Detokenize, keeping special tokens visible."""
        return str(self._tok.decode(list(token_ids), skip_special_tokens=False))

    def token_piece(self, token_id: int) -> str:
        """Human-readable piece for one id (used by reports)."""
        return str(self._tok.convert_ids_to_tokens(int(token_id)))

    @property
    def raw(self) -> Any:
        """The underlying Hugging Face tokenizer."""
        return self._tok


def assert_same_tokenizer(student: Any, teacher: Any) -> None:
    """Raise unless structural identity, or the legacy fallback, matches."""
    student_identity = getattr(student, "identity", {})
    teacher_identity = getattr(teacher, "identity", {})
    student_structural = student_identity.get("structural_digest_v2")
    teacher_structural = teacher_identity.get("structural_digest_v2")
    same = (
        student_structural == teacher_structural
        if student_structural and teacher_structural
        else student.fingerprint == teacher.fingerprint
    )
    if not same:
        raise TokenizerMismatchError(
            "student and teacher tokenizers are not identical "
            f"({student.fingerprint[:12]}... vs {teacher.fingerprint[:12]}...)",
            hint="miniVERL currently supports same-tokenizer distillation only; choose a "
            "teacher from the same model family (see docs/limitations.md)",
        )
