"""AlphaEdit delta storage and automatic probe-score-select routing.

route_and_run probes the latest delta of every stored session independently and
generates the final answer with the highest-scoring session. Shared-model calls
are serialized. Explicit binding helpers remain for internal use and regression
tests; incoming questions do not require a session ID.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterable

import torch

def apply_AlphaEdit_to_model(*args, **kwargs):
    # Reading/probing saved deltas does not require the AlphaEdit write package.
    from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model as write
    return write(*args, **kwargs)


TensorMap = Dict[str, torch.Tensor]


@dataclass(frozen=True)
class ProbeScore:
    session_id: str
    version: int
    candidate: str
    score: float


@dataclass(frozen=True)
class RoutedAnswer:
    selected: ProbeScore
    probes: tuple[ProbeScore, ...]
    answer: str
    tied: bool


class SessionFFNMemory:
    """Store one AlphaEdit FFN delta per chat session."""

    def __init__(self, model, weight_names: Iterable[str], root: str | Path):
        self.model = model
        self.weight_names = tuple(weight_names)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._active_session: str | None = None

        # Keep one immutable reference copy. Session files contain only ΔW.
        self._base: TensorMap = {
            name: model.get_parameter(name).detach().cpu().clone()
            for name in self.weight_names
        }

    def _restore_base(self) -> None:
        """Reset every edited FFN matrix to W_base."""
        with torch.no_grad():
            for name, base in self._base.items():
                parameter = self.model.get_parameter(name)
                parameter.copy_(base.to(parameter.device, parameter.dtype))
        self._active_session = None

    def write_session(
        self,
        session_id: str,
        tok,
        requests,
        hparams,
        cache_c,
        projection,
        version: int,
        parent_version: int | None = None,
    ):
        """Append an AlphaEdit write to a session and save the resulting total ΔW.

        `parent_version=None` starts a new session from W_base. To add another
        turn without forgetting earlier turns, pass the previous session version.
        """
        with self._lock:
            self._restore_base()
            try:
                if parent_version is not None:
                    self._apply_delta(self._load(session_id, parent_version))
                self.model, updated_cache = apply_AlphaEdit_to_model(
                    self.model,
                    tok,
                    requests,
                    hparams,
                    cache_c=cache_c,
                    P=projection,
                )
                delta = {
                    name: (
                        self.model.get_parameter(name).detach().cpu()
                        - self._base[name]
                    )
                    for name in self.weight_names
                }
                self._save(session_id, version, delta, parent_version)
                return updated_cache
            finally:
                # The shared model never retains a session after writing.
                self._restore_base()

    def activate(self, session_id: str, version: int) -> None:
        """Atomically bind W_base + ΔW(session) for a single-worker demo."""
        delta = self._load(session_id, version)
        with self._lock, torch.no_grad():
            self._restore_base()
            self._apply_delta(delta)
            self._active_session = session_id

    @contextmanager
    def session_scope(self, session_id: str, version: int):
        """Keep one session bound for the entire inference call.

        Holding the lock through inference prevents another request from swapping
        the shared model's weights midway through generation.
        """
        with self._lock:
            try:
                self.activate(session_id, version)
                yield self.model
            finally:
                self._restore_base()

    def run(self, session_id: str, version: int, inference: Callable, *args, **kwargs):
        """Run one inference atomically under the selected session memory."""
        with self.session_scope(session_id, version) as model:
            return inference(model, *args, **kwargs)

    def deactivate(self) -> None:
        with self._lock:
            self._restore_base()

    def latest_sessions(self) -> list[tuple[str, int]]:
        """Discover one current delta per session, ordered for stable tie-breaking."""
        sessions = []
        for directory in sorted(self.root.iterdir()):
            if directory.is_dir():
                versions = [int(p.stem[1:]) for p in directory.glob("v*.pt")
                            if p.stem[1:].isdigit()]
                if versions:
                    sessions.append((directory.name, max(versions)))
        return sessions

    def route_and_run(self, question: str, probe: Callable, score: Callable,
                      generate: Callable | None = None, *,
                      score_session: Callable | None = None) -> RoutedAnswer:
        """Probe every latest session delta, score candidates, then answer.

        probe(model, question) returns a short answer string.
        score(question, candidate) returns a finite scalar; larger is better.
        Optional score_session(question, candidate, session_id, version) replaces
        score, allowing verification against that session's original stored facts.
        Session IDs identify provenance only; they are not part of the question.
        generate(model, question) returns the complete answer synchronously.
        Omit generate for routing only; answer is then an empty string.
        Scoring runs with base weights restored, never under a candidate delta.
        Exact ties prefer the most recently saved memory and are exposed in the result.
        Equal save timestamps fall back to lexicographic session order.
        A lock covers discovery, all probes, scoring and final generation.
        """
        if not question.strip():
            raise ValueError("question must not be empty")
        with self._lock:
            training = self.model.training
            self.model.eval()
            try:
                self._restore_base()
                sessions = self.latest_sessions()
                if not sessions:
                    raise ValueError("No saved session deltas available")
                results = []
                with torch.inference_mode():
                    for session_id, version in sessions:
                        candidate = self.run(session_id, version, probe, question)
                        if not isinstance(candidate, str) or not candidate.strip():
                            raise ValueError(f"Empty or invalid probe from {session_id}")
                        value = float(score_session(question, candidate, session_id, version)
                                      if score_session is not None else score(question, candidate))
                        if not math.isfinite(value):
                            raise ValueError(f"Non-finite score from {session_id}")
                        results.append(ProbeScore(session_id, version, candidate, value))
                    best_score = max(item.score for item in results)
                    finalists = [item for item in results if item.score == best_score]
                    selected = max(finalists, key=lambda item:
                                   self._saved_at_ns(item.session_id, item.version))
                    answer = (self.run(selected.session_id, selected.version,
                                       generate, question) if generate is not None else "")
                return RoutedAnswer(selected, tuple(results), answer,
                                    sum(r.score == selected.score for r in results) > 1)
            finally:
                self._restore_base()
                self.model.train(training)

    def _apply_delta(self, delta: TensorMap) -> None:
        with torch.no_grad():
            for name, session_delta in delta.items():
                if name not in self._base:
                    raise ValueError(f"Unexpected weight in session delta: {name}")
                parameter = self.model.get_parameter(name)
                if tuple(session_delta.shape) != tuple(parameter.shape):
                    raise ValueError(f"Shape mismatch for {name}")
                parameter.add_(session_delta.to(parameter.device, parameter.dtype))

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not session_id or Path(session_id).name != session_id:
            raise ValueError("session_id must be one non-empty path component")

    def _save(
        self,
        session_id: str,
        version: int,
        delta: TensorMap,
        parent_version: int | None,
    ) -> None:
        self._validate_session_id(session_id)
        session_dir = self.root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        torch.save(delta, session_dir / f"v{version}.pt")
        (session_dir / f"v{version}.json").write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "version": version,
                    "parent_version": parent_version,
                    "weight_names": list(delta),
                    "saved_at_ns": time.time_ns(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _saved_at_ns(self, session_id: str, version: int) -> int:
        """Persistent write recency; old checkpoints fall back to delta mtime."""
        metadata = self.root / session_id / f"v{version}.json"
        if metadata.exists():
            saved_at = json.loads(metadata.read_text(encoding="utf-8")).get("saved_at_ns")
            if saved_at is not None:
                return int(saved_at)
        return (self.root / session_id / f"v{version}.pt").stat().st_mtime_ns

    def _load(self, session_id: str, version: int) -> TensorMap:
        self._validate_session_id(session_id)
        path = self.root / session_id / f"v{version}.pt"
        return torch.load(path, map_location="cpu", weights_only=True)


def llama3_weight_names(hparams) -> list[str]:
    """Resolve AlphaEdit's configured Llama 3 FFN down-projection weights."""
    return [
        f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        for layer in hparams.layers
    ]


# Integration sketch:
# store = SessionFFNMemory(
#     model,
#     llama3_weight_names(hparams),
#     root="session_deltas",
# )
# cache_by_session[session_id] = store.write_session(
#     session_id, tok, requests, hparams,
#     cache_by_session[session_id], P, version=1,
# )
# result = store.route_and_run(question, probe, score, generate)
# print(result.selected.session_id, result.answer)
