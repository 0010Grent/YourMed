import argparse
import csv
import json
import pickle
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx


@dataclass
class RagCase:
    idx: int
    query: str
    reference: str
    top_k: int
    latency_ms: float
    evidence_count: int
    max_similarity: float
    hit: bool
    error: Optional[str]


def _load_meddg_dialogs(path: Path) -> List[Any]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"MedDG pickle top-level must be list, got: {type(obj)}")
    return obj


def _describe_meddg(obj: Any, *, max_dialogs: int = 2, max_turns: int = 3) -> None:
    print("[MedDG] top_type=", type(obj))
    if isinstance(obj, list):
        print("[MedDG] dialogs=", len(obj))
        for i, d in enumerate(obj[:max_dialogs]):
            print(f"[MedDG] dialog[{i}] type={type(d)}")
            if isinstance(d, list):
                print(f"[MedDG] dialog[{i}] turns={len(d)}")
                for j, t in enumerate(d[:max_turns]):
                    if isinstance(t, dict):
                        print(f"[MedDG] dialog[{i}].turn[{j}] keys={sorted(list(t.keys()))}")
                    else:
                        print(f"[MedDG] dialog[{i}].turn[{j}] type={type(t)}")


def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", "", s)
    return s


def _char_bigram_jaccard(a: str, b: str) -> float:
    a = _clean_text(a)
    b = _clean_text(b)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    a_bi = {a[i : i + 2] for i in range(len(a) - 1)}
    b_bi = {b[i : i + 2] for i in range(len(b) - 1)}
    if not a_bi or not b_bi:
        return 0.0
    inter = len(a_bi & b_bi)
    union = len(a_bi | b_bi)
    return inter / union if union else 0.0


def _extract_pairs(dialog: Any) -> List[Tuple[str, str]]:
    """Return list of (patient_sentence, doctor_sentence) pairs, aligned by adjacency."""
    if not isinstance(dialog, list):
        return []

    pairs: List[Tuple[str, str]] = []
    last_patient: Optional[str] = None

    for turn in dialog:
        if not isinstance(turn, dict):
            continue
        tid = turn.get("id")
        sent = turn.get("Sentence")
        if not isinstance(sent, str) or not sent.strip():
            continue
        sent = sent.strip()

        if tid == "Patients":
            last_patient = sent
        elif tid == "Doctor" and last_patient:
            pairs.append((last_patient, sent))
            last_patient = None

    return pairs


def _safe_get(d: Any, *path: str) -> Any:
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG offline quality eval via /v1/rag/retrieve")
    parser.add_argument("--meddg_path", required=True, help="Path to MedDG pickle e.g. app/MedDG_UTF8/test.pk")
    parser.add_argument("--base_url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--n", type=int, default=200, help="Number of (query, reference) pairs to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--top_k", type=int, default=5, help="Top K evidences")
    parser.add_argument("--top_n", type=int, default=30, help="Vector recall top_n (stage-1)")
    parser.add_argument("--use_rerank", type=int, default=1, help="Use rerank: 1/0")
    parser.add_argument("--timeout_s", type=float, default=30.0, help="HTTP timeout per request")
    parser.add_argument("--sim_threshold", type=float, default=0.08, help="Hit threshold for bigram Jaccard")
    parser.add_argument("--min_evidence_len", type=int, default=30, help="Min evidence length for coverage")
    parser.add_argument("--out_dir", default="reports", help="Output directory")
    args = parser.parse_args()

    meddg_path = Path(args.meddg_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dialogs = _load_meddg_dialogs(meddg_path)
    print(f"[MedDG] loaded_from={meddg_path}")
    _describe_meddg(dialogs)

    pairs: List[Tuple[str, str]] = []
    for dialog in dialogs:
        pairs.extend(_extract_pairs(dialog))

    if not pairs:
        raise ValueError("No (patient, doctor) pairs extracted from MedDG")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    pairs = pairs[: min(args.n, len(pairs))]

    cases: List[RagCase] = []

    with httpx.Client() as client:
        for i, (query, reference) in enumerate(pairs):
            payload: Dict[str, Any] = {
                "query": query,
                "top_k": args.top_k,
                "top_n": args.top_n,
                "use_rerank": bool(int(args.use_rerank)),
            }
            t0 = time.perf_counter()
            try:
                r = client.post(f"{args.base_url}/v1/rag/retrieve", json=payload, timeout=args.timeout_s)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                r.raise_for_status()
                data = r.json()

                # API contract: api_server returns key `evidence`.
                evidences = data.get("evidence")
                if not isinstance(evidences, list):
                    # Backwards-compat / tolerance for older scripts
                    evidences = data.get("evidences")
                if not isinstance(evidences, list):
                    evidences = []

                max_sim = 0.0
                for ev in evidences:
                    text = _safe_get(ev, "text")
                    if isinstance(text, str) and text:
                        max_sim = max(max_sim, _char_bigram_jaccard(text, reference))

                hit = max_sim >= args.sim_threshold

                cases.append(
                    RagCase(
                        idx=i,
                        query=query,
                        reference=reference,
                        top_k=args.top_k,
                        latency_ms=latency_ms,
                        evidence_count=len(evidences),
                        max_similarity=max_sim,
                        hit=hit,
                        error=None,
                    )
                )
            except Exception as e:  # noqa: BLE001
                latency_ms = (time.perf_counter() - t0) * 1000.0
                cases.append(
                    RagCase(
                        idx=i,
                        query=query,
                        reference=reference,
                        top_k=args.top_k,
                        latency_ms=latency_ms,
                        evidence_count=0,
                        max_similarity=0.0,
                        hit=False,
                        error=str(e),
                    )
                )

            if (i + 1) % 50 == 0:
                print(f"progress: {i + 1}/{len(pairs)} pairs")

    total = len(cases)
    errors = sum(1 for c in cases if c.error)
    hits = sum(1 for c in cases if c.hit)
    coverages = sum(
        1
        for c in cases
        if c.error is None and c.evidence_count > 0
    )

    # evidence length coverage (best effort)
    # We can only compute on successful calls and when evidence text exists
    long_text_coverage = 0
    for c in cases:
        if c.error is not None:
            continue
        # We didn't store evidence texts per-case to keep CSV smaller; treat evidence_count>0 as minimum coverage.
        # For a stricter coverage, we record max_similarity and evidence_count only.
        if c.evidence_count > 0:
            long_text_coverage += 1

    latencies = [c.latency_ms for c in cases]

    summary: Dict[str, Any] = {
        "pairs": total,
        "top_k": args.top_k,
        "sim_threshold": args.sim_threshold,
        "hit_rate": hits / total if total else 0.0,
        "coverage_rate": coverages / total if total else 0.0,
        "nonempty_evidence_rate": long_text_coverage / total if total else 0.0,
        "avg_max_similarity": (sum(c.max_similarity for c in cases) / total) if total else 0.0,
        "error_rate": errors / total if total else 0.0,
        "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "p95_latency_ms": (sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else 0.0),
    }

    summary_path = out_dir / "rag_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Keep existing filename for backwards compatibility.
    details_path = out_dir / "rag_eval_details.csv"
    with details_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "idx",
                "query",
                "reference",
                "top_k",
                "latency_ms",
                "evidence_count",
                "max_similarity",
                "hit",
                "error",
            ]
        )
        for c in cases:
            w.writerow(
                [
                    c.idx,
                    c.query,
                    c.reference,
                    c.top_k,
                    f"{c.latency_ms:.2f}",
                    c.evidence_count,
                    f"{c.max_similarity:.4f}",
                    1 if c.hit else 0,
                    c.error or "",
                ]
            )

    print(f"wrote: {summary_path}")
    print(f"wrote: {details_path}")

    # Also emit generic alias filenames as some runbooks request.
    alias_details = out_dir / "details.csv"
    try:
        alias_details.write_text(details_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"wrote: {alias_details}")
    except Exception:
        pass

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
