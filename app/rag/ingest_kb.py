import os
import platform
import re
import shutil
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


def _apply_windows_openmp_workaround() -> None:
    """Avoid common OpenMP runtime conflicts on Windows.

    Some ML stacks on Windows may load multiple OpenMP runtimes (libiomp5md.dll),
    which can cause a hard crash. For classroom demo stability, enable the
    documented workaround unless the user explicitly configured it.
    """

    try:
        if platform.system().lower() != "windows":
            return
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    except Exception:
        return


_apply_windows_openmp_workaround()


@dataclass
class RawDoc:
    text: str
    metadata: Dict


@dataclass
class IngestProgress:
    version: int
    files: Dict[str, Dict[str, Any]]


def _env_flag(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _progress_key(kb_dir: Path, path: Path) -> str:
    """Stable relative key for progress file."""

    try:
        return str(path.resolve().relative_to(kb_dir.resolve())).replace("\\", "/")
    except Exception:
        return str(path.name)


def _load_progress(progress_path: Path) -> IngestProgress:
    try:
        if not progress_path.exists():
            return IngestProgress(version=1, files={})
        data = json.loads(progress_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return IngestProgress(version=1, files={})
        files = data.get("files")
        if not isinstance(files, dict):
            files = {}
        cleaned: Dict[str, Dict[str, Any]] = {}
        for k, v in files.items():
            if isinstance(k, str) and isinstance(v, dict):
                cleaned[k] = v
        return IngestProgress(version=int(data.get("version") or 1), files=cleaned)
    except Exception:
        return IngestProgress(version=1, files={})


def _save_progress(progress_path: Path, progress: IngestProgress) -> None:
    try:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = progress_path.with_suffix(progress_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps({"version": progress.version, "files": progress.files}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, progress_path)
    except Exception:
        return


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _count_cjk(s: str) -> int:
    return sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")


def _count_latin1_suspects(s: str) -> int:
    # Characters that often show up in mojibake when UTF-8 bytes are decoded as Latin-1.
    return sum(1 for ch in s if "\u00c0" <= ch <= "\u00ff")


def _pick_best_encoding(path: Path, candidates: List[str]) -> str:
    """Pick a likely-correct text encoding for Chinese KB CSVs.

    We avoid accepting the first "errors=replace" decode, because it can silently
    ingest mojibake into the vector store, which later surfaces as UI乱码。
    """

    try:
        with path.open("rb") as f:
            sample_bytes = f.read(65536)
    except Exception:
        return candidates[0] if candidates else "utf-8"

    best_enc = candidates[0] if candidates else "utf-8"
    best_score = -10**18

    for enc in candidates:
        try:
            s = sample_bytes.decode(enc, errors="replace")
        except Exception:
            continue

        # Badness signals
        repl = s.count("\ufffd")  # replacement char
        bom_garble = s.count("ï»¿")
        latin1 = _count_latin1_suspects(s)
        # Common UTF-8 mojibake markers
        mojibake_markers = s.count("Ã") + s.count("Â")

        # Goodness signal: CJK density
        cjk = _count_cjk(s)

        # Score tuned for Chinese medical QA CSVs.
        # Prefer high CJK and strongly penalize replacement/mojibake/latin1 noise.
        score = (cjk * 5) - (repl * 200) - (bom_garble * 200) - (mojibake_markers * 50) - (latin1 * 3)

        if score > best_score:
            best_score = score
            best_enc = enc

    return best_enc


def _parse_yaml_front_matter(md_text: str) -> Tuple[Dict[str, str], str]:
    """Parse a simple YAML front-matter block and strip it from the markdown.

    Supports the common pattern:
    ---
    key: value
    ...
    ---
    body...

    Returns:
        (meta, body_without_front_matter)
    """

    if not md_text:
        return {}, ""

    lines = md_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, md_text

    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return {}, md_text

    meta: Dict[str, str] = {}
    for raw in lines[1:end_idx]:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
            meta[k] = v

    body = "\n".join(lines[end_idx + 1 :]).lstrip("\n")
    return meta, body


def _read_pdf_pages(path: Path) -> List[RawDoc]:
    # PDF parsing via pypdf, preserving page metadata.
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise RuntimeError(
            "缺少依赖 pypdf，无法读取PDF。请先安装 pypdf。"
        ) from e

    reader = PdfReader(str(path))
    docs: List[RawDoc] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if not text:
            continue
        docs.append(
            RawDoc(
                text=text,
                metadata={
                    "source": str(path.name),
                    "page": i,
                    "section": "",
                },
            )
        )
    return docs


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    raw = (os.getenv(name) or "").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _resolve_embedding_device() -> str:
    """Resolve device for HuggingFaceEmbeddings.

    Env:
      - RAG_EMBEDDING_DEVICE: auto|cpu|cuda|cuda:0...

    Notes:
    - If forced to cuda* but torch/cuda isn't available, raise a clear error.
    """

    raw = (os.getenv("RAG_EMBEDDING_DEVICE") or "").strip().lower()
    if raw == "":
        raw = "auto"

    if raw != "auto":
        if raw.startswith("cuda"):
            try:
                import torch  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "已设置 RAG_EMBEDDING_DEVICE=cuda，但当前环境缺少 torch。请安装 CUDA 版 torch，或改用 RAG_EMBEDDING_DEVICE=cpu。"
                ) from e

            if not bool(getattr(torch.cuda, "is_available", lambda: False)()):
                raise RuntimeError(
                    "已设置 RAG_EMBEDDING_DEVICE=cuda，但 torch.cuda.is_available() 为 False。"
                    "请安装 CUDA 版 torch/驱动，或改用 RAG_EMBEDDING_DEVICE=cpu。"
                )
        return raw

    # auto
    try:
        import torch  # type: ignore

        if bool(getattr(torch.cuda, "is_available", lambda: False)()):
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _clean_cell(v: object, max_len: int = 6000) -> str:
    s = "" if v is None else str(v)
    s = s.replace("\u00a0", " ")
    s = " ".join(s.split())
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _csv_get_cell(row: List[str], header: List[str], *names: str) -> str:
    """Get a cell value by matching header names.

    Args:
        row: Current CSV row.
        header: Lowercased header columns.
        names: Candidate column names (English/Chinese).
    """

    for name in names:
        name_l = name.strip().lower()
        if name_l in header:
            i = header.index(name_l)
            if i < len(row):
                return _clean_cell(row[i])
    return ""


def _read_csv_docs(path: Path, *, max_rows: Optional[int] = None) -> List[RawDoc]:
    """Read medical QA pairs from a CSV file.

    The Chinese-medical-dialogue-data repo contains CSVs with columns like:
      department, title, ask, answer

    We try to detect a header; otherwise we fall back to positional columns.
    """

    docs: List[RawDoc] = []

    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    chosen = _pick_best_encoding(path, encodings)

    def _read_with_encoding(enc: str) -> List[RawDoc]:
        with path.open("r", encoding=enc, errors="replace", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            dialect = csv.excel
            try:
                # Restrict delimiters to avoid Sniffer incorrectly picking line breaks (e.g. '\r').
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            except Exception:
                pass

            # Defensive fallback: some files may be parsed as a single column containing comma-joined values.
            if getattr(dialect, "delimiter", None) in {"\r", "\n"}:
                dialect = csv.excel

            reader = csv.reader(f, dialect)
            rows_read = 0
            header: Optional[List[str]] = None

            for row_idx, row in enumerate(reader):
                if not row:
                    continue

                # If row is a single comma-separated string, split it.
                if len(row) == 1 and isinstance(row[0], str) and "," in row[0]:
                    row = [c.strip() for c in row[0].split(",")]

                # Detect header only on the first non-empty row.
                if header is None:
                    cand = [str(x).strip().lower() for x in row]
                    if any(
                        k in cand
                        for k in [
                            "department",
                            "ask",
                            "answer",
                            "title",
                            "科室",
                            "问题",
                            "回答",
                            "标题",
                        ]
                    ):
                        header = cand
                        continue
                    header = []  # no header

                if max_rows is not None and rows_read >= max_rows:
                    break

                    department = ""
                    title = ""
                    ask = ""
                    answer = ""

                header_cols: List[str] = header or []
                if header_cols:
                    # Map by known names (English or Chinese)
                    department = _csv_get_cell(row, header_cols, "department", "科室")
                    title = _csv_get_cell(row, header_cols, "title", "标题")
                    ask = _csv_get_cell(row, header_cols, "ask", "question", "问题")
                    answer = _csv_get_cell(row, header_cols, "answer", "response", "回答")
                else:
                    # Positional fallback: department, title, ask, answer
                    if len(row) >= 1:
                        department = _clean_cell(row[0])
                    if len(row) >= 2:
                        title = _clean_cell(row[1])
                    if len(row) >= 3:
                        ask = _clean_cell(row[2])
                    if len(row) >= 4:
                        answer = _clean_cell(row[3])

                # Skip rows without meaningful QA content.
                if not ask or not answer:
                    continue

                qa_text = "\n".join(
                    [
                        f"科室：{department}" if department else "",
                        f"主题：{title}" if title else "",
                        f"患者问题：{ask}",
                        f"医生回答：{answer}",
                    ]
                ).strip()
                if not qa_text:
                    continue

                docs.append(
                    RawDoc(
                        text=qa_text,
                        metadata={
                            "source_file": str(path.name),
                            "source": str(path.name),
                            "page": None,
                            "section": department or "",
                            "department": department,
                            "title": title,
                            "row": row_idx,
                            "domain": "medical_qa",
                        },
                    )
                )

                rows_read += 1

            return docs

    try:
        return _read_with_encoding(chosen)
    except Exception:
        # Fallback: try other encodings in case sampling heuristic was wrong.
        last_err: Optional[Exception] = None
        for enc in encodings:
            if enc == chosen:
                continue
            try:
                return _read_with_encoding(enc)
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"无法读取CSV文件：{path}. last_error={type(last_err).__name__}: {last_err}")


def _iter_csv_docs(
    path: Path,
    *,
    start_row: int = 0,
    max_rows: Optional[int] = None,
) -> Iterator[Tuple[int, RawDoc]]:
    """Stream QA rows from a huge CSV.

    Yields:
      (row_idx, RawDoc)

    Notes:
    - row_idx is the csv.reader enumeration index (0-based).
    - start_row allows resumable ingestion.
    """

    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    chosen = _pick_best_encoding(path, encodings)

    def _iter_with_encoding(enc: str) -> Iterator[Tuple[int, RawDoc]]:
        with path.open("r", encoding=enc, errors="replace", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            dialect = csv.excel
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            except Exception:
                pass

            if getattr(dialect, "delimiter", None) in {"\r", "\n"}:
                dialect = csv.excel

            reader = csv.reader(f, dialect)
            rows_yielded = 0
            header: Optional[List[str]] = None

            for row_idx, row in enumerate(reader):
                if row_idx < start_row:
                    continue
                if not row:
                    continue

                if len(row) == 1 and isinstance(row[0], str) and "," in row[0]:
                    row = [c.strip() for c in row[0].split(",")]

                # Detect header only on the first non-empty row after start_row.
                if header is None:
                    cand = [str(x).strip().lower() for x in row]
                    if any(
                        k in cand
                        for k in [
                            "department",
                            "ask",
                            "answer",
                            "title",
                            "科室",
                            "问题",
                            "回答",
                            "标题",
                        ]
                    ):
                        header = cand
                        continue
                    header = []  # no header

                if max_rows is not None and rows_yielded >= max_rows:
                    break

                    department = ""
                    title = ""
                    ask = ""
                    answer = ""

                header_cols: List[str] = header or []
                if header_cols:
                    department = _csv_get_cell(row, header_cols, "department", "科室")
                    title = _csv_get_cell(row, header_cols, "title", "标题")
                    ask = _csv_get_cell(row, header_cols, "ask", "question", "问题")
                    answer = _csv_get_cell(row, header_cols, "answer", "response", "回答")
                else:
                    # Positional fallback: department, title, ask, answer
                    if len(row) >= 1:
                        department = _clean_cell(row[0])
                    if len(row) >= 2:
                        title = _clean_cell(row[1])
                    if len(row) >= 3:
                        ask = _clean_cell(row[2])
                    if len(row) >= 4:
                        answer = _clean_cell(row[3])

                if not ask or not answer:
                    continue

                qa_text = "\n".join(
                    [
                        f"科室：{department}" if department else "",
                        f"主题：{title}" if title else "",
                        f"患者问题：{ask}",
                        f"医生回答：{answer}",
                    ]
                ).strip()
                if not qa_text:
                    continue

                yield (
                    row_idx,
                    RawDoc(
                        text=qa_text,
                        metadata={
                            "source_file": str(path.name),
                            "source": str(path.name),
                            "page": None,
                            "section": department or "",
                            "department": department,
                            "title": title,
                            "row": row_idx,
                            "domain": "medical_qa",
                        },
                    ),
                )
                rows_yielded += 1

            return

    try:
        yield from _iter_with_encoding(chosen)
        return
    except Exception:
        last_err: Optional[Exception] = None
        for enc in encodings:
            if enc == chosen:
                continue
            try:
                yield from _iter_with_encoding(enc)
                return
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"无法读取CSV文件：{path}. last_error={type(last_err).__name__}: {last_err}")


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)\s*$")


def _split_markdown_into_sections(md_text: str) -> List[Tuple[str, str]]:
    """Return list of (section_title, section_text)."""
    lines = md_text.splitlines()

    sections: List[Tuple[str, List[str]]] = []
    current_title = ""
    current_buf: List[str] = []

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            # flush previous
            if current_buf:
                sections.append((current_title, current_buf))
            current_title = m.group(2).strip()
            current_buf = []
        else:
            current_buf.append(line)

    if current_buf:
        sections.append((current_title, current_buf))

    out: List[Tuple[str, str]] = []
    for title, buf in sections:
        text = "\n".join(buf).strip()
        if text:
            out.append((title, text))
    return out


def load_raw_docs(kb_dir: Path) -> List[RawDoc]:
    docs: List[RawDoc] = []
    if not kb_dir.exists():
        return docs

    for path in sorted(kb_dir.rglob("*")):
        if path.is_dir():
            continue

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            docs.extend(_read_pdf_pages(path))
        elif suffix == ".csv":
            # IMPORTANT: this dataset can be huge (hundreds of thousands rows).
            # Use RAG_CSV_MAX_ROWS to cap rows for a demo ingestion.
            max_rows = _env_int("RAG_CSV_MAX_ROWS", default=None)
            docs.extend(_read_csv_docs(path, max_rows=max_rows))
        elif suffix in {".md", ".markdown"}:
            md = _read_text_file(path)
            fm, body = _parse_yaml_front_matter(md)

            base_md = {
                "source_file": str(path.name),
                # Prefer the original URL (if present) so citations are traceable.
                "source": (fm.get("source_url") or str(path.name)).strip(),
                "source_url": (fm.get("source_url") or "").strip(),
                "title": (fm.get("title") or "").strip(),
                "publisher": (fm.get("publisher") or "").strip(),
                "captured_at": (fm.get("captured_at") or "").strip(),
                "domain": (fm.get("domain") or "").strip(),
                "page": None,
            }

            for section_title, section_text in _split_markdown_into_sections(body):
                docs.append(
                    RawDoc(
                        text=section_text,
                        metadata={
                            **base_md,
                            "section": section_title or "",
                        },
                    )
                )
        elif suffix in {".txt"}:
            txt = _read_text_file(path).strip()
            if txt:
                docs.append(
                    RawDoc(
                        text=txt,
                        metadata={
                            "source": str(path.name),
                            "page": None,
                            "section": "",
                        },
                    )
                )
        else:
            # skip unsupported types
            continue

    return docs


def build_and_persist_store(
    kb_dir: Path,
    persist_dir: Path,
    collection_name: str = "medical_kb",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    embedding_model_name: Optional[str] = None,
) -> int:
    """Ingest kb_docs into a persistent Chroma store.

    Returns number of chunks ingested.
    """
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import Chroma
        from langchain.schema import Document
    except Exception as e:
        raise RuntimeError(
            "缺少 langchain/chroma 相关依赖，无法构建向量库。"
        ) from e

    only_medical_qa = _env_flag("RAG_ONLY_MEDICAL_QA", "0")
    use_streaming = only_medical_qa or _env_flag("RAG_STREAMING", "0")

    if not use_streaming:
        raw_docs = load_raw_docs(kb_dir)
        if not raw_docs:
            return 0
    else:
        raw_docs = []

    model_name = (
        embedding_model_name
        or (os.getenv("HF_EMBEDDING_MODEL") or "").strip()
        # Default to a smaller multilingual model to keep Windows demo setup smooth.
        # (You can override via HF_EMBEDDING_MODEL.)
        or "intfloat/multilingual-e5-small"
    )

    device = _resolve_embedding_device()
    print(f"[RAG_INGEST] embedding_model={model_name} embedding_device={device}", flush=True)

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    # Optional: reset existing store for reproducible rebuilds.
    do_reset = _env_flag("RAG_RESET", "0")
    if do_reset:
        try:
            shutil.rmtree(persist_dir, ignore_errors=True)
        except Exception:
            pass

    persist_dir.mkdir(parents=True, exist_ok=True)

    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    if not use_streaming:
        documents: List[Document] = []
        chunk_counter = 0
        for rd in raw_docs:
            chunks = splitter.split_text(rd.text)
            for local_idx, chunk in enumerate(chunks):
                chunk_counter += 1
                chunk_id = f"{rd.metadata.get('source','')}:{rd.metadata.get('page','')}:{rd.metadata.get('section','')}:{local_idx}"
                md = dict(rd.metadata)
                md["chunk_id"] = chunk_id
                documents.append(Document(page_content=chunk, metadata=md))

        vectordb.add_documents(documents)
        vectordb.persist()
        return chunk_counter

    # Streaming branch: designed for full ingestion of Chinese-medical-dialogue-data.
    # Avoids building huge in-memory lists.
    qa_root = kb_dir / "medical_qa"
    csv_files = sorted([p for p in qa_root.rglob("*.csv") if p.is_file()]) if qa_root.exists() else []
    if not csv_files:
        return 0

    def _infer_department_group(csv_path: Path) -> str:
        """Infer a stable department label from dataset folder structure.

        For Chinese-medical-dialogue-data, CSVs are typically under:
          .../Data_数据/<DEPT_DIR>/*.csv
        Example:
          .../Data_数据/IM_内科/内科5000-33000.csv
        """

        try:
            parts = list(csv_path.parts)
            for i, part in enumerate(parts):
                if part in {"Data_数据", "Data", "数据"}:
                    if i + 1 < len(parts):
                        return str(parts[i + 1])
        except Exception:
            pass
        return str(csv_path.parent.name)

    progress_path = Path((os.getenv("RAG_PROGRESS_PATH") or "").strip() or (persist_dir / "ingest_progress.json"))
    if not progress_path.is_absolute():
        progress_path = persist_dir / progress_path

    resume = _env_flag("RAG_RESUME", "1")
    progress = IngestProgress(version=1, files={}) if (do_reset or (not resume)) else _load_progress(progress_path)
    if do_reset:
        try:
            if progress_path.exists():
                progress_path.unlink()
        except Exception:
            pass

    # Per-department row cap (for a controlled demo run).
    # 0 or empty means unlimited.
    per_dept_max_rows = _env_int("RAG_QA_PER_DEPT_MAX_ROWS", default=0) or 0

    # Load department counters from progress for resumability.
    dept_counts: Dict[str, int] = {}
    try:
        dc = progress.files.get("__dept_counts__")
        if isinstance(dc, dict):
            for k, v in dc.items():
                if not isinstance(k, str):
                    continue
                try:
                    dept_counts[k] = int(v)
                except Exception:
                    dept_counts[k] = 0
    except Exception:
        dept_counts = {}

    batch_size = int((os.getenv("RAG_INGEST_BATCH_SIZE") or "256").strip() or "256")
    persist_every_batches = int((os.getenv("RAG_PERSIST_EVERY_N_BATCHES") or "10").strip() or "10")
    progress_every_rows = int((os.getenv("RAG_PROGRESS_EVERY_ROWS") or "2000").strip() or "2000")

    # For CSV QA, default is no extra splitting unless text is very long.
    split_csv = _env_flag("RAG_SPLIT_CSV", "0")
    csv_soft_limit = int((os.getenv("RAG_CSV_SOFT_MAX_CHARS") or "6000").strip() or "6000")
    max_csv_rows = _env_int("RAG_CSV_MAX_ROWS", default=None)

    batch: List[Document] = []
    chunk_counter = 0
    batches_written = 0

    print(
        f"[RAG_INGEST] mode=streaming qa_only=1 files={len(csv_files)} batch_size={batch_size} "
        f"per_dept_max_rows={per_dept_max_rows} resume={resume} reset={do_reset}",
        flush=True,
    )

    for csv_path in csv_files:
        dept = _infer_department_group(csv_path)

        if per_dept_max_rows > 0 and dept_counts.get(dept, 0) >= per_dept_max_rows:
            # Skip whole file once the department budget is consumed.
            key = _progress_key(kb_dir, csv_path)
            st = progress.files.get(key, {})
            if not (isinstance(st, dict) and bool(st.get("done"))):
                progress.files[key] = {"last_row": int(st.get("last_row") or 0) if isinstance(st, dict) else 0, "done": True, "reason": "dept_limit_reached"}
                progress.files["__dept_counts__"] = dict(dept_counts)
                _save_progress(progress_path, progress)
            print(f"[SKIP] dept={dept} already_ingested={dept_counts.get(dept,0)} reached_limit={per_dept_max_rows} file={csv_path.name}", flush=True)
            continue

        key = _progress_key(kb_dir, csv_path)
        st = progress.files.get(key, {})
        if isinstance(st, dict) and bool(st.get("done")):
            # If we previously stopped because of a per-department cap, allow continuation
            # when the cap is increased OR removed (0 means unlimited).
            reason = str(st.get("reason") or "")
            if not (
                reason == "dept_limit_reached"
                and (
                    per_dept_max_rows == 0
                    or int(dept_counts.get(dept, 0) or 0) < per_dept_max_rows
                )
            ):
                continue
        start_row = 0
        if isinstance(st, dict):
            try:
                start_row = int(st.get("last_row") or 0)
            except Exception:
                start_row = 0

        rows_since_save = 0
        last_seen_row = start_row

        print(
            f"[FILE] dept={dept} start file={key} start_row={start_row} dept_ingested={dept_counts.get(dept,0)}",
            flush=True,
        )

        for row_idx, rd in _iter_csv_docs(csv_path, start_row=start_row, max_rows=max_csv_rows):
            last_seen_row = row_idx
            text = rd.text
            if not text:
                continue

            if per_dept_max_rows > 0 and dept_counts.get(dept, 0) >= per_dept_max_rows:
                # Stop reading more rows for this department.
                break

            # Count accepted QA rows (not raw CSV rows).
            dept_counts[dept] = int(dept_counts.get(dept, 0)) + 1

            if (not split_csv) and len(text) <= csv_soft_limit:
                chunks = [text]
            else:
                chunks = splitter.split_text(text)

            for local_idx, chunk in enumerate(chunks):
                chunk_counter += 1
                md = dict(rd.metadata)
                md["department_group"] = dept
                md["source_path"] = key
                md["chunk_id"] = f"{rd.metadata.get('source','')}:{rd.metadata.get('row','')}:{local_idx}"
                batch.append(Document(page_content=chunk, metadata=md))

                if len(batch) >= batch_size:
                    vectordb.add_documents(batch)
                    batch.clear()
                    batches_written += 1
                    if persist_every_batches > 0 and (batches_written % persist_every_batches == 0):
                        vectordb.persist()

                        print(
                            f"[PERSIST] dept={dept} file={csv_path.name} batches_written={batches_written} chunks_total={chunk_counter}",
                            flush=True,
                        )

            rows_since_save += 1
            if rows_since_save >= progress_every_rows:
                progress.files[key] = {"last_row": int(last_seen_row), "done": False}
                progress.files["__dept_counts__"] = dict(dept_counts)
                _save_progress(progress_path, progress)
                rows_since_save = 0

                print(
                    f"[PROGRESS] dept={dept} file={csv_path.name} last_row={last_seen_row} "
                    f"dept_ingested={dept_counts.get(dept,0)} chunks_total={chunk_counter}",
                    flush=True,
                )

        reason = "done"
        if per_dept_max_rows > 0 and dept_counts.get(dept, 0) >= per_dept_max_rows:
            reason = "dept_limit_reached"

        progress.files[key] = {"last_row": int(last_seen_row), "done": True, "reason": reason}
        progress.files["__dept_counts__"] = dict(dept_counts)
        _save_progress(progress_path, progress)

        print(
            f"[DONE_FILE] dept={dept} file={csv_path.name} last_row={last_seen_row} dept_ingested={dept_counts.get(dept,0)} reason={reason}",
            flush=True,
        )

    if batch:
        vectordb.add_documents(batch)
        batch.clear()

    vectordb.persist()

    # Final summary (useful for demo runs)
    try:
        top = sorted(dept_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
        shown = top[:50]
        print("[SUMMARY] per-dept ingested QA rows:", flush=True)
        for dept, cnt in shown:
            print(f"  - {dept}: {cnt}", flush=True)
    except Exception:
        pass
    return chunk_counter


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    kb_dir = base_dir / "rag/kb_docs" # 
    persist_dir = base_dir / "rag/kb_store"

    chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))

    count = build_and_persist_store(
        kb_dir=kb_dir,
        persist_dir=persist_dir,
        collection_name=os.getenv("RAG_COLLECTION", "medical_kb"),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model_name=os.getenv("HF_EMBEDDING_MODEL") or None,
    )

    print(f"Ingested chunks: {count}")


if __name__ == "__main__":
    main()
