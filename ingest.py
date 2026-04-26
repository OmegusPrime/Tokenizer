import os
import pandas as pd
from pathlib import Path
from Metrics import Metrics
from JsonFormatter import JsonFormatter, log
import pypdf
from charset_normalizer import from_path
MAX_FILE_BYTES = 10 * 1024 * 1024 * 1024
CHUNK_SIZE     = 1024 * 1024
CSV_CHUNKSIZE  = 500
PERSONA_SEP    = "<|persona|>"
UTTERANCE_SEP  = "<|utterance|>"
SPEAKER1       = "<|speaker1|>"
SPEAKER2       = "<|speaker2|>"
class Ingest:

    def __init__(
        self,
        path,
        *,
        max_file_bytes: int  = MAX_FILE_BYTES,
        chunk_size: int      = CHUNK_SIZE,
        csv_chunksize: int   = CSV_CHUNKSIZE,
        allowlist_dirs: list = None,     # only descend into these dir names (None = all)
    ):
        self.path           = Path(path).resolve()
        self.max_file_bytes = max_file_bytes
        self.chunk_size     = chunk_size
        self.csv_chunksize  = csv_chunksize
        self.allowlist_dirs = set(allowlist_dirs) if allowlist_dirs else None
        self.metrics        = Metrics()
        self._seen_inodes: set[int] = set()   # dedup by inode

    # ── Safety gates ─────────────────────────────────────────────────────
    def _is_dotfile(self, p: Path) -> bool:
        return any(part.startswith('.') for part in p.parts)

    def _is_symlink(self, p: Path) -> bool:
        try:
            for part in [p, *p.parents]:
                if part.is_symlink():
                    return True
                if part == self.path:
                    break
        except OSError:
            return True
        return False

    def _is_confined(self, p: Path) -> bool:
        try:
            p.resolve().relative_to(self.path)
            return True
        except ValueError:
            return False

    def _is_duplicate_inode(self, p: Path) -> bool:
        try:
            inode = p.stat().st_ino
            if inode in self._seen_inodes:
                return True
            self._seen_inodes.add(inode)
            return False
        except OSError:
            return False

    def _safe(self, file_path: Path) -> bool:
        ctx = {"file": str(file_path)}

        if self._is_dotfile(file_path):
            self.metrics.skipped_dotfile += 1
            log.debug("skip dotfile", extra=ctx)
            return False

        if self._is_symlink(file_path):
            self.metrics.skipped_symlink += 1
            log.warning("skip symlink", extra=ctx)
            return False

        if not self._is_confined(file_path):
            self.metrics.skipped_confined += 1
            log.warning("skip path-traversal attempt", extra=ctx)
            return False

        if self._is_duplicate_inode(file_path):
            self.metrics.skipped_inode += 1
            log.debug("skip duplicate inode", extra=ctx)
            return False

        size = file_path.stat().st_size
        if size > self.max_file_bytes:
            self.metrics.skipped_size += 1
            log.warning("skip oversized file", extra={**ctx, "size_mb": round(size/1e6, 2)})
            return False

        return True

    # ── Encoding detection ────────────────────────────────────────────────

    def _read_text(self, file_path: Path) -> tuple[str, str]:
        """
        Detect encoding with charset-normalizer.
        Falls back to utf-8 errors='replace' and increments warning counter.
        Returns (text, encoding_used).
        """
        result = from_path(file_path).best()
        if result is not None:
            return str(result), result.encoding

        # fallback
        self.metrics.encoding_warns += 1
        log.warning("encoding detection failed, using utf-8 replace",
                    extra={"file": str(file_path)})
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read(), "utf-8-replace"

    # ── TXT ──────────────────────────────────────────────────────────────

    def ingest_txt(self, file_path: Path):
        ctx = {"file": str(file_path)}
        try:
            full_text, enc = self._read_text(file_path)
            log.debug("txt encoding detected", extra={**ctx, "encoding": enc})

            # stream in fixed-size chunks
            for i in range(0, len(full_text), self.chunk_size):
                chunk = full_text[i:i + self.chunk_size]
                if chunk.strip():
                    self.metrics.ingested += 1
                    self.metrics.by_type[".txt"] += 1
                    yield {"file": str(file_path), "type": "txt", "text": chunk}

        except Exception:
            self.metrics.errors += 1
            log.exception("txt ingest failed", extra=ctx)

    # ── CSV ──────────────────────────────────────────────────────────────

    def ingest_csv(self, file_path: Path):
        ctx = {"file": str(file_path)}
        try:
            for chunk_df in pd.read_csv(file_path, chunksize=self.csv_chunksize,
                                         dtype=str, keep_default_na=False):
                for row in chunk_df.itertuples(index=False):   # itertuples is faster than .values
                    text = ' '.join(str(v) for v in row)
                    if text.strip():
                        self.metrics.ingested += 1
                        self.metrics.by_type[".csv"] += 1
                        yield {"file": str(file_path), "type": "csv", "text": text}
        except Exception:
            self.metrics.errors += 1
            log.exception("csv ingest failed", extra=ctx)

    # ── PDF ──────────────────────────────────────────────────────────────

    def ingest_pdf(self, file_path: Path):
        ctx = {"file": str(file_path)}
        try:
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    if text and text.strip():
                        self.metrics.ingested += 1
                        self.metrics.by_type[".pdf"] += 1
                        yield {
                            "file":    str(file_path),
                            "type":    "pdf",
                            "page":    page_num,
                            "text":    text,
                        }
        except Exception:
            self.metrics.errors += 1
            log.exception("pdf ingest failed", extra=ctx)

    # ── DailyDialog schema ────────────────────────────────────────────────

    def ingest_dailydialog(self, file_path: Path):
        """
        DailyDialog format: each line is a dialogue, turns separated by `__eou__`.
        Yields one dict per dialogue with the turn list preserved.

        Schema:
            {
                "file":   str,
                "type":   "dailydialog",
                "turns":  ["utterance1", "utterance2", ...],
                "text":   "<flat joined string for tokenizer>"
            }
        """
        ctx = {"file": str(file_path)}
        try:
            full_text, enc = self._read_text(file_path)
            log.debug("dailydialog encoding", extra={**ctx, "encoding": enc})

            for line in full_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                turns = [t.strip() for t in line.split("__eou__") if t.strip()]
                if not turns:
                    continue
                self.metrics.ingested += 1
                self.metrics.by_type["dailydialog"] += 1
                yield {
                    "file":  str(file_path),
                    "type":  "dailydialog",
                    "turns": turns,
                    "text":  " ".join(turns),
                }
        except Exception:
            self.metrics.errors += 1
            log.exception("dailydialog ingest failed", extra=ctx)

    # ── PersonaChat schema ────────────────────────────────────────────────

    def ingest_personachat(self, file_path: Path):
        """
        PersonaChat format (ConvAI2 train/valid .txt):
            1 your persona: ...
            2 your persona: ...
            3 <speaker1 utt>\t<speaker2 utt>
            ...

        Yields one dict per conversation block.

        Schema:
            {
                "file":      str,
                "type":      "personachat",
                "persona":   ["persona line 1", ...],
                "turns":     [{"speaker": 1|2, "text": str}, ...],
                "text":      "<|persona|> p1 p2 <|utterance|> <|speaker1|> u1 <|speaker2|> u2 ..."
            }
        """
        ctx = {"file": str(file_path)}
        try:
            full_text, enc = self._read_text(file_path)
            log.debug("personachat encoding", extra={**ctx, "encoding": enc})

            persona, turns = [], []

            def _flush():
                if not (persona or turns):
                    return None
                parts = [PERSONA_SEP]
                parts += persona
                parts.append(UTTERANCE_SEP)
                for t in turns:
                    parts.append(SPEAKER1 if t["speaker"] == 1 else SPEAKER2)
                    parts.append(t["text"])
                return {
                    "file":    str(file_path),
                    "type":    "personachat",
                    "persona": list(persona),
                    "turns":   list(turns),
                    "text":    " ".join(parts),
                }

            for line in full_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                idx, _, rest = line.partition(" ")
                if not idx.isdigit():
                    continue

                if "your persona:" in rest:
                    if int(idx) == 1 and turns:
                        # new conversation block starting
                        doc = _flush()
                        if doc:
                            self.metrics.ingested += 1
                            self.metrics.by_type["personachat"] += 1
                            yield doc
                        persona.clear()
                        turns.clear()
                    persona.append(rest.split("your persona:", 1)[1].strip())
                else:
                    parts = rest.split("\t")
                    if len(parts) >= 2:
                        turns.append({"speaker": 1, "text": parts[0].strip()})
                        turns.append({"speaker": 2, "text": parts[1].strip()})

            # flush last block
            doc = _flush()
            if doc:
                self.metrics.ingested += 1
                self.metrics.by_type["personachat"] += 1
                yield doc

        except Exception:
            self.metrics.errors += 1
            log.exception("personachat ingest failed", extra=ctx)

    # ── Walk ─────────────────────────────────────────────────────────────

    def _detect_schema(self, file_path: Path):
        """Return the right handler based on filename hints or extension."""
        name = file_path.name.lower()
        if "dailydialog" in name:
            return self.ingest_dailydialog
        if "persona" in name or "convai" in name:
            return self.ingest_personachat
        return {
            ".txt": self.ingest_txt,
            ".csv": self.ingest_csv,
            ".pdf": self.ingest_pdf,
        }.get(file_path.suffix.lower())

    def ingest(self):
        for dirpath, dirnames, filenames in os.walk(self.path, followlinks=False):
            dir_path = Path(dirpath)

            # skip dotdirs in-place so os.walk doesn't descend into them
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith('.')
                and (self.allowlist_dirs is None or d in self.allowlist_dirs)
                and self._is_confined(dir_path / d)
            ]

            for filename in filenames:
                file_path = dir_path / filename

                if not self._safe(file_path):
                    continue

                handler = self._detect_schema(file_path)
                if handler is None:
                    continue

                yield from handler(file_path)

        log.info("ingest complete", extra={"metrics": self.metrics.report()})
