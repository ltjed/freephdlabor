# ============================================================================
# index_repo.py — semantic indexer with thread‑safe live updates + rich demo
# ============================================================================
"""
Self‑contained toolkit that
  • Syncs an on‑disk knowledge base to a FAISS vector index on instantiation.
  • Optionally starts a Watchdog observer (`watch=True`) so future file events
    stream into the index.
  • Guards *all* read/write paths with an `RLock`; if a search arrives while
    re‑indexing is in progress it blocks until the update finishes, ensuring
    query results always reflect the newest state.

"""

from __future__ import annotations

import ast
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
import textwrap

import faiss  # type: ignore
import numpy as np
from openai import OpenAI  # type: ignore
from watchdog.events import FileSystemEventHandler  # type: ignore
from watchdog.observers import Observer  # type: ignore

from dotenv import load_dotenv
load_dotenv(override=True)

# ─────────────────────────────── Globals ────────────────────────────────────
EMBED_MODEL = "text-embedding-3-small"
ROOT = Path("knowledge_base")
INDEX_DIR = Path("vector_store")
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "metadata.json"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_EMBEDDINGS"))

# ---------- helper: (mtime_ns, size) signature ----------

def _file_sig(p: Path):
    st = p.stat()
    return st.st_mtime_ns, st.st_size

# ────────────────────────────── Chunking ────────────────────────────────────

def chunk_python(path: Path) -> List[Dict[str, Any]]:
    source = path.read_text(encoding="utf-8", errors="ignore")
    norm_path = str(path.resolve())
    tree = ast.parse(source)
    lines = source.splitlines()
    chunks: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = max(
                (getattr(n, "end_lineno", getattr(n, "lineno", start)) - 1
                 for n in ast.walk(node)
                 if hasattr(n, "lineno") or hasattr(n, "end_lineno")),
                default=start
            )

            text = "\n".join(lines[start : end + 1])
            uid = f"{norm_path}:{start}:{end}"
            chunks.append({
                "id": uid,
                "content": text,
                "meta": {
                    "file": str(norm_path),
                    "start_line": start + 1,
                    "end_line": end + 1,
                    "type": node.__class__.__name__,
                    "content": text,  # embed content directly
                },
            })
    return chunks



def chunk_markdown(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    norm_path = str(path.resolve())
    # skip chunking if the file is small
    if len(text) < 500:
        return [{
            "id": f"{norm_path}:whole",
            "content": text,
            "meta": {
                "file": str(norm_path),
                "type": "Text",
                "index": 0,
                "content": text
            }
        }]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    blocks = splitter.split_text(text)
    chunks: List[Dict[str, Any]] = []
    for i, block in enumerate(blocks):
        uid = f"{norm_path}:chunk-{i}"
        chunks.append({
            "id": uid,
            "content": block,
            "meta": {
                "file": str(norm_path),
                "type": "Text",
                "index": i,
                "content": block  # ✅ Add this line
            },
        })
    return chunks


def chunk_file(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".py":
        return chunk_python(path)
    if path.suffix in {".md", ".txt"}:
        return chunk_markdown(path)
    return []

# ─────────────────────────────── Embedding ──────────────────────────────────

def embed_texts(texts: list[str], client, embed_model: str) -> np.ndarray:
    resp = client.embeddings.create(model=embed_model, input=texts)
    return np.asarray([d.embedding for d in resp.data], dtype="float32")

# ─────────────────────────────── Vector store ───────────────────────────────
class FaissStore:
    def __init__(self, dim: int, index_dir: Path, embed_model: str):
        self.index_dir = index_dir
        self.index_path = index_dir / "faiss.index"
        self.meta_path = index_dir / "metadata.json"
        self.embed_model = embed_model
        self.index_dir.mkdir(exist_ok=True)
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.meta: dict[str, dict[str, Any]] = json.loads(self.meta_path.read_text())
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
            self.meta = {}

    def add(self, ids, vectors, metas):
        id_ints = np.array([abs(hash(x)) % 2**63 for x in ids], dtype="int64")
        self.index.add_with_ids(vectors, id_ints)
        self.meta.update({str(i): m for i, m in zip(id_ints, metas)})

    def remove(self, ids):
        id_ints = np.array([abs(hash(x)) % 2**63 for x in ids], dtype="int64")
        self.index.remove_ids(id_ints)
        for i in id_ints:
            self.meta.pop(str(i), None)

    def save(self):
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.meta, indent=2))

    def search(self, vector, k):
        D, I = self.index.search(vector, k)
        hits = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            hits.append({"score": float(score), **self.meta[str(idx)]})
        return hits

# ─────────────────────────────── Indexer ────────────────────────────────────
class RepoIndexer:
    def __init__(
        self,
        root: str | Path,
        *,
        watch: bool = False,
        index_dir: Path = None,
        embed_model: str = "text-embedding-3-small",
        openai_api_key: str = None,
    ):
        self.root = Path(root)
        self.index_dir = index_dir or Path("vector_store")
        self.embed_model = embed_model
        self.client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY_EMBEDDINGS"))
        self._lock = threading.RLock()
        dim = len(embed_texts(["probe"], self.client, self.embed_model)[0])
        self.store = FaissStore(dim, self.index_dir, self.embed_model)
        self._sync_on_init()
        if watch:
            self._start_watcher()

    # ---------- public ----------
    def semantic_search(self, query: str, k: int = 5):
        with self._lock:
            vec = embed_texts([query], self.client, self.embed_model)
            return self.store.search(vec, k=k)

    def update_file(self, path: str | Path):
        with self._lock:
            path = Path(path)
            if not path.exists():
                self._delete_file(path)
            else:
                self._reindex_file(path)
            self.store.save()

    # ---------- sync ----------
    # def _sync_on_init(self):
    #     with self._lock:
    #         existing_files = {m["file"] for m in self.store.meta.values()}
    #         current_files = {str(p) for p in self.root.rglob("*.*")}
    #         for dead in existing_files - current_files:
    #             self._delete_file(Path(dead))
    #         for p_str in current_files:
    #             p = Path(p_str)
    #             sig = _file_sig(p)
    #             sig_in_meta: Optional[tuple[int, int]] = None
    #             for m in self.store.meta.values():
    #                 if m["file"] == p_str:
    #                     sig_in_meta = tuple(m.get("sig", ())) or None
    #                     break
    #             if sig_in_meta != sig:
    #                 self._reindex_file(p, sig)
    #         self.store.save()

    def _sync_on_init(self):
        with self._lock:
            existing_files = {m["file"] for m in self.store.meta.values()}
            current_files = {str(p.resolve()) for p in self.root.rglob("*.*")}
            for dead in existing_files - current_files:
                self._delete_file(Path(dead))
            for p_str in current_files:
                p = Path(p_str)
                sig = _file_sig(p)
                sig_in_meta: Optional[tuple[int, int]] = None
                for m in self.store.meta.values():
                    if m["file"] == p_str:
                        sig_in_meta = tuple(m.get("sig", ())) or None
                        break
                if sig_in_meta != sig:
                    self._reindex_file(p, sig)
            self.store.save()

    # ---------- helpers ----------
    # def _reindex_file(self, path: Path, sig: Optional[tuple[int, int]] = None):
    #     new_chunks = chunk_file(path)
    #     old_ids = [id_ for id_, m in self.store.meta.items() if m["file"] == str(path)]
    #     if old_ids:
    #         self.store.remove(old_ids)
    #     if new_chunks:
    #         ids = [c["id"] for c in new_chunks]
    #         vecs = embed_texts([c["content"] for c in new_chunks], self.client, self.embed_model)
    #         for c in new_chunks:
    #             c["meta"]["sig"] = sig or _file_sig(path)
    #             c["meta"]["content"] = c["content"]
    #         metas = [c["meta"] for c in new_chunks]
    #         self.store.add(ids, vecs, metas)

    def _reindex_file(self, path: Path, sig: Optional[tuple[int, int]] = None):
        norm_path = str(path.resolve())
        new_chunks = chunk_file(path)
        old_ids = [id_ for id_, m in self.store.meta.items() if m["file"] == norm_path]
        if old_ids:
            self.store.remove(old_ids)
        if new_chunks:
            ids = [c["id"] for c in new_chunks]
            vecs = embed_texts([c["content"] for c in new_chunks], self.client, self.embed_model)
            for c in new_chunks:
                c["meta"]["sig"] = sig or _file_sig(path)
                c["meta"]["content"] = c["content"]
                c["meta"]["file"] = norm_path  # always store normalized path
            metas = [c["meta"] for c in new_chunks]
            self.store.add(ids, vecs, metas)

    # def _delete_file(self, path: Path):
    #     dead = [id_ for id_, m in self.store.meta.items() if m["file"] == str(path)]
    #     if dead:
    #         self.store.remove(dead)

    def _delete_file(self, path: Path):
        norm_path = str(path.resolve())
        dead = [id_ for id_, m in self.store.meta.items() if m["file"] == norm_path]
        if dead:
            self.store.remove(dead)

    # ---------- Watchdog ----------
    def _start_watcher(self):
        class _H(FileSystemEventHandler):
            def __init__(self_outer, outer: "RepoIndexer"):
                self_outer.outer = outer
            def on_modified(self_outer, event):
                if not event.is_directory:
                    self_outer.outer.update_file(event.src_path)
            def on_created(self_outer, event):
                if not event.is_directory:
                    self_outer.outer.update_file(event.src_path)
            def on_deleted(self_outer, event):
                if not event.is_directory:
                    self_outer.outer.update_file(event.src_path)
        obs = Observer()
        obs.schedule(_H(self), str(self.root), recursive=True)
        obs.daemon = True
        obs.start()
        self._observer = obs

    def stop_watcher(self):
        if hasattr(self, "_observer"):
            self._observer.stop()
            self._observer.join()

    def get_query_results(self, query: str, k: int = 3) -> str:
        """Returns a clearly formatted string of top-k semantic search results."""
        header = f"\n🔍 Semantic Search Results\nQuery: {query}\n{'=' * 60}\n"
        result_blocks = []
        for i, hit in enumerate(self.semantic_search(query, k=k), 1):
            # rel_path = Path(hit["file"]).relative_to(self.root)
            rel_path = Path(hit["file"]).resolve().relative_to(self.root.resolve())
            score = hit["score"]
            content = hit.get("content", "[No content]").strip()
            block = (
                f"Result {i}:\n"
                f"File   : {rel_path}\n"
                f"Score  : {score:.3f}\n"
                f"Content:\n{textwrap.indent(content, '    ')}\n"
                f"{'-' * 60}"
            )
            result_blocks.append(block)
        return header + "\n\n".join(result_blocks)

    def get_unique_query_results(self, query: str, k: int = 3, max_attempts: int = 10) -> str:
        """Returns a clearly formatted string of up to k unique semantic search results."""
        seen = set()
        final_results = []
        attempt = 0

        while len(final_results) < k and attempt < max_attempts:
            results = self.semantic_search(query, k=k * (attempt + 1))
            for hit in results:
                content = hit.get("content", "").strip()
                if content in seen:
                    continue
                seen.add(content)
                final_results.append(hit)
                if len(final_results) == k:
                    break
            attempt += 1

        header = f"\n🔍 Semantic Search Results (Unique)\nQuery: {query}\n{'=' * 60}\n"
        result_blocks = []
        for i, hit in enumerate(final_results, 1):
            rel_path = Path(hit["file"]).resolve().relative_to(self.root.resolve())
            score = hit["score"]
            content = hit.get("content", "[No content]").strip()
            block = (
                f"Result {i}:\n"
                f"File   : {rel_path}\n"
                f"Score  : {score:.3f}\n"
                f"Content:\n{textwrap.indent(content, '    ')}\n"
                f"{'-' * 60}"
            )
            result_blocks.append(block)

        return header + "\n\n".join(result_blocks)



# ─────────────────────────────── Demo ───────────────────────────────────────
if __name__ == "__main__":
    # 1. build a richer sample repo -------------------------------------------------
    root = Path("knowledge_base")
    index_dir = Path("vector_store")
    if root.exists():
        shutil.rmtree(root)
    (root / "code_implementation" / "linear").mkdir(parents=True)
    (root / "code_implementation" / "network").mkdir(parents=True)
    (root / "textual_knowledge" / "algorithms").mkdir(parents=True)

    # Python files
    (root / "code_implementation" / "linear" / "simplex.py").write_text(
        '''"""Simplex core operations"""

def pivot(tableau, row, col):
    "Perform the pivotal exchange on the tableau"
    pass

class SimplexSolver:
    def solve(self):
        "Solve a linear program via the primal simplex method"
        pass
''',
        encoding="utf-8",
    )

    (root / "code_implementation" / "network" / "dijkstra.py").write_text(
        '''class Dijkstra:
    def shortest_path(self, graph, source):
        "Compute single-source shortest paths"
        pass
''',
        encoding="utf-8",
    )

    # Markdown docs
    (root / "textual_knowledge" / "algorithms" / "simplex.md").write_text(
        '''# Simplex Method

The simplex method iteratively pivots along edges of the feasible polytope
until the optimum vertex is reached.''',
        encoding="utf-8",
    )

    (root / "textual_knowledge" / "algorithms" / "dijkstra.md").write_text(
        '''# Dijkstra's Algorithm

Finds shortest paths from a source node in nonnegative weighted graphs.''',
        encoding="utf-8",
    )

    # Plain text glossary
    (root / "textual_knowledge" / "glossary.txt").write_text(
        "Branch and bound: algorithm for integer programming using upper/lower bounds.",
        encoding="utf-8",
    )

    # 2. instantiate indexer (auto sync + live updates) ---------------------------
    idx = RepoIndexer(
        root,
        watch=False,
        index_dir=index_dir,
        embed_model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY_EMBEDDINGS"),
    )
    print("[demo] Initial index built.\n")

    # 3. initial searches ----------------------------------------------------------
    for q in ("simplex pivot", "shortest path", "branch and bound"):
        print(idx.get_query_results(q))
        print("=" * 40)

    # 4. mutate the repo while watcher is running ---------------------------------
    print("[demo] Modifying files...")
    # append to simplex.md (update existing file)
    with (root / "textual_knowledge" / "algorithms" / "simplex.md").open("a", encoding="utf-8") as f:
        f.write("\n## Dual Simplex\nOperates on infeasible primal but feasible dual solutions.\n")

    # delete dijkstra doc
    os.remove(root / "textual_knowledge" / "algorithms" / "dijkstra.md")

    # add new branch and bound markdown
    (root / "textual_knowledge" / "algorithms" / "branch_and_bound.md").write_text(
        '''# Branch & Bound

Tree‑search method for mixed‑integer programs.''',
        encoding="utf-8",
    )

    # give the watcher a moment to process events
    time.sleep(1)

    # 5. post‑update searches ------------------------------------------------------
    for q in ("dual simplex", "branch and bound", "shortest path"):
        print(idx.get_query_results(q))
        print("=" * 40)

    # done
    idx.stop_watcher()
    print("[demo] Finished. You just witnessed live, thread-safe re-indexing 🤖")