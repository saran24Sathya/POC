import os
import json
import time
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional
import requests
import argparse

class CodebaseRAG:
    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        gen_model: str = "gpt-oss",
        embed_model: str = "nomic-embed-text",
        num_predict: int = 300,
        read_timeout: int = 120,
        max_context_chars: int = 18000,
        stream: bool = False,
        chunk_size: int = 1000,
        overlap_size: int = 200,
        cache_dir: Optional[str] = None,
        autosave_interval: int = 200,
        extensions: Optional[List[str]] = None,
        force_rebuild: bool = False
    ):
        self.ollama_host = ollama_host.rstrip("/")
        self.gen_model = gen_model
        self.embed_model = embed_model
        self.num_predict = num_predict
        self.read_timeout = read_timeout
        self.max_context_chars = max_context_chars
        self.stream = stream
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.cache_dir = cache_dir
        self.autosave_interval = autosave_interval
        self.force_rebuild = force_rebuild
        self.extensions = set(e.lower() for e in (extensions or [
            '.py','.js','.ts','.jsx','.tsx','.java','.cpp','.c','.cs',
            '.go','.rs','.yml','.yaml','.json','.md'
        ]))

        self.documents: List[Dict] = []
        self.document_embeddings: List[np.ndarray] = []
        self.conversation_history: List[Dict] = []
        self._embed_dim: int = 0
        self._available_models = self._fetch_models()
        self._normalize_and_validate_models()

        self._chunk_to_index: Dict[str, int] = {}
        self._existing_embeddings: Dict[str, np.ndarray] = {}

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            if not self.force_rebuild:
                self._load_cache()
            else:
                print("[INFO] force_rebuild enabled; ignoring existing cache.")

    # -------------------- Model utilities --------------------
    def _fetch_models(self) -> List[str]:
        try:
            r = requests.get(f"{self.ollama_host}/api/tags", timeout=10)
            if r.status_code == 200:
                return [m.get("name","") for m in r.json().get("models",[])]
        except Exception:
            pass
        return []

    def _normalized(self, name: str) -> str:
        return name.replace(":latest","")

    def _normalize_and_validate_models(self):
        normalized = {self._normalized(n) for n in self._available_models}
        if self._normalized(self.embed_model) not in normalized:
            print(f"[WARN] Embedding model '{self.embed_model}' not found. Pull it: ollama pull {self.embed_model}")
        if self._normalized(self.gen_model) not in normalized:
            if "gpt-oss" in normalized:
                print(f"[INFO] Generation model '{self.gen_model}' missing. Falling back to gpt-oss.")
                self.gen_model = "gpt-oss"
            else:
                print(f"[WARN] Generation model '{self.gen_model}' not found. Pull e.g. gpt-oss / mistral / llama3.2:3b.")
        self._ping_server()

    def _ping_server(self):
        try:
            r = requests.get(f"{self.ollama_host}/api/version", timeout=5)
            if r.status_code == 200:
                print(f"[INFO] Ollama version: {r.json().get('version','?')}")
        except Exception:
            print("[WARN] Could not ping Ollama.")

    # -------------------- Cache handling --------------------
    def _cache_docs_path(self) -> str:
        return os.path.join(self.cache_dir, "documents.json")

    def _cache_emb_path(self) -> str:
        return os.path.join(self.cache_dir, "embeddings.npy")

    def _chunk_key(self, doc: Dict) -> str:
        return f"{doc['file_path']}:{doc['start_line']}-{doc['end_line']}:{doc['hash']}"

    def _load_cache(self):
        docs_path = self._cache_docs_path()
        emb_path = self._cache_emb_path()
        if not (os.path.exists(docs_path) and os.path.exists(emb_path)):
            print("[INFO] No existing cache found; starting fresh.")
            return
        try:
            with open(docs_path, "r", encoding="utf-8") as f:
                cached_docs = json.load(f)
            cached_embs = np.load(emb_path, allow_pickle=True)
            if len(cached_docs) != len(cached_embs):
                print("[WARN] Cache size mismatch; deleting corrupt cache.")
                try:
                    os.remove(docs_path)
                    os.remove(emb_path)
                except Exception:
                    pass
                return
            self.documents = cached_docs
            self.document_embeddings = list(cached_embs)
            if self.document_embeddings:
                self._embed_dim = self.document_embeddings[0].shape[0]
            for idx, doc in enumerate(self.documents):
                key = self._chunk_key(doc)
                self._chunk_to_index[key] = idx
                self._existing_embeddings[key] = self.document_embeddings[idx]
            print(f"[INFO] Loaded cache: {len(self.documents)} chunks.")
        except Exception as e:
            print(f"[WARN] Failed to load cache: {e}")

    def _save_cache(self):
        if not self.cache_dir:
            return
        try:
            with open(self._cache_docs_path(), "w", encoding="utf-8") as f:
                json.dump(self.documents, f)
            np.save(self._cache_emb_path(), np.array(self.document_embeddings, dtype=object))
            print(f"[INFO] Cache saved ({len(self.documents)} chunks).")
        except Exception as e:
            print(f"[WARN] Failed to save cache: {e}")

    # -------------------- Chunking --------------------
    def chunk_code_with_overlap(self, content: str, file_path: str) -> List[Dict]:
        lines = content.split('\n')
        chunks: List[Dict] = []
        cur_lines: List[str] = []
        cur_size = 0
        for i, line in enumerate(lines):
            cur_lines.append(line)
            cur_size += len(line) + 1
            if (cur_size >= self.chunk_size or
                (line.strip().startswith(('def ', 'class ')) and cur_size > self.chunk_size // 2)):
                chunk_text = '\n'.join(cur_lines)
                chunks.append({
                    'content': chunk_text,
                    'file_path': file_path,
                    'start_line': i - len(cur_lines) + 1,
                    'end_line': i,
                    'chunk_type': 'code'
                })
                overlap_line_count = max(1, self.overlap_size // 50)
                overlap = cur_lines[-overlap_line_count:]
                cur_lines = overlap
                cur_size = sum(len(l) + 1 for l in overlap)
        if cur_lines:
            chunk_text = '\n'.join(cur_lines)
            chunks.append({
                'content': chunk_text,
                'file_path': file_path,
                'start_line': len(lines) - len(cur_lines) + 1,
                'end_line': len(lines),
                'chunk_type': 'code'
            })
        return chunks

    # -------------------- Embeddings --------------------
    def embed_text(self, text: str) -> np.ndarray:
        try:
            r = requests.post(
                f"{self.ollama_host}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=(10, self.read_timeout)
            )
            if r.status_code == 200:
                vec = r.json().get("embedding", [])
                arr = np.array(vec, dtype=np.float32)
                if self._embed_dim == 0:
                    self._embed_dim = arr.shape[0]
                return arr
            print(f"[ERROR] Embedding failed {r.status_code}: {r.text[:120]}")
        except requests.exceptions.ReadTimeout:
            print(f"[ERROR] Embedding read timeout.")
        except Exception as e:
            print(f"[ERROR] Embedding exception: {e}")
        dim = self._embed_dim if self._embed_dim else 768
        return np.zeros(dim, dtype=np.float32)

    # -------------------- Processing & Resume --------------------
    def process_codebase(self, repo_path: str = "."):
        print("[INFO] Scanning files...")
        new_docs: List[Dict] = []

        if os.path.isfile(repo_path):
            try:
                with open(repo_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                rel = os.path.basename(repo_path)
                file_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                for c in self.chunk_code_with_overlap(content, rel):
                    c['hash'] = file_hash
                    new_docs.append(c)
                print(f"[INFO] Single file mode: {rel} -> {len(new_docs)} chunks.")
            except Exception as e:
                print(f"[WARN] Could not read file {repo_path}: {e}")
        else:
            for root, dirs, files in os.walk(repo_path):
                dirs[:] = [d for d in dirs if d not in {'.git','node_modules','__pycache__','.vscode','dist','build'}]
                for fname in files:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in self.extensions:
                        continue
                    path = os.path.join(root, fname)
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                    except Exception:
                        continue
                    rel = os.path.relpath(path, repo_path)
                    file_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                    chunks = self.chunk_code_with_overlap(content, rel)
                    for c in chunks:
                        c['hash'] = file_hash
                        new_docs.append(c)

        print(f"[INFO] Discovered {len(new_docs)} chunks this scan.")

        merged_docs: List[Dict] = []
        merged_embs: List[np.ndarray] = []
        reembed_queue: List[Dict] = []

        for doc in new_docs:
            key = self._chunk_key(doc)
            if key in self._existing_embeddings:
                merged_docs.append(doc)
                merged_embs.append(self._existing_embeddings[key])
            else:
                merged_docs.append(doc)
                reembed_queue.append(doc)

        self.documents = merged_docs
        self.document_embeddings = merged_embs
        print(f"[INFO] Reusing {len(merged_embs)} embeddings; need to embed {len(reembed_queue)} new chunks.")

        if reembed_queue:
            self._embed_missing(reembed_queue)
            self._save_cache()
        else:
            print("[INFO] Nothing new to embed; using cached vectors.")

    def _embed_missing(self, queue: List[Dict]):
        start = time.time()
        print("[INFO] Embedding new/changed chunks...")
        embedded = 0
        try:
            for doc in queue:
                emb = self.embed_text(f"File: {doc['file_path']}\n\n{doc['content']}")
                self.document_embeddings.append(emb)
                key = self._chunk_key(doc)
                self._existing_embeddings[key] = emb
                embedded += 1
                if embedded % self.autosave_interval == 0:
                    self._save_cache()
                    elapsed = time.time() - start
                    rate = embedded / elapsed if elapsed > 0 else 0
                    print(f"[PROGRESS] {embedded}/{len(queue)} new embeddings ({rate:.2f}/s)")
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted. Saving partial progress...")
            self._save_cache()
            raise

        total = time.time() - start
        avg = total / max(embedded,1)
        print(f"[INFO] Embedded {embedded} chunks in {total:.2f}s (avg {avg:.3f}s/chunk).")

    # -------------------- Retrieval --------------------
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        if not self.document_embeddings:
            return []
        q_emb = self.embed_text(query)
        sims: List[Tuple[Dict, float]] = []
        for i, emb in enumerate(self.document_embeddings):
            n1 = np.linalg.norm(q_emb); n2 = np.linalg.norm(emb)
            sim = 0.0 if n1 == 0 or n2 == 0 else float(np.dot(q_emb, emb)/(n1*n2))
            sims.append((self.documents[i], sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    def build_context(self, query: str, top_k: int = 5) -> str:
        results = self.search(query, top_k)
        if not results:
            return "[No context available]"
        parts = []
        for doc, score in results:
            parts.append(f"[{doc['file_path']} {doc['start_line']}-{doc['end_line']}] score={score:.3f}\n{doc['content']}")
        ctx = "\n\n".join(parts)
        if len(ctx) > self.max_context_chars:
            ctx = ctx[:self.max_context_chars] + "\n...[TRUNCATED]..."
        return ctx

    # -------------------- Generation --------------------
    def _post_generate(self, payload: Dict) -> requests.Response:
        return requests.post(
            f"{self.ollama_host}/api/generate",
            json=payload,
            timeout=(15, self.read_timeout)
        )

    def generate_response(self, query: str, top_k: int = 5) -> str:
        if not self.documents:
            return "[INFO] No documents indexed. Provide a directory path or valid file."
        context = self.build_context(query, top_k)
        system_prompt = (
            "You are an expert assistant for this repository. answers query and ask context if you need more."
        )
        prompt = f"System:\n{system_prompt}\n\nUser Query:\n{query}\n\nCode Context:\n{context}\n\nAnswer:"
        payload = {
            "model": self.gen_model,
            "prompt": prompt,
            "stream": self.stream,
            "options": {"num_predict": self.num_predict, "temperature": 0.3}
        }
        t0 = time.time()
        try:
            resp = self._post_generate(payload)
            if resp.status_code == 200:
                ans = resp.json().get("response","").strip()
            else:
                ans = f"[ERROR] Generation failed {resp.status_code}: {resp.text[:120]}"
        except Exception as e:
            ans = f"[ERROR] Generation exception: {e}"
        print(f"[INFO] Generation time {time.time()-t0:.2f}s")
        self.conversation_history.append({"query": query, "response": ans})
        return ans

# -------------------- CLI --------------------
def main():
    parser = argparse.ArgumentParser(
        description="Resumable codebase RAG using Ollama",
        usage=(
            "python rag_codebase.py [repo_or_file] [query] [options]\n\n"
            "Short examples:\n"
            "  python rag_codebase.py README.md \"summarize\" -m llama\n"
            "  python rag_codebase.py src \"how are embeddings cached?\" -m oss -k 8\n"
            "  python rag_codebase.py -r src -q \"chunking logic\" -m llama\n"
        )
    )

    # Positional (optional) for ultra-short usage
    parser.add_argument("pos_repo", nargs="?", help="Repository root or single file (positional)")
    parser.add_argument("pos_query", nargs="?", help="Query string (positional)")

    # Short flags
    parser.add_argument("-r", "--repo", default=".", help="Root path OR single file")
    parser.add_argument("-q", "--query", default=None, help="Question about the code")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Chunks to retrieve")
    parser.add_argument("-H", "--ollama-host", default="http://localhost:11434", help="Ollama host URL")
    parser.add_argument("-m", "--model", dest="gen_model", default="llama3.2:3b",
                        help="Generation model (e.g. llama3.2:3b, gpt-oss, mistral)")
    parser.add_argument("-e", "--embed-model", default="nomic-embed-text", help="Embedding model")
    parser.add_argument("-n", "--num-predict", type=int, default=300, help="Max tokens to predict")
    parser.add_argument("-t", "--read-timeout", type=int, default=120, help="Read timeout seconds")
    parser.add_argument("-C", "--max-context-chars", type=int, default=18000, help="Context char limit")
    parser.add_argument("-cs", "--chunk-size", type=int, default=1000, help="Chunk size chars")
    parser.add_argument("-ov", "--overlap-size", type=int, default=200, help="Chunk overlap chars")
    parser.add_argument("-c", "--cache-dir", type=str, default=".rag_cache", help="Cache directory")
    parser.add_argument("-a", "--autosave-interval", type=int, default=200, help="Autosave embedding interval")
    parser.add_argument("-x", "--extensions", type=str, default=None,
                        help="Comma list (e.g. .py,.cs)")
    parser.add_argument("-s", "--stream", action="store_true", help="Stream generation")
    parser.add_argument("-F", "--force-rebuild", action="store_true", help="Rebuild embeddings")

    # Ultra-short model alias flag
    parser.add_argument("-M", "--model-alias", choices=["oss", "llama"],
                        help="Quick model alias: oss=gpt-oss, llama=llama3.2:3b")

    args = parser.parse_args()

    # Apply positional fallbacks
    if args.pos_repo and args.repo == ".":
        args.repo = args.pos_repo
    if args.pos_query and args.query is None:
        args.query = args.pos_query

    # Apply alias if provided
    if args.model_alias:
        alias_map = {"oss": "gpt-oss", "llama": "llama3.2:3b"}
        args.gen_model = alias_map[args.model_alias]
        print(f"[INFO] Model alias selected: {args.gen_model}")

    if not args.query:
        parser.error("Query is required (provide via positional or -q/--query)")

    ext_list = [e.strip() for e in args.extensions.split(",")] if args.extensions else None

    rag = CodebaseRAG(
        ollama_host=args.ollama_host,
        gen_model=args.gen_model,
        embed_model=args.embed_model,
        num_predict=args.num_predict,
        read_timeout=args.read_timeout,
        max_context_chars=args.max_context_chars,
        stream=args.stream,
        chunk_size=args.chunk_size,
        overlap_size=args.overlap_size,
        cache_dir=args.cache_dir,
        autosave_interval=args.autosave_interval,
        extensions=ext_list,
        force_rebuild=args.force_rebuild
    )

    try:
        rag.process_codebase(args.repo)
    except KeyboardInterrupt:
        print("\n[INFO] Aborted during processing. Partial data saved.")
        return

    try:
        answer = rag.generate_response(args.query, top_k=args.top_k)
    except KeyboardInterrupt:
        print("\n[INFO] Aborted during generation.")
        return

    print("\n=== Answer ===\n")
    if not args.stream:
        print(answer)

if __name__ == "__main__":
    main()