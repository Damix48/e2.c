#!/usr/bin/env python3
"""mteb_e2 — run MTEB tasks against the e2 engine (and optionally the OpenVINO INT8 IR).

    python tools/mteb_e2.py --tasks STSBenchmark,STS12,NFCorpus --engines e2,ov

EmbeddingGemma's documented prompt scheme is applied identically for every engine, so the scores
compare engines rather than prompt tuning: queries get "task: search result | query: ...",
documents get "title: <t> | text: ...", and symmetric tasks (STS) get
"task: sentence similarity | query: ...".
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import json
import numpy as np
from mteb.models import EncoderProtocol

HERE = Path(__file__).resolve().parent.parent

QS = "task: search result | query: "
DS = "task: sentence similarity | query: "

# EmbeddingGemma's documented per-task prefixes; picked from the task type when the harness does not
# say query/document (STS, classification and clustering are symmetric).
BY_TYPE = (
    ("classif", "task: classification | query: "),
    ("cluster", "task: clustering | query: "),
    ("sts", DS),
    ("pairclass", DS),
    ("rerank", QS),
    ("retriev", QS),
)


def _leaves(obj, path=""):
    """Flatten mteb's nested score structure into {path: value} for numeric leaves."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_leaves(v, path + "/" + str(k) if path else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(_leaves(v, path + "[%d]" % i))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[path] = round(float(obj), 5)
    return out


def prefix_query(texts):
    return [QS + t for t in texts]


def prefix_document(texts, titles=None):
    if titles is None:
        return ["title: none | text: " + t for t in texts]
    return [("title: " + (ti if ti else "none") + " | text: " + t) for t, ti in zip(texts, titles)]


def flatten_batches(inputs):
    """mteb hands a DataLoader of BatchedInput dicts; collect texts plus the title/query marker."""
    cols = {}
    for batch in inputs:
        if not isinstance(batch, dict):
            batch = dict(batch)
        for key in ("query", "document", "text", "title"):
            if key in batch and batch[key] is not None:
                cols.setdefault(key, []).extend(list(batch[key]))
    return cols


def strings_for(inputs, task_metadata=None, prompt_type=None):
    """The exact strings an engine should see for one MTEB split (model-card prompt scheme)."""
    cols = flatten_batches(inputs)
    if "query" in cols:
        return prefix_query(cols["query"])
    if "document" in cols:
        titles = cols.get("title")
        if titles is not None and len(titles) != len(cols["document"]):
            titles = None
        return prefix_document(cols["document"], titles)
    if "text" not in cols:
        raise ValueError("unsupported input batch: %r" % list(cols))
    pt = str(prompt_type) if prompt_type is not None else ""
    if "document" in pt.lower():
        return prefix_document(cols["text"], cols.get("title"))
    if "query" in pt.lower():
        return prefix_query(cols["text"])
    pre = DS
    ttype = str(getattr(task_metadata, "type", "")).lower() if task_metadata else ""
    for key, pfx in BY_TYPE:
        if key in ttype:
            pre = pfx
            break
    return [pre + t for t in cols["text"]]


def chunks(texts, size):
    for i in range(0, len(texts), size):
        yield texts[i:i + size]


class E2Encoder(EncoderProtocol):
    """Runs the C engine through tools/embed_cli in one process per encode() call."""

    mteb_model_meta = None

    def __init__(self, model_dir, cli, threads=14, chunk=4096):
        self.model_dir = str(model_dir)
        self.cli = str(cli)
        self.threads = threads
        self.chunk = chunk
        self.cache = {}


    def similarity(self, embeddings1, embeddings2):
        a = np.asarray(embeddings1, dtype=np.float64)
        b = np.asarray(embeddings2, dtype=np.float64)
        return np.asarray(a @ b.T)

    def similarity_pairwise(self, embeddings1, embeddings2):
        a = np.asarray(embeddings1, dtype=np.float64)
        b = np.asarray(embeddings2, dtype=np.float64)
        return np.asarray(np.sum(a * b, axis=1))

    def _run(self, texts):
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            fin, fout, fids = (os.path.join(td, n) for n in ("in.bin", "out.f32", "out.ids"))
            with open(fin, "wb") as f:
                for t in texts:
                    b = t.encode("utf-8")
                    f.write(struct.pack("<I", len(b)))
                    f.write(b)
            env = dict(os.environ, E2_THREADS=str(self.threads))
            subprocess.run([self.cli, self.model_dir, fin, fout, fids], check=True, env=env)
            v = np.fromfile(fout, dtype=np.float32).reshape(-1, 768)
            ids = []
            with open(fids, "rb") as f:
                data = f.read()
            i = 0
            while i < len(data):
                n = struct.unpack_from("<I", data, i)[0]
                i += 4
                ids.append(list(struct.unpack_from("<%di" % n, data, i)))
                i += 4 * n
        for t, d in zip(texts, ids):
            self.cache[t] = d
        return v

    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None,
               prompt_type=None, **kwargs):
        texts = strings_for(inputs, task_metadata, prompt_type)
        out = [self._run(c) for c in chunks(texts, self.chunk)]
        return np.concatenate(out) if len(out) > 1 else out[0]


class OvEncoder(EncoderProtocol):
    """The same prompts through the NNCF INT8 OpenVINO IR, on CPU with 14 threads."""

    mteb_model_meta = None

    def __init__(self, xml, model_dir, threads=14, device="CPU", chunk=64):
        import openvino as ov
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.core = ov.Core()
        model = self.core.read_model(str(xml))
        self.model = model
        cfg = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}
        if device == "CPU":
            cfg["INFERENCE_NUM_THREADS"] = threads
        self.compiled = self.core.compile_model(model, device, cfg)
        self.names = [i.get_any_name() for i in self.compiled.inputs]
        self.device = device
        self.chunk = chunk

    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None,
               prompt_type=None, **kwargs):
        texts = strings_for(inputs, task_metadata, prompt_type)
        outs = []
        for c in chunks(texts, self.chunk):
            enc = self.tok(c, padding="max_length", truncation=True, max_length=2048,
                           return_tensors="np")
            # pad to a 64-token multiple of the chunk's own max so the plugin sees few shapes
            smax = int(enc["attention_mask"].sum(axis=1).max())
            keep = max(16, (smax + 63) // 64 * 64)
            feeds = {"input_ids": enc["input_ids"][:, :keep].astype(np.int64),
                     "attention_mask": enc["attention_mask"][:, :keep].astype(np.int64)}
            outs.append(np.asarray(self.compiled(feeds)[0], dtype=np.float32).reshape(-1, 768))
        return np.concatenate(outs) if len(outs) > 1 else outs[0]

    def similarity(self, embeddings1, embeddings2):
        a = np.asarray(embeddings1, dtype=np.float64)
        b = np.asarray(embeddings2, dtype=np.float64)
        return np.asarray(a @ b.T)

    def similarity_pairwise(self, embeddings1, embeddings2):
        a = np.asarray(embeddings1, dtype=np.float64)
        b = np.asarray(embeddings2, dtype=np.float64)
        return np.asarray(np.sum(a * b, axis=1))



class HfEncoder(EncoderProtocol):
    """The full-precision checkpoint through sentence-transformers - the published-baseline engine."""

    mteb_model_meta = None

    def __init__(self, model_dir, threads=14, chunk=32, max_seq=2048):
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(threads)
        self.m = SentenceTransformer(str(model_dir), trust_remote_code=True, device="cpu")
        try:
            self.m.max_seq_length = max_seq
        except Exception:
            pass
        self.max_seq = max_seq
        self.chunk = chunk

    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None,
               prompt_type=None, **kwargs):
        texts = strings_for(inputs, task_metadata, prompt_type)
        return np.asarray(self.m.encode(texts, normalize_embeddings=True, batch_size=self.chunk,
                                        show_progress_bar=False), dtype=np.float32)

    def similarity(self, a, b):
        a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
        return np.asarray(a @ b.T)

    def similarity_pairwise(self, a, b):
        a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
        return np.asarray(np.sum(a * b, axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=Path.home() / "Projects/embeddinggemma-2")
    ap.add_argument("--xml", type=Path, default=None)
    ap.add_argument("--cli", type=Path, default=HERE / "build-perf" / "e2_embed_cli")
    ap.add_argument("--tasks", default="STSBenchmark")
    ap.add_argument("--engines", default="e2")
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    import mteb

    tasks = mteb.get_tasks(tasks=[t.strip() for t in args.tasks.split(",") if t.strip()])
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    for name in engines:
        if name == "e2":
            model = E2Encoder(args.model_dir, args.cli, args.threads)
        elif name == "hf":
            model = HfEncoder(args.model_dir, args.threads)
        elif name == "ov":
            xml = args.xml or (args.model_dir / "onnx" / "ov" / "text_int8.xml")
            model = OvEncoder(xml, args.model_dir, args.threads)
        else:
            raise SystemExit("unknown engine %r" % name)
        for task in tasks:
            res = mteb.evaluate(model=model, tasks=task,
                                encode_kwargs={"batch_size": args.batch_size},
                                raise_error=True, overwrite_strategy="always")
            events = getattr(res, "task_results", [res])
            for ev in events:
                tname = getattr(getattr(ev, "mteb_task_metadata", None), "name", None) or task.metadata.name
                print("RESULT %s %s %s" % (name, tname, json.dumps(_leaves(getattr(ev, "scores", None)),
                                                                 default=str)))
                sys.stdout.flush()


if __name__ == "__main__":
    main()
