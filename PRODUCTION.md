# CPU release and deployment checklist

The CPU engine has a software release gate, not a blanket production certification. The GPU backends
are outside this checklist. No server, request queue or process supervisor is included.

## Run the software gate

```sh
tools/check_cpu.sh /absolute/path/to/embeddinggemma-2
```

This requires the checkpoint **and** tokenizer; it refuses a checkpoint-free release check. It runs:

* CPU CTest at 1 and 4 threads, including frozen text/vision/audio references, batch masking,
  allocation failures, malformed safetensors, incomplete caches, numeric boundaries and lifecycle.
* Whole-text inference through scalar bf16, AVX2 and every VNNI backend supported by the machine.
  Unsupported instruction sets are reported, not emulated.
* Independent FP32 audio references at 1/12/13 and 743/744/745/749/750 soft tokens (up to 30 seconds).
* bf16 fidelity checks, then ASan/UBSan with the dependency-free image decoder.
* Export/read-back verification of all four packed tower sets, SHA256 verification, and inference
  from the all-tower cache. Artifacts remain under the ignored `build-release-check/` directory.

`.github/workflows/cpu.yml` runs checkpoint-free GCC/Clang and sanitizer checks on hosted runners.
The manually dispatched checkpoint job runs this full gate on a **trusted** self-hosted Linux x64
runner. Set repository variable `E2_MODEL_DIR` to that runner's checkpoint directory. Do not expose
that runner to untrusted pull-request jobs.

Before a release, save the logs, compiler version, CPU model/flags, codec versions, checkpoint hashes
and git revision. Run on the CPU families you will ship: AVX2-only, AVX-VNNI, and AVX512-VNNI where
claimed. A successful run on one laptop does not validate an unavailable ISA or another operating
system.

AVX512-VNNI CPU validation passed on an **AMD Ryzen 7 9700X (Zen 5)**, Linux x86-64, GCC 14.2.0,
at revision `6cf862d`. The full software gate passed: 24/24 tests at 1 and 4 threads, 24/24 under
ASan/UBSan, bf16 fidelity checks, and export/read-back of all four packed variants. An additional
8-thread run passed 24/24; automatic sizing selected the eight physical cores. The 42-case text
corpus scored minimum cosine **0.999432190**, with all **8/8** retrieval rankings preserved.
This validates the CPU AVX512 path on AMD Zen 5, not Intel Xeon coverage or the AMD HIP backend.

## Fidelity is an acceptance decision

The existing short text/image/audio fixtures retain their **0.999 cosine** gate. The text golden
corpus also checks eight retrieval rankings. These are regression tests, not a production retrieval
benchmark.

The new audio corpus deliberately uses steady tones, including five clips near 30 seconds. On the
reference CPU, int8 minimum cosine against independent FP32 is approximately **0.987**; bf16 is
**0.99998**. Its explicit regression floors are 0.98 for int8 and 0.999 for bf16. This is a disclosed
quantization limitation, **not** a claim that 0.98 is an acceptable application-quality target.
The independent oracle and short-fixture gates have not been relaxed to cover it.

Before accepting int8 audio, evaluate real speech/music/noise/silence from your workload. For retrieval,
compare FP32 and e2 with identical prompts, tokenization/truncation, corpus and query set; record
Recall@k/nDCG@10 and ranking disagreements. Set the acceptable loss before measuring. The existing
`tools/mteb_e2.py` can run public retrieval/STS tasks, but does not replace your application dataset.
If the observed loss is unacceptable, use the original checkpoint with `E2_INT8=0` (slower, no packed
cache), or do not enable audio. Improving the int8 recipe requires a separate measured calibration
change; a passing memory-safety gate is not evidence of that quality.

## Process and concurrency contract

* One live model per process. Serialize **all** load/free/embed/preprocess calls behind one request
  queue or application mutex. Tokenizers own scratch; error reporting and preprocessing also share
  process state. Separate handles/tokenizers do not make concurrent calls safe.
* Use a bounded queue and a fixed number of worker processes. Choose batch size, queue depth and
  timeout from measured peak memory and p95/p99 latency under sustained load. The C API has no
  cancellation mechanism: apply deadlines between calls, or terminate/restart an isolated worker.
* Start worker processes before `e2_load`; do not fork a live worker pool. Free the model only after
  draining work. Destroy any GPU/HIP contexts before freeing their source model.
* Set `E2_THREADS` explicitly for each worker's CPU allocation. Automatic sizing now limits itself
  to Linux affinity and visible cgroup-v2 `cpu.max` quotas (including visible ancestors). cgroup-v1
  quotas and hidden host limits require an explicit setting. An explicit override is intentional and
  is not silently reduced. Multiple worker processes must not each claim the entire CPU budget.

## Input and output contract

* Enforce a request-byte budget before tokenization or image decoding. The engine is not an HTTP
  upload limiter. `e2_encode*` does not truncate; reserve its documented worst-case ID capacity or
  reject overly large text. The embedding sequence limit is 2048 IDs.
* Validate PCM sampling rate/channel layout in the caller: 16 kHz mono, finite values in [-1,1].
  The engine retains the reference's truncation to the first 30 seconds. Pre-patchified pixels,
  including supplied padding, must be finite and in [0,1]. NaN/Inf and out-of-range inputs fail.
* Decoder limits are a last defense, not a recommended request budget: at most 2^28 decoded pixels;
  PNG/RGB also limit either side to 2^20. A legal image can still require hundreds of MB. Apply much
  smaller input dimensions/bytes in an untrusted service, and provision a worker memory limit.
* Check every return value. On failure, discard the whole result/batch; runtime failure may occur
  after earlier batch members were written. Input validation failures leave embedding output alone.
  Successful nonempty embeddings are checked for finite values and unit norm outside fast-math;
  all-masked batch rows remain exact zero vectors by contract.
* `e2_last_error()` describes the last fallible load/encode/preprocess/embed call and is cleared on
  success. Free/info/pure size queries leave it unchanged. Copy/log it before another call.

## Model publication and startup

* `download_model.sh` pins the two original files by SHA256 and resumes into `.part` files. A changed
  upstream file fails verification; update the pins only with a deliberate reference/fidelity review.
* Packed downloads require `manifest.sha256`; there is no unverified size-only fallback. Export to a
  **new release directory**, verify all four variants and their tokenizer, then make that directory
  read-only. Preserve a trusted copy of its manifest; a hash fetched from the same mutable URL gives
  transport integrity, not independent provenance or authenticity.
* Cache writes use a sibling temporary file, flush/fsync it, and rename only on successful close.
  Failed loads abort the write. Export/download scripts stage and verify replacements before
  publishing. Do not modify/truncate a mapped artifact in place: mmap cannot protect against an
  external writer. Atomically switch the deployment's release-directory pointer and start new workers;
  do not hot-reload a live handle. Power-loss durability of the directory rename depends on the host
  filesystem; the engine does not fsync the containing directory.
* Warm every enabled modality before readiness, including a representative maximum shape if that is
  allowed by the service. Audio builds weights/scratch lazily. Check known canary embeddings, unit
  norms, actual resident memory and latency; only then accept requests. A text-only cache should not
  advertise image/audio endpoints.
* Record failed calls, queue saturation, restarts, RSS, latency percentiles and canary failures in the
  host service. Keep `E2_DUMP` disabled in production: it writes user-derived activations to disk.

Remaining deployment-specific acceptance: real workload quality, sustained-load/resource budgets and
hardware/OS coverage. The repository cannot choose those thresholds without the application's data.
