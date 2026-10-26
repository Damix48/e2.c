# tools/

Four kinds of things live here. Nothing in `src/` or the test suite depends on any of them at run
time.

| path | what | needs |
| :-- | :-- | :-- |
| `bench/bench_compare.py` | A/B harness: builds the C worker, runs `e2` and OpenVINO INT8 serially on frozen inputs, writes `benchmark-results/<stamp>/` + `REPORT.md` | numpy, openvino, nncf, torch; runbook in `docs/llms/BENCHMARKS.md` (local, not published) |
| `bench/bench_audio_ov.py` | audio-only comparison against OpenVINO | numpy, openvino, torch |
| `bench/*.c`, `bench/*.hip` | benchmark and probe workers built by CMake (`e2_text_bench`, `e2_compare_bench`, `e2_gpu_bench`, `e2_hip_bench`, `e2_hip_kernel_bench`, `l0_device_id`) | a compiler; GPU/HIP ones need the backend enabled |
| `embed_cli.c` | the CLI: text/image/audio -> 768-d vector (`e2_embed_cli`) | the model directory |
| `export_cache.c` | load a checkpoint and write the packed int8 cache (`e2_export_cache`); which towers is `E2_Q8_TOWERS` | the model directory |
| `export_models.sh` | writes the four published packed models + `manifest.txt`, then reads each one back and runs the fixture test for its shape | a built `e2_export_cache` and the fixture tests |
| `mteb_e2.py` | run MTEB retrieval/STS tasks through `e2_embed_cli` (and optionally OpenVINO) | mteb, the model directory |
| `dump_*.py` | fixture generators — they *wrote* `tests/fixtures/*.bin` and `tests/ref_act/*.f32` | PyTorch, transformers, the checkpoint; four of them also need `bench/fixtures/*.npz` and `scripts/onnx_common.py` from the model workspace |

The generators are development-time only: their outputs are committed (`tests/fixtures/`) or ignored
(`tests/ref_act/`), so a normal build and `ctest` never run them. Each script's docstring states what
it writes.
