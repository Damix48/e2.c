"""No runtime/model dependencies beyond NumPy: python -m unittest discover -s tests -p test_bench_compare.py."""
import importlib.util
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("bench_compare", Path(__file__).resolve().parents[1] / "tools/bench/bench_compare.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class BenchmarkChecks(unittest.TestCase):
    def test_metrics_reject_invalid_outputs(self):
        ref = np.zeros((1, 768), dtype=np.float32)
        ref[0, 0] = 1
        self.assertAlmostEqual(bench.check(ref, ref), 1)
        for bad in (np.zeros_like(ref), -ref, ref * 2, np.full_like(ref, np.nan),
                    np.full_like(ref, np.inf), ref[:, :767]):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                bench.check(bad, ref)

    def test_each_batch_row_is_checked(self):
        ref = np.zeros((2, 768), np.float32)
        ref[:, 0] = 1
        bad = ref.copy()
        bad[1] *= -1
        with self.assertRaises(ValueError):
            bench.check(bad, ref)

    def test_statistics(self):
        self.assertEqual(bench.summary([4, 1, 3, 2]), {"median_ms": 2.5, "min_ms": 1, "p95_ms": 4})
        for samples in ([], [0], [float("nan")], [-1]):
            with self.assertRaises(ValueError):
                bench.summary(samples)

    def test_fixtures_and_worker_input(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ref = np.zeros(768, dtype="<f4")
            ref[0] = 1
            path = root / "text_query.bin"
            path.write_bytes(struct.pack("<4i", 2, 768, 2, 1) + ref.tobytes())
            case = bench.prepare_case("query_batch8", root, root, root)
            self.assertEqual((case["batch"], case["length"]), (8, 2))
            data = np.load(root / "query_batch8.npz")
            np.testing.assert_array_equal(data["input_ids"], np.tile([2, 1], (8, 1)))
            self.assertEqual(data["input_ids"].dtype, np.int64)
            worker = (root / "query_batch8.bin").read_bytes()
            self.assertEqual(struct.unpack("<3i", worker[:12]), (0, 8, 2))
            self.assertEqual(len(worker), 12 + 8 * 2 * 4 + 8 * 768 * 4)
            for broken in (path.read_bytes()[:-1], path.read_bytes() + b"x", struct.pack("<2i", -1, 768)):
                path.write_bytes(broken)
                with self.assertRaises(ValueError):
                    bench.prepare_case("query", root, root, root)

    def test_failed_round_is_not_a_speedup(self):
        report = {"started": "test", "cases": [{"name": "query", "boundary": "IDs"}],
                  "devices": ["CPU"], "rounds": 1, "results": [
                      {"case": "query", "device": "CPU", "engine": "e2", "status": "failed", "median_ms": 1},
                      {"case": "query", "device": "CPU", "engine": "ov", "status": "ok", "median_ms": 10}]}
        rendered = bench.render(report)
        self.assertNotIn("10.00x", rendered)
        self.assertIn("FAILED", rendered)

    def test_c_worker_rejects_malformed_inputs(self):
        worker = bench.HERE / "build-compare/e2_compare_bench"
        if not worker.exists():
            self.skipTest("build e2_compare_bench to check its input boundary")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "bad.bin"
            for payload in (b"", struct.pack("<3i", 0, 1, -1), struct.pack("<3i", 0, 33, 2),
                            struct.pack("<3i", 1, 1, 2521), struct.pack("<3i", 2, 1, 480001),
                            struct.pack("<3i", 0, 1, 2) + b"x"):
                path.write_bytes(payload)
                result = subprocess.run([str(worker), str(root / "no-model"), "cpu", str(path),
                                         "1", "0", str(root / "out.f32")], capture_output=True)
                self.assertEqual(result.returncode, 2, result.stderr.decode())

    def test_checked_in_fixtures(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in bench.CASES[:-1]:
                case = bench.prepare_case(name, bench.HERE / "tests/fixtures", root, root)
                self.assertEqual(case["name"], name)
            pcm, mel, mask, n = bench.audio_fixture(bench.HERE / "tests/fixtures/audio.bin")
            self.assertEqual(len(pcm), 12000)
            self.assertEqual(mel.shape, (1, mask.shape[1], 128))
            self.assertEqual(n, 19)


if __name__ == "__main__":
    unittest.main()
