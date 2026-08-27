from __future__ import annotations

import unittest

from benchmarks.qwen3_runtime import (
    ACTIVE_CONFIG_RELATIVE,
    REPOSITORY_ROOT,
    load_active_runtime,
)
from benchmarks.run_dpp_ttft_weight_grid import (
    as_smoke,
    load_settings,
    matrix,
    preview,
)


class DppTtftWeightGridTests(unittest.TestCase):
    def test_fixed_nine_run_matrix(self) -> None:
        runtime = load_active_runtime(REPOSITORY_ROOT / ACTIVE_CONFIG_RELATIVE)
        settings = load_settings(runtime)
        runs = matrix(settings)
        self.assertEqual(len(runs), 9)
        self.assertEqual(runs[0].policy, "stock")
        self.assertEqual(runs[1].selection_mode, "forced_stock_plan")
        self.assertEqual(
            tuple(run.weight for run in runs[2:]),
            (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
        )
        resolved = preview(runtime, settings)
        self.assertEqual(resolved["qps"], 0.25)
        self.assertEqual(resolved["seed"], 1001)
        self.assertEqual(resolved["request_count_per_run"], 150)
        self.assertTrue(resolved["shared_stock_run"])

    def test_isolated_three_path_smoke_matrix(self) -> None:
        runtime = load_active_runtime(REPOSITORY_ROOT / ACTIVE_CONFIG_RELATIVE)
        settings = as_smoke(load_settings(runtime))
        runs = matrix(settings)
        self.assertEqual(settings.request_count, 1)
        self.assertEqual(len(runs), 3)
        self.assertEqual(
            [(run.policy, run.selection_mode, run.weight) for run in runs],
            [
                ("stock", "normal", None),
                ("dpp", "forced_stock_plan", None),
                ("dpp", "normal", 1.0),
            ],
        )
        resolved = preview(runtime, settings)
        self.assertTrue(resolved["smoke"])
        self.assertNotEqual(
            settings.campaign_id, load_settings(runtime).campaign_id
        )


if __name__ == "__main__":
    unittest.main()
