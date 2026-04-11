from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmark_selective_mixing import (
    compute_aggregate_summary,
    discover_profile_status,
    enumerate_benchmark_pairs,
)


class BenchmarkSelectiveMixingTests(unittest.TestCase):
    def test_enumerate_benchmark_pairs_returns_all_filtered_pairs(self) -> None:
        entries = [
            {"id": "character_1", "category": "character"},
            {"id": "object_1", "category": "object"},
            {"id": "style_1", "category": "style"},
        ]
        pairs = enumerate_benchmark_pairs(entries, selected_categories=["character", "object", "style"])
        self.assertEqual([row["combination_id"] for row in pairs], [
            "character_1__object_1",
            "character_1__style_1",
            "object_1__style_1",
        ])

    def test_enumerate_benchmark_pairs_can_limit_to_character_vs_other(self) -> None:
        entries = [
            {"id": "character_1", "category": "character"},
            {"id": "character_2", "category": "character"},
            {"id": "object_1", "category": "object"},
            {"id": "style_1", "category": "style"},
        ]
        pairs = enumerate_benchmark_pairs(entries, pair_mode="character_vs_other")
        self.assertEqual([row["combination_id"] for row in pairs], [
            "character_1__object_1",
            "character_1__style_1",
            "character_2__object_1",
            "character_2__style_1",
        ])

    def test_discover_profile_status_distinguishes_present_and_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            present_dir = root / "character_3"
            present_dir.mkdir(parents=True)
            (present_dir / "lora_profile.json").write_text("{}", encoding="utf-8")

            present = discover_profile_status(root, "character_3")
            missing = discover_profile_status(root, "object_2")

            self.assertTrue(present["exists"])
            self.assertFalse(missing["exists"])

    def test_compute_aggregate_summary_tracks_wins_and_skips(self) -> None:
        pair_results = [
            {
                "combination_id": "a__b",
                "category_pair": "character+object",
                "status": "ok",
                "method_summaries": {
                    "merge": {"mean_retention": 0.8, "min_retention": 0.7, "pairwise_concept_retention": 0.8, "mean_image_similarity_to_single": 0.6, "generic_quality_score": 0.5, "mean_semantic_specificity": 0.1, "mean_runtime_sec": 10.0, "runtime_overhead_vs_merge": 1.0},
                    "switch": {"mean_retention": 0.82, "min_retention": 0.72, "pairwise_concept_retention": 0.82, "mean_image_similarity_to_single": 0.61, "generic_quality_score": 0.49, "mean_semantic_specificity": 0.09, "mean_runtime_sec": 9.0, "runtime_overhead_vs_merge": 0.9},
                    "selective_module_step": {"mean_retention": 0.9, "min_retention": 0.8, "pairwise_concept_retention": 0.9, "mean_image_similarity_to_single": 0.7, "generic_quality_score": 0.52, "mean_semantic_specificity": 0.08, "mean_runtime_sec": 11.0, "runtime_overhead_vs_merge": 1.1},
                },
            },
            {
                "combination_id": "c__d",
                "category_pair": "character+style",
                "status": "skipped",
            },
        ]
        summary = compute_aggregate_summary(pair_results, methods=["merge", "switch", "selective_module_step"])
        self.assertEqual(summary["num_combinations_evaluated"], 1)
        self.assertEqual(summary["num_combinations_skipped"], 1)
        comparison = summary["comparisons"]["selective_module_step_vs_merge"]["mean_retention"]
        self.assertEqual(comparison["wins"], 1)
        self.assertEqual(comparison["losses"], 0)


if __name__ == "__main__":
    unittest.main()
