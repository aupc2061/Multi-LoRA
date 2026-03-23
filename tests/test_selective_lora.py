from __future__ import annotations

import unittest

from selective_lora import build_profile_mask, build_selective_policy


class SelectiveLoRATests(unittest.TestCase):
    def test_build_profile_mask_union_keeps_top_module_or_step_nodes(self) -> None:
        profile = {
            "module_scores": {"a": 3.0, "b": 2.0},
            "step_scores": {"0": 5.0, "1": 1.0},
            "module_step_scores": {"a@0": 2.0, "a@1": 1.0, "b@1": 0.5},
        }
        mask = build_profile_mask(profile, top_modules=1, top_steps=1, support_mode="union")
        self.assertEqual(mask["selected_modules"], ["a"])
        self.assertEqual(mask["selected_steps"], [0])
        self.assertIn("a@0", mask["active_nodes"])
        self.assertIn("a@1", mask["active_nodes"])
        self.assertNotIn("b@1", mask["active_nodes"])

    def test_build_selective_policy_resolves_overlap_by_weighted_score(self) -> None:
        profiles = {
            "lora_a": {
                "module_scores": {"m": 3.0},
                "step_scores": {"0": 2.0},
                "module_step_scores": {"m@0": 3.0},
            },
            "lora_b": {
                "module_scores": {"m": 2.0},
                "step_scores": {"0": 2.0},
                "module_step_scores": {"m@0": 2.0},
            },
        }
        policy = build_selective_policy(
            ["lora_a", "lora_b"],
            profiles,
            lora_weights={"lora_a": 1.0, "lora_b": 1.0},
            denoise_steps=1,
            top_modules=1,
            top_steps=1,
            support_mode="union",
        )
        self.assertEqual(policy["module_step_assignments"][0]["m"], ["lora_a"])
        self.assertEqual(sorted(policy["timestep_schedule"][0]), ["lora_a", "lora_b"])


if __name__ == "__main__":
    unittest.main()
