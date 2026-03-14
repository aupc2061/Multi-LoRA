from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

if torch is not None:
    from circuit_attribution import (
        FullRunPatchObserver,
        StepTraceObserver,
        activation_direction_recovery,
        directional_recovery,
    )


if torch is not None:
    class _ToyModule(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1


class CircuitAttributionTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is required for circuit attribution tests")
    def test_directional_recovery_zero_when_no_change(self) -> None:
        corr = torch.tensor([[1.0, 2.0]])
        direction = torch.tensor([[0.5, -0.5]])
        self.assertAlmostEqual(directional_recovery(corr, corr, direction), 0.0, places=6)

    @unittest.skipIf(torch is None, "torch is required for circuit attribution tests")
    def test_activation_direction_recovery_positive_on_target_delta(self) -> None:
        corr = torch.tensor([[0.0, 0.0]])
        delta = torch.tensor([[2.0, 0.0]])
        patched = torch.tensor([[1.0, 0.0]])
        self.assertGreater(activation_direction_recovery(patched, corr, delta), 0.0)

    @unittest.skipIf(torch is None, "torch is required for circuit attribution tests")
    def test_step_trace_observer_only_records_target_step(self) -> None:
        module = _ToyModule()
        observer = StepTraceObserver({"toy": module}, target_steps={2})
        try:
            observer.start_step(
                step_index=1,
                timestep=10,
                latent_model_input=torch.zeros(1, 1),
                latents=torch.zeros(1, 1),
                prompt_embeds=torch.zeros(1, 1),
                timestep_cond=None,
                added_cond_kwargs=None,
                cross_attention_kwargs=None,
                guidance_scale=1.0,
                guidance_rescale=0.0,
                do_classifier_free_guidance=False,
            )
            module(torch.zeros(1, 1))
            observer.end_step(step_index=1, timestep=10, noise_pred=torch.zeros(1, 1))

            observer.start_step(
                step_index=2,
                timestep=20,
                latent_model_input=torch.zeros(1, 1),
                latents=torch.zeros(1, 1),
                prompt_embeds=torch.zeros(1, 1),
                timestep_cond=None,
                added_cond_kwargs=None,
                cross_attention_kwargs=None,
                guidance_scale=1.0,
                guidance_rescale=0.0,
                do_classifier_free_guidance=False,
            )
            module(torch.zeros(1, 1))
            observer.end_step(step_index=2, timestep=20, noise_pred=torch.ones(1, 1))
        finally:
            observer.close()

        self.assertNotIn(1, observer.records)
        self.assertIn(2, observer.records)
        self.assertEqual(observer.records[2].execution_order, ["toy"])

    @unittest.skipIf(torch is None, "torch is required for circuit attribution tests")
    def test_full_run_patch_observer_replaces_output(self) -> None:
        module = _ToyModule()
        observer = FullRunPatchObserver({"toy": module}, {0: {"toy": torch.zeros(1, 1)}})
        try:
            observer.start_step(step_index=0)
            out = module(torch.ones(1, 1))
            observer.end_step()
        finally:
            observer.close()
        self.assertTrue(torch.equal(out, torch.zeros(1, 1)))


if __name__ == "__main__":
    unittest.main()
