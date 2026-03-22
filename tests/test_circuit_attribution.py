from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

if torch is not None:
    from circuit_ap.attribution import (
        FullRunPatchObserver,
        StepTraceObserver,
        activation_direction_recovery,
        advance_latents_with_scheduler,
        build_cross_step_edge_frontier,
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

    @unittest.skipIf(torch is None, "torch is required for circuit attribution tests")
    def test_cross_step_edge_frontier_uses_adjacent_steps(self) -> None:
        positive_nodes = [
            {"node": "a@0", "module_path": "a", "step_index": 0, "median_score": 3.0},
            {"node": "b@0", "module_path": "b", "step_index": 0, "median_score": 2.0},
            {"node": "c@1", "module_path": "c", "step_index": 1, "median_score": 4.0},
            {"node": "d@1", "module_path": "d", "step_index": 1, "median_score": 1.0},
            {"node": "e@2", "module_path": "e", "step_index": 2, "median_score": 5.0},
        ]
        frontier = build_cross_step_edge_frontier(
            positive_nodes,
            source_topk=1,
            target_topk=1,
            max_step_delta=1,
            denoise_steps=3,
        )
        self.assertEqual(len(frontier), 2)
        self.assertEqual(frontier[0]["source_step"], 0)
        self.assertEqual(frontier[0]["target_step"], 1)
        self.assertEqual(frontier[0]["source_paths"], ["a"])
        self.assertEqual(frontier[0]["target_paths"], ["c"])
        self.assertEqual(frontier[1]["source_step"], 1)
        self.assertEqual(frontier[1]["target_step"], 2)

    @unittest.skipIf(torch is None, "torch is required for circuit attribution tests")
    def test_advance_latents_with_scheduler_uses_step_trace_latents(self) -> None:
        class _FakeScheduler:
            def step(self, noise_pred, timestep, latents, **kwargs):
                return (latents + noise_pred + timestep.float().view(1, 1),)

        class _FakePipeline:
            def __init__(self) -> None:
                self._execution_device = torch.device("cpu")
                self.unet = torch.nn.Linear(1, 1, bias=False)
                self.scheduler = _FakeScheduler()

            def prepare_extra_step_kwargs(self, generator, eta):
                return {}

        pipeline = _FakePipeline()
        trace = StepTrace(
            step_index=0,
            timestep=2,
            latents=torch.tensor([[1.0]]),
            latent_model_input=torch.tensor([[1.0]]),
            prompt_embeds=torch.tensor([[0.0]]),
            timestep_cond=None,
            added_cond_kwargs=None,
            cross_attention_kwargs=None,
            guidance_scale=1.0,
            guidance_rescale=0.0,
            do_classifier_free_guidance=False,
            scheduler_state=pipeline.scheduler,
        )
        advanced = advance_latents_with_scheduler(
            pipeline,
            step_trace=trace,
            noise_pred=torch.tensor([[3.0]]),
        )
        self.assertTrue(torch.equal(advanced, torch.tensor([[6.0]])))


if __name__ == "__main__":
    unittest.main()
