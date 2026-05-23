"""SD3.5 null-mask inference engine.

Two-component LoRA composition:
  1. Lateral interference (spatial): cross-attention look-ahead → argmax binary masks
     → residual blending confined to each LoRA's owned patch region.
  2. Vertical interference (structural): SVD of base model's runtime hidden states at a
     chosen transformer block → project each LoRA's hidden-state delta into the orthogonal
     complement of the base model's principal activation directions.
"""
from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TQDM_AVAILABLE = False

from .config import SD35NullMaskConfig
from .masking import MaskBuildResult, build_exclusive_binary_masks, reduce_attention_to_patch_scores
from .projection import compute_structural_basis, project_delta_to_nullspace

if TYPE_CHECKING:
    from .inventory import InventoryAdapter


# ── hooks ──────────────────────────────────────────────────────────────────────

class AttentionCaptureHook:
    """Pre-forward hook on a JointAttention module.

    Recomputes image→text cross-attention from the module's own Q (image) and K (text)
    projections. No RoPE applied; relative ordering is preserved, which is sufficient
    for mask quality.
    """

    def __init__(self, attn_module: Any, n_img_tokens: int) -> None:
        self.n_img_tokens = n_img_tokens
        self.last_attn_map: Optional[torch.Tensor] = None  # [N_img, N_txt]
        self._handle = attn_module.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module: Any, args: tuple, kwargs: dict) -> None:
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None:
            hidden_states = args[0] if args else None
        encoder_states = kwargs.get("encoder_hidden_states")
        if encoder_states is None:
            encoder_states = args[1] if len(args) > 1 else None
        if hidden_states is None or encoder_states is None:
            return
        with torch.no_grad():
            try:
                q = module.to_q(hidden_states)
                if hasattr(module, "norm_q"):
                    q = module.norm_q(q)

                # Try known attribute names for text-key projection
                k_txt = None
                for attr in ("add_k_proj", "to_add_k", "add_k"):
                    if hasattr(module, attr):
                        k_txt = getattr(module, attr)(encoder_states)
                        break
                if k_txt is None:
                    return

                for norm_attr in ("norm_added_k", "norm_k_encoder", "norm_encoder_k"):
                    if hasattr(module, norm_attr):
                        k_txt = getattr(module, norm_attr)(k_txt)
                        break

                n_heads: int = module.heads
                B, N_img, D = q.shape
                head_dim = D // n_heads

                q = q.view(B, N_img, n_heads, head_dim).permute(0, 2, 1, 3)
                k_txt = k_txt.view(B, -1, n_heads, head_dim).permute(0, 2, 1, 3)

                scale = head_dim ** -0.5
                attn = torch.softmax(
                    q.float() @ k_txt.float().transpose(-1, -2) * scale, dim=-1
                )  # [B, H, N_img, N_txt]
                self.last_attn_map = attn.mean(dim=(0, 1)).detach()  # [N_img, N_txt]
            except Exception:
                pass  # fail silently; caller checks for None

    def remove(self) -> None:
        self._handle.remove()


class HiddenStateCaptureHook:
    """Output hook on a JointTransformerBlock.

    SD3 convention: block returns (encoder_hidden_states, hidden_states).
    We capture both for later use in injection.
    """

    def __init__(self, block: Any) -> None:
        self.last_hidden: Optional[torch.Tensor] = None      # image tokens [B, N_img, C]
        self.last_enc_hidden: Optional[torch.Tensor] = None  # text tokens  [B, N_txt, C]
        self._handle = block.register_forward_hook(self._hook)

    def _hook(self, module: Any, input: Any, output: Any) -> None:
        enc_hidden, hidden = _unpack_block_output(output)
        self.last_hidden = hidden.detach().clone()
        # enc_hidden is None for context_pre_only blocks (e.g. last block in SD3.5-large)
        self.last_enc_hidden = enc_hidden.detach().clone() if enc_hidden is not None else None

    def remove(self) -> None:
        self._handle.remove()


class HiddenStateInjectionHook:
    """Output hook on a JointTransformerBlock that replaces its output.

    Set `inject_hidden` and optionally `inject_enc_hidden` before the forward pass.
    The hook replaces the block's output with these tensors so that all subsequent
    transformer operations (norm_out, proj_out, unpatchify) see the blended state.
    """

    def __init__(self, block: Any) -> None:
        self.inject_hidden: Optional[torch.Tensor] = None      # [B, N_img, C]
        self.inject_enc_hidden: Optional[torch.Tensor] = None  # [B, N_txt, C]
        self._handle = block.register_forward_hook(self._hook)

    def _hook(self, module: Any, input: Any, output: Any) -> Optional[tuple]:
        if self.inject_hidden is None:
            return None  # pass through
        enc_hidden, hidden = _unpack_block_output(output)
        new_hidden = self.inject_hidden.to(dtype=hidden.dtype, device=hidden.device)
        # enc_hidden is None for context_pre_only blocks — preserve that
        if enc_hidden is None:
            new_enc = None
        elif self.inject_enc_hidden is not None:
            new_enc = self.inject_enc_hidden.to(dtype=enc_hidden.dtype, device=enc_hidden.device)
        else:
            new_enc = enc_hidden
        return _pack_block_output(new_enc, new_hidden, output)

    def reset(self) -> None:
        self.inject_hidden = None
        self.inject_enc_hidden = None

    def remove(self) -> None:
        self._handle.remove()


def _unpack_block_output(output: Any) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    """Return (encoder_hidden_states, hidden_states) regardless of output format.

    Handles three cases:
    - Normal JointTransformerBlock: returns (encoder_hidden_states, hidden_states) 2-tuple.
    - context_pre_only block (last block in SD3.5-large): returns a bare hidden_states tensor
      with no encoder states — returns (None, hidden_states).
    - Rare 1-tuple: treated same as bare tensor (enc → None).
    """
    if isinstance(output, torch.Tensor):
        # context_pre_only block returns a bare tensor (image hidden states only)
        return None, output
    if isinstance(output, (tuple, list)):
        if len(output) == 1:
            return None, output[0]
        if len(output) == 2:
            a, b = output
            return a, b
    raise ValueError(
        f"Unexpected JointTransformerBlock output: type={type(output)}, "
        f"len={len(output) if hasattr(output, '__len__') else 'N/A'}"
    )


def _pack_block_output(
    enc_hidden: Optional[torch.Tensor], hidden: torch.Tensor, reference: Any
) -> Any:
    """Return packed output matching the container format of reference.

    - Bare tensor reference (context_pre_only block): return bare hidden tensor only.
    - 1-tuple reference: return 1-tuple.
    - 2-tuple/list reference: return (enc_hidden, hidden) in same container type.
    """
    if isinstance(reference, torch.Tensor):
        # context_pre_only block — diffusers expects bare tensor back
        return hidden
    if isinstance(reference, (tuple, list)) and len(reference) == 1:
        return (hidden,) if isinstance(reference, tuple) else [hidden]
    if isinstance(reference, list):
        return [enc_hidden, hidden]
    return (enc_hidden, hidden)


# ── token index extraction ──────────────────────────────────────────────────────

def get_concept_token_indices(
    tokenizer: Any, full_prompt: str, trigger_phrase: str
) -> list[int]:
    """Return CLIP-L token indices in full_prompt that match trigger_phrase.

    CLIP-L tokens occupy [0, 77) in the concatenated SD3 text embedding sequence,
    so the returned indices are directly usable when the joint attention is over
    the full concatenated token sequence.
    """
    MAX_LEN = 77

    full_ids: list[int] = tokenizer(
        full_prompt,
        padding="max_length",
        max_length=MAX_LEN,
        truncation=True,
        return_tensors="pt",
    ).input_ids[0].tolist()

    trigger_ids: list[int] = tokenizer(
        trigger_phrase,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0].tolist()

    n = len(trigger_ids)
    for start in range(len(full_ids) - n + 1):
        if full_ids[start : start + n] == trigger_ids:
            return list(range(start, start + n))

    warnings.warn(
        f"Trigger '{trigger_phrase}' not found in prompt tokenization. Using index [1].",
        stacklevel=2,
    )
    return [1]


# ── attention-score smoothing ───────────────────────────────────────────────────

def _gaussian_kernel_3x3(device: torch.device) -> torch.Tensor:
    k = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32, device=device) / 16.0
    return k.view(1, 1, 3, 3)


def smooth_and_binarize(
    scores: torch.Tensor,
    *,
    n_blur_iter: int = 5,
    binarize_tau: float = 0.7,
) -> torch.Tensor:
    """Apply iterative Gaussian smoothing then threshold at the τ quantile.

    Matches LoRA-Shop §3.2: blur until connected, then threshold at τ posterior quantile.
    Returns float scores in [0, 1] (post-smoothing, pre-argmax); binarization for
    the argmax partition is handled by `build_exclusive_binary_masks`.
    """
    N = scores.shape[0]
    H = W = int(math.isqrt(N))
    if H * W != N:
        # Non-square patch grid — skip spatial smoothing
        return scores.float()

    kernel = _gaussian_kernel_3x3(scores.device)
    x = scores.float().view(1, 1, H, W)
    for _ in range(n_blur_iter):
        x = F.conv2d(F.pad(x, [1, 1, 1, 1], mode="reflect"), kernel)
        mn, mx = x.min(), x.max()
        if mx > mn:
            x = (x - mn) / (mx - mn)

    flat = x.view(-1)
    # Apply τ threshold: zero out patches below the τ quantile
    if 0.0 < binarize_tau < 1.0:
        tau_val = float(flat.quantile(binarize_tau))
        flat = flat * (flat >= tau_val).float()
    return flat


# ── soft null projection ────────────────────────────────────────────────────────

def project_delta_soft(
    delta: torch.Tensor,
    basis: Any,
    mu: float,
) -> torch.Tensor:
    """NP-LoRA soft projection: P_soft = I - (mu/(1+mu)) * V_k @ V_k^T.

    mu=-1 (sentinel) → hard projection (fully remove component in principal subspace).
    mu=0 → no projection (pass-through).
    """
    if basis is None or basis.vectors.numel() == 0:
        return delta
    if mu == 0.0:
        return delta
    if mu < 0:
        return project_delta_to_nullspace(delta, basis)
    # Soft: attenuate by factor mu/(1+mu) in the principal subspace
    flat = delta.reshape(-1, delta.shape[-1]).float()
    # Bug 7: move basis vectors to the same device/dtype as delta before matmul
    v = basis.vectors.to(device=flat.device, dtype=flat.dtype)  # [C, rank]
    component = flat @ v @ v.transpose(0, 1)
    out = flat - (mu / (1.0 + mu)) * component
    return out.reshape_as(delta).to(dtype=delta.dtype, device=delta.device)


# ── inference engine ────────────────────────────────────────────────────────────

class SD35NullMaskEngine:
    """Inference engine for spatially-masked null-space LoRA composition on SD3.5.

    Usage:
        engine = SD35NullMaskEngine(pipeline, config, adapter_ids, triggers, prompt)
        image = engine.generate(method="sd35_mask_nullproj", seed=111)
    """

    def __init__(
        self,
        pipeline: Any,
        config: SD35NullMaskConfig,
        adapter_ids: list[str],
        triggers: list[str],
        prompt: str,
        negative_prompt: str = "",
    ) -> None:
        self.pipeline = pipeline
        self.config = config
        self.adapter_ids = adapter_ids
        self.triggers = triggers
        self.prompt = prompt
        self.negative_prompt = negative_prompt

        transformer = pipeline.transformer
        self.blocks = transformer.transformer_blocks
        self.n_blocks = len(self.blocks)

        # Image token count from transformer config.
        # sample_size is the latent spatial size in pixels (e.g. 128 for a 1024px image).
        # Divide by patch_size (2) to get the patch grid side length → 64x64 = 4096 tokens.
        try:
            sample_size = transformer.config.sample_size  # latent px, NOT patch count
            patch_size = int(getattr(transformer.config, "patch_size", 2) or 2)
            n_side = sample_size // patch_size
            self.n_img_tokens: int = n_side * n_side
        except Exception:
            patch_size = getattr(getattr(transformer, "config", None), "patch_size", 2) or 2
            self.n_img_tokens = (config.height // 8 // patch_size) * (
                config.width // 8 // patch_size
            )

        # Resolve intervention block index
        self.intervention_block_idx: int = (
            min(config.intervention_block_start, self.n_blocks - 1)
            if config.intervention_block_start >= 0
            else self.n_blocks - 1
        )

        # Look-ahead block: 80th-percentile of depth (analogous to LoRA-Shop Block 19)
        self.lookahead_block_idx: int = (
            self.intervention_block_idx
            if config.intervention_block_start >= 0
            else int(self.n_blocks * 0.8)
        )

    # ── public entry point ──────────────────────────────────────────────────────

    def generate(self, method: str, seed: int) -> Any:
        """Full generation for a given method and random seed. Returns PIL Image."""
        device = self.pipeline.device
        dtype = self.pipeline.transformer.dtype
        generator = torch.Generator(device=device).manual_seed(seed)

        prompt_embeds, negative_prompt_embeds, pooled_embeds, negative_pooled_embeds = (
            self._encode_prompts(device)
        )

        scheduler = self.pipeline.scheduler
        scheduler.set_timesteps(self.config.denoise_steps, device=device)
        # Explicit reset: some diffusers versions of FlowMatchEulerDiscreteScheduler
        # do NOT reset _step_index inside set_timesteps(). Without this, a second
        # generate() call on the same engine would crash immediately because _step_index
        # is still 28 (= denoise_steps) from the previous run, causing sigmas[29] OOB.
        scheduler._step_index = None
        timesteps = scheduler.timesteps

        latents = self._prepare_latents(generator, device, dtype)

        mask_result: Optional[MaskBuildResult] = None
        if method in ("sd35_mask_only", "sd35_mask_nullproj"):
            print(f"  [{method}] running look-ahead ({self.config.lookahead_steps[0]}–{self.config.lookahead_steps[-1]} steps)...", flush=True)
            mask_result = self._run_lookahead(
                latents.clone(),
                timesteps,
                prompt_embeds,
                pooled_embeds,
                seed=seed,
            )
            # Print mask ownership so users can verify the split is balanced
            for aid, ratio in mask_result.ownership_ratio_by_adapter.items():
                short_id = aid.split("_", 1)[-1]  # strip "character_" / "style_" prefix
                print(f"    mask {short_id}: {ratio:.1%} of patches", flush=True)
            if mask_result.unowned_ratio > 0.0:
                print(f"    unowned: {mask_result.unowned_ratio:.1%}", flush=True)

            # Lookahead called scheduler.step() advancing _step_index. Reset fully
            # so the main loop starts at index 0 with a clean sigma sequence.
            scheduler.set_timesteps(self.config.denoise_steps, device=device)
            scheduler._step_index = None  # same version guard as above
            timesteps = scheduler.timesteps

        # ── denoising loop ────────────────────────────────────────────────────
        _passes_per_step = (
            4 if method in ("sd35_mask_only", "sd35_mask_nullproj") else 1
        )
        _step_iter = (
            _tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                desc=f"  [{method}] seed={seed}",
                unit="step",
                leave=False,
                ncols=100,
            )
            if _TQDM_AVAILABLE
            else enumerate(timesteps)
        )

        for step_idx, t in _step_iter:
            noise_pred = self._denoising_step(
                latents, t, step_idx,
                prompt_embeds, negative_prompt_embeds,
                pooled_embeds, negative_pooled_embeds,
                mask_result, method,
            )
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        image = self.pipeline.vae.decode(
            latents / self.pipeline.vae.config.scaling_factor, return_dict=False
        )[0].detach()
        return self.pipeline.image_processor.postprocess(image, output_type="pil")[0]

    # ── prompt encoding ─────────────────────────────────────────────────────────

    def _encode_prompts(self, device: Any) -> tuple:
        """Encode prompt with all three SD3.5 text encoders."""
        try:
            return self.pipeline.encode_prompt(
                prompt=self.prompt,
                prompt_2=self.prompt,
                prompt_3=self.prompt,
                negative_prompt=self.negative_prompt or "",
                negative_prompt_2=self.negative_prompt or "",
                negative_prompt_3=self.negative_prompt or "",
                device=device,
            )
        except TypeError:
            # Older diffusers API without prompt_2/3 split
            return self.pipeline.encode_prompt(
                self.prompt,
                device=device,
                negative_prompt=self.negative_prompt or "",
            )

    # ── latent initialisation ───────────────────────────────────────────────────

    def _prepare_latents(
        self, generator: torch.Generator, device: Any, dtype: torch.dtype
    ) -> torch.Tensor:
        vae = self.pipeline.vae
        channels = vae.config.latent_channels
        h = self.config.height // 8
        w = self.config.width // 8
        latents = torch.randn(
            (1, channels, h, w), generator=generator, device=device, dtype=dtype
        )
        # For FlowMatchEulerDiscreteScheduler, init_noise_sigma is 1.0
        sigma_init = getattr(self.pipeline.scheduler, "init_noise_sigma", 1.0)
        return latents * sigma_init

    # ── look-ahead pass ─────────────────────────────────────────────────────────

    def _run_lookahead(
        self,
        init_latents: torch.Tensor,
        all_timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        seed: int,
    ) -> MaskBuildResult:
        """Run a short partial denoising pass per adapter to extract spatial masks."""
        la_start = self.config.lookahead_steps[0]
        la_end = self.config.lookahead_steps[-1]
        scheduler = self.pipeline.scheduler
        attn_module = self.blocks[self.lookahead_block_idx].attn
        tokenizer = self.pipeline.tokenizer

        scores_by_adapter: dict[str, torch.Tensor] = {}

        for adapter_id, trigger in zip(self.adapter_ids, self.triggers):
            # Bug 2: enable_lora() clears PEFT's _disable_adapters flag before set_adapters()
            self.pipeline.enable_lora()
            self.pipeline.set_adapters([adapter_id], adapter_weights=[1.0])

            # Bug 1: reset scheduler step index so each adapter's pass starts at t[0]
            self.pipeline.scheduler._step_index = None
            latents = init_latents.clone()
            attn_accum: Optional[torch.Tensor] = None
            n_captured = 0

            for step_idx, t in enumerate(all_timesteps[: la_end + 1]):
                hook: Optional[AttentionCaptureHook] = None
                if step_idx >= la_start:
                    hook = AttentionCaptureHook(attn_module, self.n_img_tokens)

                with torch.no_grad():
                    noise_pred = self.pipeline.transformer(
                        hidden_states=latents,
                        timestep=t.expand(latents.shape[0]),
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_embeds,
                        return_dict=False,
                    )[0]

                if hook is not None:
                    hook.remove()
                    if hook.last_attn_map is not None:
                        if attn_accum is None:
                            attn_accum = hook.last_attn_map.float()
                        else:
                            attn_accum = attn_accum + hook.last_attn_map.float()
                        n_captured += 1

                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if attn_accum is None or n_captured == 0:
                # Fallback: uniform scores → adapter gets no spatial preference.
                # Bug 5: derive token count from latent shape rather than self.n_img_tokens
                # to avoid a mismatch if the config height/width differs from the actual run.
                n_tokens = (init_latents.shape[2] // 2) * (init_latents.shape[3] // 2)
                scores_by_adapter[adapter_id] = torch.ones(
                    n_tokens, device=init_latents.device
                )
                continue

            attn_map = attn_accum / n_captured  # [N_img, N_txt]

            token_indices = get_concept_token_indices(tokenizer, self.prompt, trigger)
            # Clamp to valid text-token range (attn_map might have fewer columns)
            n_txt = attn_map.shape[-1]
            token_indices = [i for i in token_indices if i < n_txt]
            if not token_indices:
                token_indices = [0]

            raw_scores = reduce_attention_to_patch_scores(attn_map, token_indices)
            smoothed = smooth_and_binarize(
                raw_scores,
                n_blur_iter=5,
                binarize_tau=self.config.mask_binarize_tau,
            )
            scores_by_adapter[adapter_id] = smoothed

        return build_exclusive_binary_masks(
            scores_by_adapter,
            confidence_threshold=self.config.mask_confidence_threshold,
        )

    # ── denoising dispatch ──────────────────────────────────────────────────────

    def _denoising_step(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        step_idx: int,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        negative_pooled_embeds: torch.Tensor,
        mask_result: Optional[MaskBuildResult],
        method: str,
    ) -> torch.Tensor:
        do_cfg = self.config.guidance_scale > 1.0

        if method == "merge":
            self.pipeline.set_adapters(
                self.adapter_ids,
                adapter_weights=[1.0 / len(self.adapter_ids)] * len(self.adapter_ids),
            )
            return self._forward_single(
                latents, t, prompt_embeds, negative_prompt_embeds,
                pooled_embeds, negative_pooled_embeds, do_cfg,
            )

        if method == "switch":
            active = (
                self.adapter_ids[0]
                if step_idx < self.config.switch_step
                else self.adapter_ids[-1]
            )
            self.pipeline.set_adapters([active], adapter_weights=[1.0])
            return self._forward_single(
                latents, t, prompt_embeds, negative_prompt_embeds,
                pooled_embeds, negative_pooled_embeds, do_cfg,
            )

        # sd35_mask_only / sd35_mask_nullproj
        assert mask_result is not None

        if step_idx < self.config.switch_step:
            # Let the base model establish the global layout before blending starts
            self.pipeline.set_adapters(
                self.adapter_ids,
                adapter_weights=[1.0 / len(self.adapter_ids)] * len(self.adapter_ids),
            )
            return self._forward_single(
                latents, t, prompt_embeds, negative_prompt_embeds,
                pooled_embeds, negative_pooled_embeds, do_cfg,
            )

        return self._forward_blended(
            latents, t,
            prompt_embeds, negative_prompt_embeds,
            pooled_embeds, negative_pooled_embeds,
            mask_result, method, do_cfg,
        )

    # ── single forward (merge / switch / warmup) ────────────────────────────────

    def _forward_single(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        negative_pooled_embeds: torch.Tensor,
        do_cfg: bool,
    ) -> torch.Tensor:
        with torch.no_grad():
            if do_cfg:
                latent_input = torch.cat([latents, latents])
                enc = torch.cat([negative_prompt_embeds, prompt_embeds])
                pool = torch.cat([negative_pooled_embeds, pooled_embeds])
                pred = self.pipeline.transformer(
                    hidden_states=latent_input,
                    timestep=t.expand(latent_input.shape[0]),
                    encoder_hidden_states=enc,
                    pooled_projections=pool,
                    return_dict=False,
                )[0]
                pred_uncond, pred_cond = pred.chunk(2)
                return pred_uncond + self.config.guidance_scale * (pred_cond - pred_uncond)
            return self.pipeline.transformer(
                hidden_states=latents,
                timestep=t.expand(latents.shape[0]),
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]

    # ── blended forward (sd35_mask_only / sd35_mask_nullproj) ──────────────────

    def _forward_blended(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        negative_pooled_embeds: torch.Tensor,
        mask_result: MaskBuildResult,
        method: str,
        do_cfg: bool,
    ) -> torch.Tensor:
        intervention_block = self.blocks[self.intervention_block_idx]

        def _run_capture(adapter_id: Optional[str]) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            if adapter_id is None:
                self.pipeline.disable_lora()
            else:
                # Bug 2: enable_lora() clears PEFT's _disable_adapters flag first
                self.pipeline.enable_lora()
                self.pipeline.set_adapters([adapter_id], adapter_weights=[1.0])
            cap = HiddenStateCaptureHook(intervention_block)
            try:  # Bug 6: ensure hook is removed even if forward pass raises
                with torch.no_grad():
                    self.pipeline.transformer(
                        hidden_states=latents,
                        timestep=t.expand(latents.shape[0]),
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_embeds,
                        return_dict=False,
                    )
            finally:
                cap.remove()
            return cap.last_hidden, cap.last_enc_hidden

        # Pass 1: base (no LoRA)
        h_base, h_base_enc = _run_capture(None)

        if h_base is None:
            # Hook failed — fall back to merged pass
            warnings.warn(
                "HiddenStateCaptureHook returned None; falling back to merged forward.",
                stacklevel=2,
            )
            self.pipeline.set_adapters(
                self.adapter_ids,
                adapter_weights=[1.0 / len(self.adapter_ids)] * len(self.adapter_ids),
            )
            return self._forward_single(
                latents, t, prompt_embeds, negative_prompt_embeds,
                pooled_embeds, negative_pooled_embeds, do_cfg,
            )

        # Passes 2..N+1: per adapter
        h_by_adapter: dict[str, torch.Tensor] = {}
        for adapter_id in self.adapter_ids:
            h_k, _ = _run_capture(adapter_id)
            if h_k is not None:
                h_by_adapter[adapter_id] = h_k

        # Compute structural basis for null projection (once, from base hidden states)
        if method == "sd35_mask_nullproj":
            basis = compute_structural_basis(
                h_base, rank=self.config.svd_rank, center=True
            )
        else:
            basis = None

        # Blend: start from base, add spatially-masked (and optionally projected) deltas
        h_blended = h_base.clone()
        for adapter_id, mask_bool in mask_result.binary_masks_by_adapter.items():
            if adapter_id not in h_by_adapter:
                continue
            delta = h_by_adapter[adapter_id] - h_base  # [B, N_img, C]
            if basis is not None:
                mu = self.config.null_proj_mu
                delta = project_delta_soft(delta, basis, mu)
            mask_dev = mask_bool.to(device=h_blended.device)
            h_blended[:, mask_dev, :] = (
                h_blended[:, mask_dev, :]
                + delta[:, mask_dev, :].to(dtype=h_blended.dtype)
            )

        # Injection pass: conditional with blended hidden states
        self.pipeline.disable_lora()
        inject = HiddenStateInjectionHook(intervention_block)
        inject.inject_hidden = h_blended
        inject.inject_enc_hidden = h_base_enc

        try:  # Bug 6: always remove injection hook to avoid hook leaks on exceptions
            with torch.no_grad():
                pred_cond = self.pipeline.transformer(
                    hidden_states=latents,
                    timestep=t.expand(latents.shape[0]),
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_embeds,
                    return_dict=False,
                )[0]

            if do_cfg:
                # Unconditional pass: no LoRA, no injection (pure base model)
                inject.reset()
                with torch.no_grad():
                    pred_uncond = self.pipeline.transformer(
                        hidden_states=latents,
                        timestep=t.expand(latents.shape[0]),
                        encoder_hidden_states=negative_prompt_embeds,
                        pooled_projections=negative_pooled_embeds,
                        return_dict=False,
                    )[0]
                return pred_uncond + self.config.guidance_scale * (pred_cond - pred_uncond)

            return pred_cond
        finally:
            inject.remove()
