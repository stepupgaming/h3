"""Build MiniMax-H3 API-format Comfy graphs with optional turbo / Sol / cache.

Product compose order (Saganaki + Larryvrh):
  UNET → [TurboLoRA] → [KJ PathchSage] → [KJ H3 MemSage]
       → [Scheduled Sol] → [Fused Mod] → [Chunk FF]
       → [Spectrum | EasyCache | FBC]
       → SigmaShift → cond → sampler → decode → SaveVideo
"""
from __future__ import annotations

from typing import Any

FL2VA_DIT = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
REF2VA_DIT = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
W4A8_DIT = "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors"
TE_NVFP4 = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE_VIDEO = "minimax_h3_video_vae_int8_convrot.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"

TURBO_LORA = {
    "v4": "minimax_h3_turbo_v4_step600_ema.safetensors",
    "v1": "minimax_h3_turbo_4step_ema_ckpt850.safetensors",
    "ema_wan": "minimax_h3_turbo_4step_ema_ckpt850.safetensors",
}

# fal MiniMax-H3-Realism-People-LoRA (T2V/I2V/R2V). Generic model-only loader.
REALISM_LORA = "h3-realism-people-t2v-i2v-r2v.safetensors"
REALISM_TRIGGER = "r34l1sm"

# ganloss `2026-08-26 minimax_h3_r2v_story_board.json`: 8-step simple schedule
# SplitSigmas@4 (high only, no VAE decode), 3D latent upscale megapixels=1,
# then ManualSigmas 3-step. Not a full 8-step sample at 0.2 MP.
GANLOSS_SPLIT_STEP = 4
GANLOSS_REFINE_SIGMAS = "0.9035, 0.6316, 0.3158, 0.0000"
GANLOSS_UPSCALE = "minimax_h3_latent_upscaler_3d_fp16.safetensors"


def _nid(counter: list[int]) -> str:
    counter[0] += 1
    return str(counter[0])


def _save_video_node(video: Any, filename_prefix: str) -> dict[str, Any]:
    """Comfy 0.36 SaveVideo: option-key strings, codec as dotted child."""
    return _node(
        "SaveVideo",
        video=video,
        filename_prefix=filename_prefix,
        format="auto",
        **{"format.codec": "auto"},
    )


def _node(class_type: str, **inputs: Any) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs}


def _link(node_id: str, slot: int = 0) -> list:
    return [node_id, slot]


def build_h3_workflow(
    *,
    mode: str = "fl2va",
    prompt: str,
    width: int = 1280,
    height: int = 736,
    length: int = 124,
    seed: int = 42,
    steps: int = 20,
    first_image: str | None = None,
    last_image: str | None = None,
    ref_images: list[str] | None = None,
    ref_videos: list[str] | None = None,
    ref_video_audios: list[str | None] | None = None,
    ref_audios: list[str] | None = None,
    unet_name: str | None = None,
    turbo: str = "off",
    turbo_strength: float = 1.0,
    turbo_low_vram: bool = False,
    realism: bool = False,
    realism_strength: float = 1.0,
    sol: bool = False,
    cache: str = "off",
    kj_sage_patch: bool | None = None,
    fused_mod: bool | None = None,
    chunk_ff: bool | None = None,
    filename_prefix: str = "h3/gemmy",
    shift_video: float = 12.0,
    shift_audio: float = 3.0,
    dit_quant: str = "int8",
    persist_latent: bool = True,
    latent_prefix: str | None = None,
    fingerprint_json: str = "{}",
    continuation_mode: str = "none",
    context_latent: str | None = None,
    context_video: str | None = None,
    context_frames: int = 39,
    audio_mode: str = "generated_audio",
    trim_prefix: bool = False,
    guide_images: list[dict[str, Any]] | None = None,
    denoise: float = 1.0,
    refine_sigmas: str | None = None,
    upscale_model_name: str | None = None,
    init_audio: str | None = None,
    crop_video: str | None = None,
    crop_mask_video: str | None = None,
    video_vae: str | None = None,
) -> dict[str, Any]:
    """Return an API-format prompt dict ready for PromptExecutor.

    Dense Sage/FA2 still come from process CLI flags. KJ/Sol/fusion patches
    default on only when ``sol`` is requested so chick-baseline graphs stay
    bit-compatible with the proven 20-step sage path.
    ``init_audio`` encodes a wav into the target audio latent (not a frozen
    ref). Pair with ``denoise`` < 1 on the euler KSampler path.
    """
    mode = (mode or "fl2va").lower()
    turbo = (turbo or "off").lower()
    cache = (cache or "off").lower()
    if cache not in ("off", "spectrum", "easy", "fbc"):
        raise ValueError(f"unsupported cache={cache!r}")
    if turbo not in ("off", "v4", "v1", "ema_wan"):
        raise ValueError(f"unsupported turbo={turbo!r}")
    # Sol compose: Saganaki fusion/FF always; KJ sage patches when available
    # (headless PromptServer stub). Baseline dense sage uses only
    # ``--use-sage-attention`` (no graph patches).
    if kj_sage_patch is None:
        kj_sage_patch = bool(sol)
    if fused_mod is None:
        fused_mod = bool(sol)
    if chunk_ff is None:
        chunk_ff = bool(sol)

    if unet_name is None:
        if mode == "ref2va":
            unet_name = REF2VA_DIT
        elif dit_quant == "w4a8":
            unet_name = W4A8_DIT
        else:
            unet_name = FL2VA_DIT

    g: dict[str, Any] = {}
    c = [0]

    n_unet = _nid(c)
    g[n_unet] = _node("UNETLoader", unet_name=unet_name, weight_dtype="default")

    n_clip = _nid(c)
    g[n_clip] = _node(
        "CLIPLoader", clip_name=TE_NVFP4, type="minimax", device="cpu"
    )

    n_vvae = _nid(c)
    g[n_vvae] = _node("VAELoader", vae_name=video_vae or VAE_VIDEO)

    n_avae = _nid(c)
    g[n_avae] = _node("VAELoader", vae_name=VAE_AUDIO)

    model_src = n_unet

    # --- optional acceleration patches on MODEL ---
    if turbo != "off":
        lora_name = TURBO_LORA[turbo]
        n_turbo = _nid(c)
        g[n_turbo] = _node(
            "MiniMaxH3TurboLoRA",
            model=_link(model_src),
            lora_name=lora_name,
            strength=float(turbo_strength),
            low_vram=bool(turbo_low_vram),
        )
        model_src = n_turbo

    # Style LoRA after turbo (turbo node owns pruned-AdaLN apply; realism is plain).
    if realism:
        n_real = _nid(c)
        g[n_real] = _node(
            "LoraLoaderModelOnly",
            model=_link(model_src),
            lora_name=REALISM_LORA,
            strength_model=float(realism_strength),
        )
        model_src = n_real

    if kj_sage_patch:
        # Global KJ sage patch (classic mapping name is intentionally misspelled upstream).
        # Process-level --use-sage-attention remains the dense baseline; this graph
        # patch is for Sol compose order (Sage captured as Sol fallback).
        n_ps = _nid(c)
        g[n_ps] = _node(
            "PathchSageAttentionKJ",
            model=_link(model_src),
            sage_attention="auto",
        )
        model_src = n_ps

        # Optional H3-specific mem-efficient sage (KJ LTXV module). Skip at build
        # time only when explicitly disabled; runner validates class presence.
        if kj_sage_patch is not False:
            n_ms = _nid(c)
            g[n_ms] = _node(
                "MiniMaxH3MemoryEfficientSageAttentionPatch",
                model=_link(model_src),
            )
            model_src = n_ms

    if sol:
        n_sol = _nid(c)
        g[n_sol] = _node(
            "MiniMaxH3ScheduledSolAttentionPatch",
            model=_link(model_src),
            enabled=True,
            tau_start=1.3,
            tau_end=0.8,
            curve="linear",
            min_tokens=4096,
            strict=False,
            dense_percent=0.0,
            # All required by Saganaki node (Comfy does not apply INPUT_TYPES defaults
            # for missing API-graph keys the way the UI does).
            thresh_type="diag",
            int8_qk=False,
            int8_pv=False,
            sink_conditioning="exact_kv",
            dense_blocks="",
        )
        model_src = n_sol

    if fused_mod:
        n_fm = _nid(c)
        g[n_fm] = _node(
            "MiniMaxH3FusedModulation",
            model=_link(model_src),
            enabled=True,
        )
        model_src = n_fm

    if chunk_ff:
        n_ff = _nid(c)
        g[n_ff] = _node(
            "MiniMaxH3ChunkFeedForward",
            model=_link(model_src),
            enabled=True,
            chunks=2,
            min_tokens=8192,
        )
        model_src = n_ff

    if cache == "spectrum":
        n_sp = _nid(c)
        g[n_sp] = _node(
            "SpectrumApplyMiniMaxH3",
            model=_link(model_src),
            enabled=True,
            blend_weight=0.50,
            degree=1,
            ridge_lambda=0.10,
            window_size=2.0,
            flex_window=0.75,
            warmup_steps=1,
            tail_actual_steps=1,
            max_history=8,
            debug=False,
            history_storage="system_ram",
            bootstrap_first_forecast=True,
            anchor_residual_feedback=False,
            selective_rollback_correction=False,
            offline_smoothing_replay=True,
            audio_blend_weight=0.0,
            offline_archive_storage="system_ram",
        )
        model_src = n_sp
    elif cache == "easy":
        n_ez = _nid(c)
        g[n_ez] = _node(
            "EasyCache",
            model=_link(model_src),
            reuse_threshold=0.2,
            start_percent=0.15,
            end_percent=0.95,
        )
        model_src = n_ez
    elif cache == "fbc":
        n_fb = _nid(c)
        g[n_fb] = _node(
            "ApplyMiniMaxH3FirstBlockCache",
            model=_link(model_src),
            mode="H3 Fast — 0.10 / max 2",
            threshold=0.10,
            start_percent=0.10,
            end_percent=0.95,
            max_consecutive_hits=2,
            temporal_guard=False,
        )
        model_src = n_fb

    # Both source shortfilm graphs keep MiniMaxH3SigmaShift 12/3. Pass 0/0 to skip.
    if float(shift_video) > 0.0 or float(shift_audio) > 0.0:
        n_shift = _nid(c)
        g[n_shift] = _node(
            "MiniMaxH3SigmaShift",
            model=_link(model_src),
            shift_video=float(shift_video),
            shift_audio=float(shift_audio),
        )
        model_src = n_shift

    # --- conditioning ---
    image_nodes: dict[str, str] = {}

    def _load_image(basename: str) -> str:
        if basename in image_nodes:
            return image_nodes[basename]
        nid = _nid(c)
        g[nid] = _node("LoadImage", image=basename)
        image_nodes[basename] = nid
        return nid

    video_nodes: dict[str, str] = {}
    audio_nodes: dict[str, str] = {}

    def _load_video(basename: str) -> tuple[str, str]:
        if basename in video_nodes:
            nid = video_nodes[basename]
            return nid, f"{nid}_comp"
        nid = _nid(c)
        g[nid] = _node("LoadVideo", file=basename)
        video_nodes[basename] = nid
        cid = _nid(c)
        g[cid] = _node("GetVideoComponents", video=_link(nid))
        return nid, cid

    def _load_audio(basename: str) -> str:
        if basename in audio_nodes:
            return audio_nodes[basename]
        nid = _nid(c)
        g[nid] = _node("LoadAudio", audio=basename)
        audio_nodes[basename] = nid
        return nid

    continuation_mode = (continuation_mode or "none").lower()
    if continuation_mode not in (
        "none",
        "masked_av",
        "native_guide",
        "guide",
        "vae_tail",
        "latent_refine",
        "ganloss_two_stage",
        "mask_edit",
    ):
        raise ValueError(f"unsupported continuation_mode={continuation_mode!r}")
    audio_mode = (audio_mode or "generated_audio").lower()
    if audio_mode == "generated":
        audio_mode = "generated_audio"
    if audio_mode not in ("generated_audio", "source_track"):
        raise ValueError(f"unsupported audio_mode={audio_mode!r}")

    if mode == "ref2va":
        refs = list(ref_images or [])
        videos = list(ref_videos or [])
        standalone_audios = list(ref_audios or [])
        if not refs and not videos:
            raise ValueError("ref2va requires at least one ref image or video basename")
        cond_in: dict[str, Any] = {
            "clip": _link(n_clip),
            "vae": _link(n_vvae),
            "audio_vae": _link(n_avae),
            "prompt": prompt,
            "width": int(width),
            "height": int(height),
            "length": int(length),
            "ref_image_size": "match",
        }
        for i, name in enumerate(refs):
            iid = _load_image(name)
            cond_in[f"ref_images.ref_image_{i}"] = _link(iid)
        paired_audios = list(ref_video_audios or [])
        while len(paired_audios) < len(videos):
            paired_audios.append(None)
        for i, name in enumerate(videos):
            _vid, cid = _load_video(name)
            cond_in[f"ref_videos.ref_video_{i}"] = _link(cid, 0)
            if paired_audios[i]:
                aid = _load_audio(paired_audios[i])
                cond_in[f"ref_video_audios.ref_video_audio_{i}"] = _link(aid)
            else:
                # Soundtrack muxed in the video file.
                cond_in[f"ref_video_audios.ref_video_audio_{i}"] = _link(cid, 1)
        for i, name in enumerate(standalone_audios):
            aid = _load_audio(name)
            cond_in[f"ref_audios.ref_audio_{i}"] = _link(aid)
        n_cond = _nid(c)
        g[n_cond] = _node("MiniMaxH3ReferenceToVideo", **cond_in)
    else:
        cond_in = {
            "clip": _link(n_clip),
            "vae": _link(n_vvae),
            "prompt": prompt,
            "width": int(width),
            "height": int(height),
            "length": int(length),
        }
        if first_image:
            cond_in["first_frame"] = _link(_load_image(first_image))
        if last_image:
            cond_in["last_frame"] = _link(_load_image(last_image))
        n_cond = _nid(c)
        g[n_cond] = _node("MiniMaxH3ImageToVideo", **cond_in)

    cond_src = n_cond
    empty_latent = _link(n_cond, 1)

    if continuation_mode == "guide" or guide_images:
        for gi, guide in enumerate(guide_images or []):
            name = guide.get("image") or guide.get("file")
            if not name:
                raise ValueError(f"guide_images[{gi}] needs an image basename")
            n_guide = _nid(c)
            g[n_guide] = _node(
                "MiniMaxH3AddGuide",
                positive=_link(cond_src, 0),
                latent=empty_latent,
                vae=_link(n_vvae),
                image=_link(_load_image(str(name))),
                frame_idx=int(guide.get("frame_idx") or 0),
            )
            cond_src = n_guide

    latent_in = empty_latent
    if init_audio:
        aid = _load_audio(str(init_audio))
        n_init = _nid(c)
        g[n_init] = _node(
            "GemmyH3InitFromAudio",
            samples=latent_in,
            audio=_link(aid),
            audio_vae=_link(n_avae),
        )
        latent_in = _link(n_init)
    protected_frames_link = None
    if continuation_mode in ("masked_av", "vae_tail"):
        ctx_in: dict[str, Any] = {
            "target": empty_latent,
            "context_frames": int(context_frames),
            "audio_mode": audio_mode,
            "vae": _link(n_vvae),
            "audio_vae": _link(n_avae),
        }
        if continuation_mode == "masked_av":
            if not context_latent:
                raise ValueError("masked_av continuation needs context_latent")
            n_prev = _nid(c)
            g[n_prev] = _node("GemmyH3LoadAVLatent", path=str(context_latent))
            ctx_in["previous"] = _link(n_prev)
        else:
            if not context_video:
                raise ValueError("vae_tail continuation needs context_video")
            _vid, cid = _load_video(context_video)
            ctx_in["tail_image"] = _link(cid, 0)
            ctx_in["tail_audio"] = _link(cid, 1)
        n_ctx = _nid(c)
        g[n_ctx] = _node("GemmyH3MaskedAVContext", **ctx_in)
        latent_in = _link(n_ctx, 0)
        protected_frames_link = _link(n_ctx, 1)
    elif continuation_mode == "native_guide":
        if not context_latent:
            raise ValueError("native_guide continuation needs context_latent")
        n_prev = _nid(c)
        g[n_prev] = _node("GemmyH3LoadAVLatent", path=str(context_latent))
        n_guide = _nid(c)
        g[n_guide] = _node(
            "GemmyH3LatentTailGuide",
            positive=_link(cond_src, 0),
            previous=_link(n_prev),
            latent=empty_latent,
            context_frames=int(context_frames),
        )
        cond_src = n_guide
        latent_in = empty_latent
    elif continuation_mode == "latent_refine":
        if not context_latent:
            raise ValueError("latent_refine needs context_latent")
        if not upscale_model_name:
            raise ValueError("latent_refine needs upscale_model_name")
        n_prev = _nid(c)
        g[n_prev] = _node("GemmyH3LoadAVLatent", path=str(context_latent))
        n_split = _nid(c)
        g[n_split] = _node("GemmyH3SplitAV", samples=_link(n_prev))
        n_up = _nid(c)
        g[n_up] = _node(
            "MinimaxH3LatentUpscaler3D",
            latent=_link(n_split, 0),
            model_name=str(upscale_model_name),
            mode="target dimensions",
            align=32,
            keep_proportion=True,
            device="cuda",
            precision="fp16",
            **{
                "mode.width": int(width),
                "mode.height": int(height),
            },
        )
        n_join = _nid(c)
        g[n_join] = _node(
            "GemmyH3JoinAV",
            video=_link(n_up),
            audio=_link(n_split, 1),
        )
        latent_in = _link(n_join)
    elif continuation_mode == "mask_edit":
        if not crop_video:
            raise ValueError("mask_edit needs crop_video")
        if not crop_mask_video:
            raise ValueError("mask_edit needs crop_mask_video")
        _cvid, ccid = _load_video(crop_video)
        n_enc = _nid(c)
        g[n_enc] = _node(
            "GemmyH3EncodeVideoFrames",
            images=_link(ccid, 0),
            vae=_link(n_vvae),
        )
        n_split_empty = _nid(c)
        g[n_split_empty] = _node("GemmyH3SplitAV", samples=empty_latent)
        n_join_init = _nid(c)
        g[n_join_init] = _node(
            "GemmyH3JoinAV",
            video=_link(n_enc),
            audio=_link(n_split_empty, 1),
        )
        _mvid, mcid = _load_video(crop_mask_video)
        n_m2i = _nid(c)
        g[n_m2i] = _node("ImageToMask", image=_link(mcid, 0), channel="red")
        n_setm = _nid(c)
        g[n_setm] = _node(
            "GemmyH3SetAVNoiseMask",
            samples=_link(n_join_init),
            mask=_link(n_m2i),
        )
        latent_in = _link(n_setm)

    n_neg = _nid(c)
    g[n_neg] = _node("ConditioningZeroOut", conditioning=_link(cond_src, 0))

    # --- sampler ---
    # LBH latent refine: start from the upscaled AV latent and take a short
    # second sample at the new size (their example: ManualSigmas 3-step euler).
    # Continue/loop stay on denoise=1.0 + empty/masked target latents.
    # ganloss_two_stage is STAGE 1 ONLY on 16 GB: SplitSigmas@4 (high), persist
    # the denoised latent, decode a preview. Stage 2 (3D SR + 3-step) is a
    # fresh Comfy process via run_eros_stage2. One graph that also refines
    # dumps the DiT to 0 MB VRAM (~24 min/step). Their JSON is one graph
    # because they have the VRAM to keep the DiT loaded.
    refine_pass = continuation_mode == "latent_refine"
    ganloss_pass = continuation_mode == "ganloss_two_stage"
    if ganloss_pass:
        n_noise = _nid(c)
        g[n_noise] = _node("RandomNoise", noise_seed=int(seed))
        n_guider = _nid(c)
        g[n_guider] = _node(
            "BasicGuider",
            model=_link(model_src),
            conditioning=_link(cond_src, 0),
        )
        n_sched = _nid(c)
        g[n_sched] = _node(
            "BasicScheduler",
            model=_link(model_src),
            scheduler="simple",
            steps=int(steps),
            denoise=1.0,
        )
        n_split_sig = _nid(c)
        g[n_split_sig] = _node(
            "SplitSigmas",
            sigmas=_link(n_sched),
            step=int(GANLOSS_SPLIT_STEP),
        )
        n_ksel = _nid(c)
        g[n_ksel] = _node("KSamplerSelect", sampler_name="euler")
        n_samp1 = _nid(c)
        g[n_samp1] = _node(
            "SamplerCustomAdvanced",
            noise=_link(n_noise),
            guider=_link(n_guider),
            sampler=_link(n_ksel),
            sigmas=_link(n_split_sig, 0),
            latent_image=latent_in,
        )
        # ganloss feeds denoised_output (slot 1) into the upscaler. Persist that.
        latent_out = _link(n_samp1, 1)
    elif refine_pass and refine_sigmas:
        n_noise = _nid(c)
        g[n_noise] = _node("RandomNoise", noise_seed=int(seed))
        n_guider = _nid(c)
        g[n_guider] = _node(
            "BasicGuider",
            model=_link(model_src),
            conditioning=_link(cond_src, 0),
        )
        n_sig = _nid(c)
        g[n_sig] = _node("ManualSigmas", sigmas=str(refine_sigmas))
        n_ksel = _nid(c)
        g[n_ksel] = _node("KSamplerSelect", sampler_name="euler")
        n_samp = _nid(c)
        g[n_samp] = _node(
            "SamplerCustomAdvanced",
            noise=_link(n_noise),
            guider=_link(n_guider),
            sampler=_link(n_ksel),
            sigmas=_link(n_sig),
            latent_image=latent_in,
        )
        latent_out = _link(n_samp, 0)
    elif refine_pass:
        n_samp = _nid(c)
        g[n_samp] = _node(
            "KSampler",
            model=_link(model_src),
            seed=int(seed),
            steps=int(steps),
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            positive=_link(cond_src, 0),
            negative=_link(n_neg),
            latent_image=latent_in,
            denoise=float(denoise),
        )
        latent_out = _link(n_samp, 0)
    elif turbo != "off":
        n_noise = _nid(c)
        g[n_noise] = _node("RandomNoise", noise_seed=int(seed))

        n_guider = _nid(c)
        g[n_guider] = _node(
            "BasicGuider",
            model=_link(model_src),
            conditioning=_link(cond_src, 0),
        )

        n_sched = _nid(c)
        g[n_sched] = _node(
            "BasicScheduler",
            model=_link(model_src),
            scheduler="simple",
            steps=int(steps),
            denoise=1.0,
        )

        n_tsamp = _nid(c)
        g[n_tsamp] = _node("MiniMaxH3TurboSampler")

        n_samp = _nid(c)
        g[n_samp] = _node(
            "SamplerCustomAdvanced",
            noise=_link(n_noise),
            guider=_link(n_guider),
            sampler=_link(n_tsamp),
            sigmas=_link(n_sched),
            latent_image=latent_in,
        )
        latent_out = _link(n_samp, 0)
    else:
        n_samp = _nid(c)
        g[n_samp] = _node(
            "KSampler",
            model=_link(model_src),
            seed=int(seed),
            steps=int(steps),
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            positive=_link(cond_src, 0),
            negative=_link(n_neg),
            latent_image=latent_in,
            denoise=float(denoise),
        )
        latent_out = _link(n_samp, 0)

    if persist_latent:
        n_save_av = _nid(c)
        g[n_save_av] = _node(
            "GemmyH3SaveAVLatent",
            samples=latent_out,
            filename_prefix=latent_prefix or f"{filename_prefix}_av",
            fingerprint=fingerprint_json or "{}",
        )

    n_vd = _nid(c)
    g[n_vd] = _node("VAEDecode", samples=latent_out, vae=_link(n_vvae))

    n_ad = _nid(c)
    g[n_ad] = _node("VAEDecodeAudio", samples=latent_out, vae=_link(n_avae))

    images_out = _link(n_vd)
    audio_out = _link(n_ad)
    if trim_prefix and protected_frames_link is not None:
        n_trim = _nid(c)
        g[n_trim] = _node(
            "GemmyH3TrimProtectedPrefix",
            images=_link(n_vd),
            protected_frames=protected_frames_link,
            audio=_link(n_ad),
        )
        images_out = _link(n_trim, 0)
        audio_out = _link(n_trim, 1)

    n_cv = _nid(c)
    g[n_cv] = _node(
        "CreateVideo",
        images=images_out,
        fps=24.0,
        audio=audio_out,
    )

    n_sv = _nid(c)
    g[n_sv] = _save_video_node(_link(n_cv), filename_prefix)

    return g


def required_class_types(
    *,
    turbo: str = "off",
    sol: bool = False,
    cache: str = "off",
    kj_sage_patch: bool = True,
) -> list[str]:
    """Node class_types that must be present for a given stack."""
    base = [
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "LoadImage",
        "MiniMaxH3SigmaShift",
        "MiniMaxH3ImageToVideo",
        "MiniMaxH3ReferenceToVideo",
        "ConditioningZeroOut",
        "KSampler",
        "KSamplerSelect",
        "RandomNoise",
        "BasicGuider",
        "BasicScheduler",
        "SplitSigmas",
        "SamplerCustomAdvanced",
        "ManualSigmas",
        "MinimaxH3LatentUpscaler3D",
        "VAEDecode",
        "VAEDecodeAudio",
        "CreateVideo",
        "SaveVideo",
        "GemmyH3SaveAVLatent",
        "GemmyH3LoadAVLatent",
        "GemmyH3MaskedAVContext",
        "GemmyH3TrimProtectedPrefix",
        "GemmyH3SplitAV",
        "GemmyH3JoinAV",
        "MiniMaxH3AddGuide",
        "LoadVideo",
        "GetVideoComponents",
        "LoadAudio",
        "GemmyH3EncodeVideoFrames",
        "GemmyH3SetAVNoiseMask",
        "ImageToMask",
        "MaskToImage",
        "CheckpointLoaderSimple",
        "CLIPTextEncode",
        "SAM3_VideoTrack",
        "SAM3_TrackToMask",
    ]
    if turbo != "off":
        base += [
            "MiniMaxH3TurboLoRA",
            "MiniMaxH3TurboSampler",
            "RandomNoise",
            "BasicGuider",
            "BasicScheduler",
            "SamplerCustomAdvanced",
        ]
    if kj_sage_patch:
        base += [
            "PathchSageAttentionKJ",
            "MiniMaxH3MemoryEfficientSageAttentionPatch",
        ]
    if sol:
        base += [
            "MiniMaxH3ScheduledSolAttentionPatch",
            "MiniMaxH3FusedModulation",
            "MiniMaxH3ChunkFeedForward",
        ]
    if cache == "spectrum":
        base.append("SpectrumApplyMiniMaxH3")
    elif cache == "easy":
        base.append("EasyCache")
    elif cache == "fbc":
        base.append("ApplyMiniMaxH3FirstBlockCache")
    return base


def build_sam3_track_workflow(
    *,
    video_basename: str,
    mask_prompt: str,
    ckpt_name: str,
    length: int,
    object_indices: str = "0",
    detection_threshold: float = 0.5,
    max_objects: int = 1,
    filename_prefix: str = "h3/gemmy_sam3",
) -> dict[str, Any]:
    """SAM3.1 video track → mask MP4. Run as its own Comfy process on 16 GB."""
    g: dict[str, Any] = {}
    c = [0]
    n_vid = _nid(c)
    g[n_vid] = _node("LoadVideo", file=str(video_basename))
    n_comp = _nid(c)
    g[n_comp] = _node("GetVideoComponents", video=_link(n_vid))
    n_ckpt = _nid(c)
    g[n_ckpt] = _node("CheckpointLoaderSimple", ckpt_name=str(ckpt_name))
    n_clip = _nid(c)
    g[n_clip] = _node(
        "CLIPTextEncode",
        text=str(mask_prompt),
        clip=_link(n_ckpt, 1),
    )
    n_track = _nid(c)
    g[n_track] = _node(
        "SAM3_VideoTrack",
        images=_link(n_comp, 0),
        model=_link(n_ckpt, 0),
        conditioning=_link(n_clip),
        detection_threshold=float(detection_threshold),
        max_objects=int(max_objects),
        detect_interval=1,
    )
    n_tomask = _nid(c)
    g[n_tomask] = _node(
        "SAM3_TrackToMask",
        track_data=_link(n_track),
        object_indices=str(object_indices),
    )
    n_m2i = _nid(c)
    g[n_m2i] = _node("MaskToImage", mask=_link(n_tomask))
    n_cv = _nid(c)
    g[n_cv] = _node("CreateVideo", images=_link(n_m2i), fps=24.0)
    n_sv = _nid(c)
    g[n_sv] = _save_video_node(_link(n_cv), filename_prefix)
    _ = length  # LoadVideo frame cap is the staged file; length is caller-owned.
    return g
