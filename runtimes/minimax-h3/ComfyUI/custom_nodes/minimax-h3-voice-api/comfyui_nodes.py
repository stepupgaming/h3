"""Native ComfyUI nodes for MiniMax H3 music and SFX studios.

Studio nodes expand into stock ComfyUI H3 nodes. They do not call the
FastAPI service, so workflows remain queue-safe and portable.
"""
from __future__ import annotations

import json

import comfy.samplers
import folder_paths
from comfy_execution.graph_utils import GraphBuilder

from .minimax_voice_api.engine import FPS, frames_for
from .minimax_voice_api.music import (
    instrumental_prompt,
    recommend_song_duration,
    song_prompt,
)
from .minimax_voice_api.music_presets import MUSIC_PRESETS
from .minimax_voice_api.sfx import (
    MAX_SECONDS as SFX_MAX_SECONDS,
)
from .minimax_voice_api.sfx import (
    MIN_SECONDS as SFX_MIN_SECONDS,
)
from .minimax_voice_api.sfx import (
    recommend_sfx_duration,
    sfx_prompt,
)
from .minimax_voice_api.sfx_presets import SFX_PRESETS

DEFAULT_UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
DEFAULT_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DEFAULT_VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
DEFAULT_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
DEFAULT_STYLE = (
    "A wry, slow, swaggering acoustic blues-country novelty song around 88 BPM, "
    "dryly funny but emotionally sincere, with restrained verses and a memorable "
    "singalong chorus; the arrangement stays coherent and small for the full take"
)
DEFAULT_INSTRUMENTATION = (
    "Fingerpicked resonator guitar, warm upright bass, compact brushed drum kit, "
    "occasional muted electric-guitar answers, and nothing else; no choir, no backing "
    "singers, no synths, and no orchestral build"
)
DEFAULT_VOCALIST = (
    "one expressive adult female singer with a warm weathered contralto, dry comic "
    "timing, clear natural English diction, subtle rasp, and professional connected phrasing"
)


class H3RefKeep:
    """Control MiniMax H3 reference conditioning without altering output audio."""

    CATEGORY = "MiniMax H3/Voice API"
    FUNCTION = "run"
    RETURN_TYPES = ("CONDITIONING",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "audio_keep": ("FLOAT", {"default": 1.0, "min": 0.90,
                                      "max": 1.0, "step": 0.005}),
            "visual_keep": ("FLOAT", {"default": 0.999, "min": 0.90,
                                       "max": 1.0, "step": 0.001}),
        }}

    def run(self, conditioning, audio_keep, visual_keep):
        output = []
        for entry in conditioning:
            metadata = dict(entry[1]) if len(entry) > 1 else {}
            metadata["minimax_audio_cond_noise_aug"] = float(audio_keep)
            metadata["minimax_visual_cond_noise_aug"] = float(visual_keep)
            output.append([entry[0], metadata])
        return (output,)

DEFAULT_LYRICS = """[Intro]

[Verse 1]
I've been sitting on this shelf since nineteen seventy-five
Got two googly eyes and no good reason to be alive
She said I was solid, she said I was sweet
Then she left me for a geode from across the street

[Chorus]
I'm a pet rock, honey, and I ain't got no game
Just a lump in a cardboard box that nobody came to claim
Got no arms to hold you, got no lips to kiss
Just a hundred million years of loneliness

[Bridge]
They call me low maintenance like that makes it okay
A hundred million years and nobody knows my name

[Outro]"""


def _preferred(values: list[str], preferred: str) -> list[str]:
    return [preferred, *[value for value in values if value != preferred]] if preferred in values else values


def _preset_labels() -> tuple[list[str], dict[str, str]]:
    labels = ["Custom"]
    mapping: dict[str, str] = {}
    for preset_id, preset in MUSIC_PRESETS.items():
        label = f"{preset['category']} · {preset['name']}"
        labels.append(label)
        mapping[label] = preset_id
    return labels, mapping


PRESET_LABELS, PRESET_IDS = _preset_labels()


class MiniMaxH3MusicStudio:
    """Compose a timed song or instrumental with the native H3 graph."""

    @classmethod
    def INPUT_TYPES(cls):
        diffusion_models = _preferred(
            folder_paths.get_filename_list("diffusion_models"), DEFAULT_UNET)
        text_encoders = _preferred(
            folder_paths.get_filename_list("text_encoders"), DEFAULT_CLIP)
        vaes = folder_paths.get_filename_list("vae")
        video_vaes = _preferred(vaes, DEFAULT_VIDEO_VAE)
        audio_vaes = _preferred(vaes, DEFAULT_AUDIO_VAE)
        samplers = list(comfy.samplers.KSampler.SAMPLERS)
        if "res_multistep" in samplers:
            samplers.remove("res_multistep")
            samplers.insert(0, "res_multistep")
        schedulers = [
            "simple", "sgm_uniform", "karras", "exponential", "ddim_uniform",
            "beta", "normal", "linear_quadratic", "kl_optimal",
        ]
        return {
            "required": {
                "mode": (["song", "instrumental"], {"default": "song"}),
                "preset": (PRESET_LABELS, {"default": "Custom"}),
                "duration_seconds": ("FLOAT", {
                    "default": 60.0, "min": 5.0, "max": 60.0, "step": 1.0,
                    "tooltip": "H3 snaps this to its 17-frame grid.",
                }),
                "steps": ("INT", {"default": 20, "min": 10, "max": 50, "step": 1}),
                "seed": ("INT", {
                    "default": 1234, "min": 0, "max": 2**63 - 1,
                    "control_after_generate": True,
                }),
                "style_notes": ("STRING", {
                    "default": DEFAULT_STYLE, "multiline": True,
                    "tooltip": "Optional additions to the selected preset, or the complete style for Custom.",
                }),
                "instrumentation_notes": ("STRING", {
                    "default": DEFAULT_INSTRUMENTATION, "multiline": True,
                    "tooltip": "Optional arrangement additions, or the complete instrumentation for Custom.",
                }),
                "vocalist_notes": ("STRING", {
                    "default": DEFAULT_VOCALIST, "multiline": True,
                    "tooltip": "Voice identity and performance direction. This is never placed inside lyric tags.",
                }),
                "lyrics": ("STRING", {
                    "default": DEFAULT_LYRICS, "multiline": True,
                    "tooltip": "Use [Intro], [Verse], [Chorus], [Bridge], [Solo], and [Outro] headers.",
                }),
                "scene": ("STRING", {
                    "default": "a dry intimate live room with one performer and a small band recorded in one continuous take",
                    "multiline": True,
                }),
                "soundscape": ("STRING", {
                    "default": "A dry intimate live-room recording with faint chair movement and natural instrument handling, but no spoken introduction and no audience",
                    "multiline": True,
                }),
                "language": ("STRING", {"default": "English"}),
                "resolution": (["32", "64", "128"], {"default": "32"}),
                "sampler": (samplers, {"default": "res_multistep"}),
                "scheduler": (schedulers, {"default": "simple"}),
                "unet_name": (diffusion_models,),
                "clip_name": (text_encoders,),
                "video_vae_name": (video_vaes,),
                "audio_vae_name": (audio_vaes,),
                # Append new controls after all v1 widgets. ComfyUI serializes
                # widget values positionally, so inserting fields in the middle
                # makes existing workflows assign model filenames to the wrong
                # combo boxes and falsely report them as missing.
                "duration_mode": (["auto", "manual"], {
                    "default": "auto",
                    "tooltip": "Auto estimates length from lyrics, tempo, and delivery style. Manual preserves the duration slider.",
                }),
                "prompt_profile": (["natural_song_sheet_v2", "petrock_timed_v1"], {
                    "default": "natural_song_sheet_v2",
                    "tooltip": "Natural song sheet keeps one continuous take. Legacy reproduces the original Pet Rock shot-timing grammar.",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("audio", "generated_prompt", "effective_duration", "section_plan")
    FUNCTION = "compose"
    CATEGORY = "MiniMax H3/Music"
    DESCRIPTION = (
        "Expands into the native MiniMax H3 FL2VA audio-first graph. Songs use "
        "a continuous music-first song sheet with exact lyric blocks."
    )

    @staticmethod
    def _validate_filename(category: str, value: str) -> str:
        if value not in folder_paths.get_filename_list(category):
            raise ValueError(f"Unknown {category} model: {value}")
        return value

    @staticmethod
    def _directions(preset: str, style_notes: str,
                    instrumentation_notes: str,
                    vocalist_notes: str) -> tuple[str, str, str]:
        preset_id = PRESET_IDS.get(preset)
        selected = MUSIC_PRESETS.get(preset_id, {})
        style = selected.get("style", "An original coherent musical performance")
        instrumentation = selected.get(
            "instrumentation", "A restrained arrangement with a clear melodic identity")
        vocalist = selected.get("vocalist", "a natural expressive adult singer")
        if style_notes.strip():
            style = f"{style.rstrip('.')}. {style_notes.strip()}" if selected else style_notes.strip()
        if instrumentation_notes.strip():
            instrumentation = (
                f"{instrumentation.rstrip('.')}. {instrumentation_notes.strip()}"
                if selected else instrumentation_notes.strip()
            )
        if vocalist_notes.strip():
            vocalist = (
                f"{vocalist.rstrip('.')}. {vocalist_notes.strip()}"
                if selected else vocalist_notes.strip()
            )
        return style, instrumentation, vocalist

    def compose(self, mode: str, preset: str, duration_mode: str,
                prompt_profile: str, duration_seconds: float,
                steps: int, seed: int, style_notes: str,
                instrumentation_notes: str, vocalist_notes: str,
                lyrics: str, scene: str, soundscape: str, language: str,
                resolution: str, sampler: str, scheduler: str,
                unet_name: str, clip_name: str, video_vae_name: str,
                audio_vae_name: str):
        unet_name = self._validate_filename("diffusion_models", unet_name)
        clip_name = self._validate_filename("text_encoders", clip_name)
        video_vae_name = self._validate_filename("vae", video_vae_name)
        audio_vae_name = self._validate_filename("vae", audio_vae_name)
        if sampler not in comfy.samplers.KSampler.SAMPLERS:
            raise ValueError(f"Unknown sampler: {sampler}")

        style, instrumentation, vocalist = self._directions(
            preset, style_notes, instrumentation_notes, vocalist_notes)
        recommendation = None
        if mode == "song":
            recommendation = recommend_song_duration(style, lyrics)
            if duration_mode == "auto":
                duration_seconds = recommendation["recommended_seconds"]
        frame_count = frames_for(duration_seconds)
        effective_duration = frame_count / FPS
        if mode == "song":
            prompt, plan = song_prompt(
                style, instrumentation, vocalist, lyrics, effective_duration,
                soundscape, language, scene, prompt_profile,
            )
        else:
            prompt = instrumental_prompt(style, instrumentation, soundscape)
            plan = []

        graph = GraphBuilder()
        model = graph.node("UNETLoader", unet_name=unet_name, weight_dtype="default")
        clip = graph.node(
            "CLIPLoader", clip_name=clip_name, type="minimax", device="default")
        video_vae = graph.node("VAELoader", vae_name=video_vae_name)
        audio_vae = graph.node("VAELoader", vae_name=audio_vae_name)
        conditioning = graph.node(
            "MiniMaxH3ImageToVideo", clip=clip.out(0), vae=video_vae.out(0),
            prompt=prompt, width=int(resolution), height=int(resolution),
            length=frame_count,
        )
        sigmas = graph.node(
            "BasicScheduler", model=model.out(0), scheduler=scheduler,
            steps=steps, denoise=1.0,
        )
        selected_sampler = graph.node("KSamplerSelect", sampler_name=sampler)
        noise = graph.node("RandomNoise", noise_seed=seed)
        guider = graph.node(
            "BasicGuider", model=model.out(0), conditioning=conditioning.out(0))
        sampled = graph.node(
            "SamplerCustomAdvanced", noise=noise.out(0), guider=guider.out(0),
            sampler=selected_sampler.out(0), sigmas=sigmas.out(0),
            latent_image=conditioning.out(1),
        )
        audio = graph.node(
            "VAEDecodeAudio", samples=sampled.out(0), vae=audio_vae.out(0))
        return {
            "result": (
                audio.out(0), prompt, effective_duration,
                json.dumps({
                    "prompt_profile": prompt_profile if mode == "song" else "instrumental_v1",
                    "duration_mode": duration_mode,
                    "duration_recommendation": recommendation,
                    "sections": plan,
                }, ensure_ascii=False, indent=2),
            ),
            "expand": graph.finalize(),
        }


def _sfx_preset_labels() -> tuple[list[str], dict[str, str]]:
    labels = ["Custom"]
    mapping: dict[str, str] = {}
    for preset_id, preset in SFX_PRESETS.items():
        label = f"{preset['category']} · {preset['name']}"
        labels.append(label)
        mapping[label] = preset_id
    return labels, mapping


SFX_PRESET_LABELS, SFX_PRESET_IDS = _sfx_preset_labels()

DEFAULT_SFX_DESCRIPTION = (
    "A single clean diegetic sound effect with a natural attack and short decay"
)
DEFAULT_SFX_SPACE = "a neutral dry recording stage with no visible source"


class MiniMaxH3SfxStudio:
    """Compose a diegetic one-shot or ambience bed with the native H3 graph."""

    @classmethod
    def INPUT_TYPES(cls):
        diffusion_models = _preferred(
            folder_paths.get_filename_list("diffusion_models"), DEFAULT_UNET)
        text_encoders = _preferred(
            folder_paths.get_filename_list("text_encoders"), DEFAULT_CLIP)
        vaes = folder_paths.get_filename_list("vae")
        video_vaes = _preferred(vaes, DEFAULT_VIDEO_VAE)
        audio_vaes = _preferred(vaes, DEFAULT_AUDIO_VAE)
        samplers = list(comfy.samplers.KSampler.SAMPLERS)
        if "res_multistep" in samplers:
            samplers.remove("res_multistep")
            samplers.insert(0, "res_multistep")
        schedulers = [
            "simple", "sgm_uniform", "karras", "exponential", "ddim_uniform",
            "beta", "normal", "linear_quadratic", "kl_optimal",
        ]
        return {
            "required": {
                "kind": (["oneshot", "loop_bed"], {"default": "oneshot"}),
                "preset": (SFX_PRESET_LABELS, {"default": "Custom"}),
                "duration_seconds": ("FLOAT", {
                    "default": 3.0, "min": SFX_MIN_SECONDS, "max": SFX_MAX_SECONDS,
                    "step": 0.5,
                    "tooltip": "H3 snaps this to its 17-frame grid. Short one-shots still need a couple of seconds of timeline.",
                }),
                "steps": ("INT", {"default": 20, "min": 10, "max": 50, "step": 1}),
                "seed": ("INT", {
                    "default": 1234, "min": 0, "max": 2**63 - 1,
                    "control_after_generate": True,
                }),
                "description": ("STRING", {
                    "default": DEFAULT_SFX_DESCRIPTION, "multiline": True,
                    "tooltip": "Describe the diegetic effect only. Do not write lyrics or dialogue.",
                }),
                "space": ("STRING", {
                    "default": DEFAULT_SFX_SPACE, "multiline": True,
                    "tooltip": "Acoustic space / recording perspective. Not spoken.",
                }),
                "intensity": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
                "resolution": (["32", "64", "128"], {"default": "32"}),
                "sampler": (samplers, {"default": "res_multistep"}),
                "scheduler": (schedulers, {"default": "simple"}),
                "unet_name": (diffusion_models,),
                "clip_name": (text_encoders,),
                "video_vae_name": (video_vaes,),
                "audio_vae_name": (audio_vaes,),
                "duration_mode": (["auto", "manual"], {
                    "default": "auto",
                    "tooltip": "Auto uses the preset default when a preset is selected, otherwise a kind-aware heuristic.",
                }),
                "prompt_profile": (["diegetic_v1"], {
                    "default": "diegetic_v1",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("audio", "generated_prompt", "effective_duration", "plan")
    FUNCTION = "compose"
    CATEGORY = "MiniMax H3/SFX"
    DESCRIPTION = (
        "Expands into the native MiniMax H3 audio-first graph for diegetic SFX "
        "and Foley. Forbids speech and non-diegetic music in the prompt."
    )

    @staticmethod
    def _validate_filename(category: str, value: str) -> str:
        if value not in folder_paths.get_filename_list(category):
            raise ValueError(f"Unknown {category} model: {value}")
        return value

    def compose(self, kind: str, preset: str, duration_mode: str,
                prompt_profile: str, duration_seconds: float,
                steps: int, seed: int, description: str, space: str,
                intensity: float, resolution: str, sampler: str, scheduler: str,
                unet_name: str, clip_name: str, video_vae_name: str,
                audio_vae_name: str):
        unet_name = self._validate_filename("diffusion_models", unet_name)
        clip_name = self._validate_filename("text_encoders", clip_name)
        video_vae_name = self._validate_filename("vae", video_vae_name)
        audio_vae_name = self._validate_filename("vae", audio_vae_name)
        if sampler not in comfy.samplers.KSampler.SAMPLERS:
            raise ValueError(f"Unknown sampler: {sampler}")

        preset_id = SFX_PRESET_IDS.get(preset)
        selected = SFX_PRESETS.get(preset_id or "", {})
        if selected:
            if description.strip() == DEFAULT_SFX_DESCRIPTION:
                description = str(selected["description"])
                kind = str(selected.get("kind") or kind)
            if space.strip() == DEFAULT_SFX_SPACE and selected.get("space"):
                space = str(selected["space"])

        recommendation = recommend_sfx_duration(description, kind=kind)
        if duration_mode == "auto":
            if selected.get("default_seconds") is not None and (
                description == str(selected.get("description", ""))
                or not description.strip()
            ):
                duration_seconds = float(selected["default_seconds"])
            else:
                duration_seconds = float(recommendation["recommended_seconds"])

        frame_count = frames_for(duration_seconds)
        effective_duration = frame_count / FPS
        prompt = sfx_prompt(
            description, kind=kind, space=space, intensity=intensity,
            profile=prompt_profile,
        )

        graph = GraphBuilder()
        model = graph.node("UNETLoader", unet_name=unet_name, weight_dtype="default")
        clip = graph.node(
            "CLIPLoader", clip_name=clip_name, type="minimax", device="default")
        video_vae = graph.node("VAELoader", vae_name=video_vae_name)
        audio_vae = graph.node("VAELoader", vae_name=audio_vae_name)
        conditioning = graph.node(
            "MiniMaxH3ImageToVideo", clip=clip.out(0), vae=video_vae.out(0),
            prompt=prompt, width=int(resolution), height=int(resolution),
            length=frame_count,
        )
        sigmas = graph.node(
            "BasicScheduler", model=model.out(0), scheduler=scheduler,
            steps=steps, denoise=1.0,
        )
        selected_sampler = graph.node("KSamplerSelect", sampler_name=sampler)
        noise = graph.node("RandomNoise", noise_seed=seed)
        guider = graph.node(
            "BasicGuider", model=model.out(0), conditioning=conditioning.out(0))
        sampled = graph.node(
            "SamplerCustomAdvanced", noise=noise.out(0), guider=guider.out(0),
            sampler=selected_sampler.out(0), sigmas=sigmas.out(0),
            latent_image=conditioning.out(1),
        )
        audio = graph.node(
            "VAEDecodeAudio", samples=sampled.out(0), vae=audio_vae.out(0))
        return {
            "result": (
                audio.out(0), prompt, effective_duration,
                json.dumps({
                    "kind": kind,
                    "preset": preset_id,
                    "prompt_profile": prompt_profile,
                    "duration_mode": duration_mode,
                    "duration_recommendation": recommendation,
                    "intensity": intensity,
                }, ensure_ascii=False, indent=2),
            ),
            "expand": graph.finalize(),
        }


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3MusicStudio": MiniMaxH3MusicStudio,
    "MiniMaxH3SfxStudio": MiniMaxH3SfxStudio,
    "H3RefKeep": H3RefKeep,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3MusicStudio": "H3 Music Studio",
    "MiniMaxH3SfxStudio": "H3 SFX Studio",
    "H3RefKeep": "H3 Reference Keep",
}
