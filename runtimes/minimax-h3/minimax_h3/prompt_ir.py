"""Open H3-Context-IR *format* stand-in (L1 templates).

Official Context-IR is a closed hosted rewriter (``POST …/v2/h3_context_ir``).
Open Base does **not** rewrite — text is tokenized as written — so quality comes
from emitting the same structured field order the official guides document.

This module is the free local path the community actually uses:

- T2VA / I2VA / FL2VA / L2VA → alignment (when needed) + 3 core fields
- Ref2VA → 6 sections in fixed order

It is **not** a neural clone of Context-IR and not H3-Regenerate-2K.
Optional later: feed the same schema as few-shot/system text to a local VLM (L2),
or collect API IR pairs to fine-tune a small rewriter — still format imitation,
not reverse-engineered closed weights.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

MODES = ("t2va", "i2va", "fl2va", "l2va", "ref2va")

STYLES = (
    "Live-action, cinematic",
    "live-action",
    "2D-animated",
    "3D CG",
    "claymation",
    "watercolor",
    "vintage film",
)

CAMERA_TYPES = (
    "Zoom In",
    "Zoom Out",
    "Push In",
    "Pull Out",
    "Pan Left",
    "Pan Right",
    "Truck Left",
    "Truck Right",
    "Tilt Up",
    "Tilt Down",
    "Pedestal Up",
    "Pedestal Down",
    "Arc Shot",
    "Tracking Shot",
    "Static Shot",
    "Shake Slightly",
    "Shake Strongly",
    "POV",
    "Roll Clockwise",
    "Roll Counterclockwise",
)

RETENTION_MARKERS = (
    "fully_preserved",
    "partially_preserved",
    "attribute_transfer",
    "weak_reference",
)

AUDIO_RETENTION = (
    "fully_copy",
    "partially_copy",
    "reference",
    "weak_reference",
)

REF_TASK_TYPES = (
    "keyframe completion",
    "reference generation",
    "video editing",
    "video continuation",
    "audio reuse",
    "audio reference",
)

# 17k+5 pixel-frame grid @ 24 fps (official / community tables).
FRAME_GRID_24FPS = (
    (124, 5.17),
    (141, 5.88),
    (158, 6.58),
    (175, 7.29),
    (192, 8.00),
    (209, 8.71),
    (226, 9.42),
    (243, 10.13),
    (260, 10.83),
    (277, 11.54),
    (294, 12.25),
    (311, 12.96),
    (328, 13.67),
    (345, 14.38),
    (362, 15.08),
)


def snap_duration_seconds(seconds: float) -> tuple[int, float]:
    """Nearest official frame-grid duration at 24 fps. Returns (frames, seconds)."""
    if seconds <= 0:
        raise ValueError("duration must be > 0")
    best = min(FRAME_GRID_24FPS, key=lambda p: abs(p[1] - seconds))
    return best[0], best[1]


def format_timestamp(seconds: float) -> str:
    """``MM:SS.mmm`` style used in shot cut lines (minutes can be 00)."""
    if seconds < 0:
        raise ValueError("timestamp must be >= 0")
    whole = int(seconds)
    ms = int(round((seconds - whole) * 1000.0))
    if ms == 1000:
        whole += 1
        ms = 0
    mm, ss = divmod(whole, 60)
    return f"{mm:02d}:{ss:02d}.{ms:03d}"


def format_end_seconds(seconds: float) -> str:
    """Two-decimal duration used in FL2VA/L2VA alignment lines."""
    return f"{seconds:.2f}"


def _clean_paragraph(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t


def _ensure_period(text: str) -> str:
    t = _clean_paragraph(text)
    if not t:
        return t
    if t[-1] not in ".!?":
        t += "."
    return t


def _normalize_style(style: Optional[str]) -> str:
    s = _clean_paragraph(style) if style else STYLES[0]
    # Avoid double "Live-action, cinematic, Live-action..."
    return s.rstrip(".,;:")


@dataclass
class SubjectDef:
    """One ``<Subject N>`` / picture / video / audio definition line."""

    label: str  # e.g. "Subject 1", "Picture 1", "Video 1", "Audio 1"
    text: str

    def render(self) -> str:
        lab = self.label.strip()
        if not lab.startswith("<"):
            lab = f"<{lab}>"
        body = _ensure_period(self.text)
        # "is the ..." preferred; allow full sentences already starting with label prose
        if body.lower().startswith(lab.lower()):
            return body
        if re.match(r"^(is|are|provides|defines|marks)\b", body, re.I):
            return f"{lab} {body}"
        return f"{lab} is {body}"


@dataclass
class RetentionLine:
    label: str
    marker: str
    note: str
    appears_in: str = "[Shot 1]"

    def render(self) -> str:
        lab = self.label.strip()
        if not lab.startswith("<"):
            lab = f"<{lab}>"
        marker = self.marker.strip()
        note = _ensure_period(self.note)
        return f"{lab} (appears in {self.appears_in}): {marker} - {note}"


@dataclass
class ShotSpec:
    """One timeline shot. Shot 1 has no timestamp; later shots need cut_time_s."""

    action: str
    cut_time_s: Optional[float] = None
    camera: Optional[str] = None  # natural phrase or type name from CAMERA_TYPES

    def render(self, index: int, *, style: Optional[str] = None) -> str:
        body = _clean_paragraph(self.action)
        cam = _clean_paragraph(self.camera) if self.camera else ""
        if cam and cam.lower() not in body.lower():
            # Prefer natural motion sentence fragment
            if cam in CAMERA_TYPES:
                cam_phrase = f"The camera performs a {cam}"
                if "amplitude" not in cam.lower() and cam not in (
                    "Static Shot",
                    "POV",
                ):
                    cam_phrase += " with small amplitude at slow speed"
                cam_phrase += "."
            else:
                cam_phrase = _ensure_period(cam)
            body = f"{body} {cam_phrase}".strip() if body else cam_phrase
        body = _ensure_period(body)
        if index == 1:
            st = _normalize_style(style)
            # If user already led with a style keyword, don't double-prefix
            lead = body[:48].lower()
            if any(k.split(",")[0].lower() in lead for k in STYLES):
                return f"[Shot 1] {body}"
            return f"[Shot 1] {st}, {body[0].lower() + body[1:] if body else body}"
        if self.cut_time_s is None:
            raise ValueError(f"Shot {index} requires cut_time_s")
        ts = format_timestamp(self.cut_time_s)
        return f"[Shot {index}] At {ts}, the camera cuts to {body[0].lower() + body[1:] if body else body}"


@dataclass
class PromptIRRequest:
    """Inputs for compiling an IR-format prompt."""

    mode: str = "t2va"
    brief: str = ""
    duration_s: float = 10.0
    style: str = STYLES[0]
    soundscape: Optional[str] = None
    music: Optional[str] = None  # None → heuristic; "" or "N/A" → N/A
    shots: List[ShotSpec] = field(default_factory=list)
    # Ref2VA
    subjects: List[SubjectDef] = field(default_factory=list)
    summary: Optional[str] = None
    task_types: Sequence[str] = field(default_factory=lambda: ("reference generation",))
    retention: List[RetentionLine] = field(default_factory=list)
    detailed_extra: Optional[str] = None
    # Keyframe alignment helpers
    last_shot_index: int = 1  # for FL2VA/L2VA Picture alignment
    n_pictures: int = 0
    n_videos: int = 0
    n_audios: int = 0


def _default_shots(req: PromptIRRequest) -> List[ShotSpec]:
    if req.shots:
        return list(req.shots)
    brief = _ensure_period(req.brief) or "A subject holds in frame while the scene develops."
    mode = req.mode.lower()
    if mode == "i2va":
        action = (
            f"the opening composition matches the first-frame reference. "
            f"From that locked start, {brief.rstrip('.')} develops forward without breaking identity, wardrobe, or layout anchors"
        )
        return [ShotSpec(action=action, camera="Static Shot")]
    if mode == "fl2va":
        action = (
            f"the first-frame state is established, then the scene continuously transitions "
            f"so that by the end it lands on the last-frame state. Path: {brief.rstrip('.')}"
        )
        return [ShotSpec(action=action)]
    if mode == "l2va":
        action = (
            f"a plausible preceding state begins off the final reference, then action and framing "
            f"converge until the last frame matches the reference. Intent: {brief.rstrip('.')}"
        )
        return [ShotSpec(action=action)]
    if mode == "ref2va":
        action = brief
        return [ShotSpec(action=action)]
    # t2va
    return [ShotSpec(action=brief)]


def _default_soundscape(req: PromptIRRequest) -> str:
    if req.soundscape is not None and _clean_paragraph(req.soundscape):
        return _ensure_period(req.soundscape)
    brief = _clean_paragraph(req.brief).lower()
    bits = ["Room tone and natural ambience fill the space"]
    if any(k in brief for k in ("rain", "storm", "thunder")):
        bits = ["Rain and distant thunder set a wet outdoor ambience"]
    elif any(k in brief for k in ("ocean", "sea", "wave", "beach")):
        bits = ["Ocean surf and light wind establish a coastal ambience"]
    elif any(k in brief for k in ("city", "street", "traffic", "crowd")):
        bits = ["Distant traffic and city murmur form the bed"]
    elif any(k in brief for k in ("space", "bridge", "ship", "fleet", "sci-fi")):
        bits = ["Low hull resonance and soft console ticks define the interior"]
    bits.append("Footsteps, cloth movement, and object contacts stay synced to on-screen action")
    return " ".join(_ensure_period(b) for b in bits)


def _default_music(req: PromptIRRequest) -> str:
    if req.music is not None:
        m = _clean_paragraph(req.music)
        return m if m else "N/A"
    brief = _clean_paragraph(req.brief).lower()
    if any(k in brief for k in ("silent", "no music", "diegetic only")):
        return "N/A"
    if any(k in brief for k in ("epic", "battle", "fleet", "opera", "trailer", "teaser")):
        return (
            "Low brass and taiko enter at a slow tempo, strings build in rising figures, "
            "percussion densifies toward a short final swell, then cuts clean."
        )
    if any(k in brief for k in ("horror", "dread", "terror")):
        return (
            "Sparse low drones and dissonant high strings hold a slow pulse with "
            "occasional dry percussive hits; dynamics stay restrained until a brief late rise."
        )
    # Default: light underscore rather than empty (empty often lets model invent music)
    return (
        "Soft sustained pads and a simple mid-tempo pulse under the scene, "
        "instrumentation thin and unobtrusive, dynamics mostly flat with a gentle late settle."
    )


def alignment_line(mode: str, duration_s: float, *, last_shot_index: int = 1) -> Optional[str]:
    """Official first-line instruction for keyframe modes; None for T2VA/Ref2VA body."""
    m = mode.lower()
    end = format_end_seconds(duration_s)
    if m == "i2va":
        return (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced."
        )
    if m == "fl2va":
        n = max(int(last_shot_index), 1)
        return (
            "How the reference pictures align with the target video — "
            "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
            f"Picture 2 (from Shot {n}) aligns with the {end}-second mark of the target video."
        )
    if m == "l2va":
        n = max(int(last_shot_index), 1)
        return (
            "How the reference pictures align with the target video — "
            f"<Picture 1> (from [Shot {n}]) aligns with the {end}-second mark of the target video."
        )
    return None


def _render_integrated(shots: Sequence[ShotSpec], style: str) -> str:
    parts = [s.render(i + 1, style=style if i == 0 else None) for i, s in enumerate(shots)]
    return " ".join(parts)


def _default_subjects(req: PromptIRRequest) -> List[SubjectDef]:
    if req.subjects:
        return list(req.subjects)
    out: List[SubjectDef] = []
    n_pic = max(req.n_pictures, 1)  # Ref2VA almost always has ≥1 image in practice
    brief = _clean_paragraph(req.brief) or "the primary subject of the scene"
    # Identity in Subject, not naked Picture
    out.append(
        SubjectDef(
            "Subject 1",
            f"the primary character or object whose appearance comes from <Picture 1> — "
            f"retain face/body proportions, wardrobe, and palette as visible in the reference. "
            f"Scene intent: {brief.rstrip('.')}",
        )
    )
    out.append(
        SubjectDef(
            "Picture 1",
            "the primary visual reference frame providing appearance and composition anchors for [Shot 1]",
        )
    )
    for i in range(2, n_pic + 1):
        out.append(
            SubjectDef(
                f"Picture {i}",
                f"an additional visual reference used for identity/angle coverage of <Subject 1>",
            )
        )
    for i in range(1, req.n_videos + 1):
        out.append(
            SubjectDef(
                f"Video {i}",
                "a motion, camera-path, or pacing reference for the target video (not a frame-preserving edit source unless summary says video editing)",
            )
        )
    for i in range(1, req.n_audios + 1):
        out.append(
            SubjectDef(
                f"Audio {i}",
                "an audio reference for timbre, ambience, or rhythm in the target video",
            )
        )
    return out


def _default_retention(req: PromptIRRequest, subjects: Sequence[SubjectDef]) -> List[RetentionLine]:
    if req.retention:
        return list(req.retention)
    lines: List[RetentionLine] = []
    for s in subjects:
        lab = s.label.strip().strip("<>")
        low = lab.lower()
        if low.startswith("subject"):
            lines.append(
                RetentionLine(
                    lab,
                    "fully_preserved",
                    "identity features, wardrobe, and key proportions from the linked picture references are retained",
                )
            )
        elif low.startswith("picture"):
            # Standalone pictures used as composition anchors
            if "first frame" in s.text.lower() or "keyframe" in s.text.lower():
                lines.append(
                    RetentionLine(
                        lab,
                        "fully_preserved",
                        "the referenced frame composition is treated as a hard visual anchor where stated",
                    )
                )
            else:
                # Appearance source already covered by Subject — weak on raw picture bleed
                lines.append(
                    RetentionLine(
                        lab,
                        "weak_reference",
                        "used only as appearance source material via subject definitions; do not paste unrelated background or extra people",
                    )
                )
        elif low.startswith("video"):
            lines.append(
                RetentionLine(
                    lab,
                    "weak_reference",
                    "only camera path, cut rhythm, or motion energy are followed; wardrobe, location, and cast from the ref video do not appear",
                )
            )
        elif low.startswith("audio"):
            lines.append(
                RetentionLine(
                    lab,
                    "reference",
                    "timbre, ambience character, or rhythm inform the new track without mandatory full signal copy",
                )
            )
    return lines


def _default_summary(req: PromptIRRequest) -> str:
    if req.summary and _clean_paragraph(req.summary):
        body = _ensure_period(req.summary)
        if body.startswith("["):
            return body
        types = " + ".join(req.task_types) if req.task_types else "reference generation"
        return f"[{types}] {body}"
    types = " + ".join(req.task_types) if req.task_types else "reference generation"
    brief = _ensure_period(req.brief) or "The target video follows the reference subjects in a new shot."
    if brief and brief[0].islower():
        brief = brief[0].upper() + brief[1:]
    return (
        f"[{types}] The target video presents <Subject 1> using appearance guidance from "
        f"<Picture 1>. {brief}"
    )


def _default_detailed(req: PromptIRRequest, shots: Sequence[ShotSpec]) -> str:
    style = _normalize_style(req.style)
    integrated = _render_integrated(shots, style)
    extra = _clean_paragraph(req.detailed_extra) if req.detailed_extra else ""
    # Official wants ~350–500 words when hand-written; L1 expands with explicit anchors
    lead = (
        f"The target video is {style.lower()} in look: natural materials, coherent lighting, "
        f"stable geometry, and continuous motion without teleporting props. "
    )
    mid = (
        f"{integrated} "
        f"Throughout, <Subject 1> keeps the same identity cues established in subject_definitions; "
        f"hands and face stay anatomically plausible; shoulders rotate with the head on turns; "
        f"background parallax matches the camera move. "
        f"Diegetic sound stays locked to visible contacts; spoken lines use stable speaker IDs if any dialogue appears. "
    )
    tail = (
        "The shot ends on a readable held state so the final frames are usable as a still. "
        "No on-screen camera body is shown as a prop unless the brief explicitly requires it."
    )
    if extra:
        return _clean_paragraph(f"{lead}{mid}{extra} {tail}")
    return _clean_paragraph(f"{lead}{mid}{tail}")


def compile_prompt(req: PromptIRRequest) -> str:
    """Compile a paste-ready IR-format prompt string."""
    mode = req.mode.lower().strip()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {req.mode!r}")

    _frames, dur = snap_duration_seconds(float(req.duration_s))
    shots = _default_shots(req)
    # Validate later-shot timestamps strictly increasing and within duration
    prev_t = -1.0
    for i, s in enumerate(shots):
        if i == 0:
            continue
        if s.cut_time_s is None:
            raise ValueError(f"shots[{i}] needs cut_time_s")
        if s.cut_time_s <= prev_t:
            raise ValueError("shot cut times must be strictly increasing")
        if s.cut_time_s >= dur:
            raise ValueError(f"shot cut {s.cut_time_s} must be < duration {dur}")
        prev_t = s.cut_time_s

    sound = _default_soundscape(req)
    music = _default_music(req)
    style = _normalize_style(req.style)

    if mode == "ref2va":
        subjects = _default_subjects(req)
        retention = _default_retention(req, subjects)
        summary = _default_summary(req)
        detailed = _default_detailed(req, shots)
        blocks = [
            "subject_definitions:\n" + "\n".join(s.render() for s in subjects),
            f"summary:\n{summary}",
            "retention_analysis:\n" + "\n".join(r.render() for r in retention),
            f"detailed_description:\n{detailed}",
            f"overall_soundscape:\n{sound}",
            f"non_diegetic_music:\n{music}",
        ]
        return "\n\n".join(blocks).strip() + "\n"

    # T2VA family
    align = alignment_line(mode, dur, last_shot_index=req.last_shot_index)
    integrated = _render_integrated(shots, style)
    body = (
        f"integrated_multimodal_description: {integrated}\n\n"
        f"overall_soundscape: {sound}\n\n"
        f"non_diegetic_music: {music}\n"
    )
    if align:
        return f"{align}\n\n{body}"
    return body


def compile_from_brief(
    brief: str,
    *,
    mode: str = "t2va",
    duration_s: float = 10.0,
    style: str = STYLES[0],
    soundscape: Optional[str] = None,
    music: Optional[str] = None,
    camera: Optional[str] = None,
    n_pictures: int = 0,
    n_videos: int = 0,
    n_audios: int = 0,
    task_types: Optional[Sequence[str]] = None,
) -> str:
    """One-shot helper: short brief → full IR-format prompt."""
    shots = []
    if _clean_paragraph(brief):
        shots = [ShotSpec(action=brief, camera=camera)]
    req = PromptIRRequest(
        mode=mode,
        brief=brief,
        duration_s=duration_s,
        style=style,
        soundscape=soundscape,
        music=music,
        shots=shots,
        n_pictures=n_pictures,
        n_videos=n_videos,
        n_audios=n_audios,
        task_types=tuple(task_types) if task_types else ("reference generation",),
        last_shot_index=1,
    )
    return compile_prompt(req)


def is_ir_format(text: str) -> bool:
    """Cheap structural check: does text already look like official IR output?"""
    t = text or ""
    if "integrated_multimodal_description:" in t and "overall_soundscape:" in t:
        return True
    if "subject_definitions:" in t and "detailed_description:" in t and "retention_analysis:" in t:
        return True
    return False


def maybe_compile(text: str, **kwargs) -> str:
    """If ``text`` is already IR-shaped, return stripped text; else compile from brief."""
    t = (text or "").strip()
    if is_ir_format(t):
        return t if t.endswith("\n") else t + "\n"
    return compile_from_brief(t, **kwargs)
