"""Unit tests for open IR-format prompt compiler."""

from __future__ import annotations

from minimax_h3.prompt_ir import (
    PromptIRRequest,
    ShotSpec,
    SubjectDef,
    alignment_line,
    compile_from_brief,
    compile_prompt,
    is_ir_format,
    maybe_compile,
    snap_duration_seconds,
)


def test_snap_duration_near_10s():
    fr, sec = snap_duration_seconds(10.0)
    assert fr == 243
    assert abs(sec - 10.13) < 1e-6


def test_t2va_has_three_fields_no_align():
    p = compile_from_brief(
        "A woman folds a letter by a rainy window while the city murmurs outside",
        mode="t2va",
        duration_s=10,
    )
    assert p.startswith("integrated_multimodal_description:")
    assert "overall_soundscape:" in p
    assert "non_diegetic_music:" in p
    assert "For the target video" not in p
    assert "[Shot 1]" in p
    assert is_ir_format(p)


def test_i2va_alignment_prefix():
    p = compile_from_brief("She stands and walks to the door", mode="i2va", duration_s=8)
    assert p.startswith("For the target video, at 0.00 seconds")
    assert "<Picture 1>" in p.split("\n\n", 1)[0]
    assert "integrated_multimodal_description:" in p


def test_fl2va_alignment_uses_snapped_end():
    line = alignment_line("fl2va", 10.13, last_shot_index=1)
    assert line is not None
    assert "10.13-second mark" in line
    assert "Picture 2" in line


def test_ref2va_six_sections_order():
    p = compile_from_brief(
        "The woman from the photo walks through a neon market at night",
        mode="ref2va",
        duration_s=10,
        n_pictures=2,
        n_videos=1,
    )
    keys = [
        "subject_definitions:",
        "summary:",
        "retention_analysis:",
        "detailed_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    ]
    positions = [p.index(k) for k in keys]
    assert positions == sorted(positions)
    assert "<Subject 1>" in p
    assert "<Picture 1>" in p
    assert "[reference generation]" in p
    assert "fully_preserved" in p or "weak_reference" in p


def test_multi_shot_timestamps():
    req = PromptIRRequest(
        mode="t2va",
        brief="two beat scene",
        duration_s=10,
        shots=[
            ShotSpec(action="A wide shot holds on a quiet hallway"),
            ShotSpec(action="a close-up of a hand on the doorknob", cut_time_s=3.5),
        ],
    )
    p = compile_prompt(req)
    assert "[Shot 2] At 00:03.500" in p


def test_maybe_compile_passthrough():
    raw = compile_from_brief("hello world", mode="t2va")
    out = maybe_compile(raw, mode="t2va")
    assert "integrated_multimodal_description:" in out
    # Should not nest another integrated_ block by re-wrapping
    assert out.count("integrated_multimodal_description:") == 1


def test_custom_subject_lines():
    req = PromptIRRequest(
        mode="ref2va",
        brief="product hero spin",
        duration_s=5,
        subjects=[
            SubjectDef("Subject 1", "the red sneakers from <Picture 1>, white sole, clean laces"),
            SubjectDef("Picture 1", "the hero packshot used as appearance reference"),
        ],
        n_pictures=1,
    )
    p = compile_prompt(req)
    assert "red sneakers" in p
    assert p.index("subject_definitions:") < p.index("detailed_description:")
