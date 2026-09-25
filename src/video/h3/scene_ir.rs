//! Live-action Ref2VA IR for one still = one clip.
//!
//! Quality lock: `outputs/locked_h3_shortfilm_parlor_20260827/`.
//! Do not name a storyboard, grid, split, contact sheet, or `[Shot N]` —
//! even as a negative. Eros prints those words.

/// One still = one live-action clip. `<Picture 1>` is that photograph.
///
/// Scene index stays out of the DiT prompt (`_scene` / `_n_scenes` are for
/// callers).
pub(crate) fn compile_scene_ir(
    brief: &str,
    duration_s: f64,
    _scene: u32,
    _n_scenes: u32,
) -> String {
    let duration_s = duration_s.max(0.5);
    let brief = brief.trim();
    let intent = {
        let mut t = if brief.is_empty() {
            "the character performs this action".to_string()
        } else {
            brief.to_string()
        };
        if !t.ends_with('.') && !t.ends_with('!') && !t.ends_with('?') {
            t.push('.');
        }
        t
    };
    let ir = format!(
        "subject_definitions:\n\
- <Subject 1>: The person visible in <Picture 1>. Keep face, hair, body, and outfit identical.\n\
- <Picture 1>: Photoreal live-action photograph. Camera height, framing, wardrobe, lighting, and room are this image.\n\n\
summary:\n\
[reference generation] Create a {duration_s:.0}-second photoreal cinematic clip of <Subject 1> from <Picture 1>. {intent}\n\n\
retention_analysis:\n\
- <Subject 1> — fully_preserved: exact identity from <Picture 1>.\n\
- <Picture 1> — fully_preserved: camera, blocking, wardrobe, lighting, and room follow this photograph.\n\n\
detailed_description:\n\
Photoreal live-action cinematography. Stay inside the same room and framing as <Picture 1>. {intent} Motion is continuous. No on-screen text, logos, or watermarks.\n\n\
overall_soundscape:\n\
Natural room ambience locked to visible action. No narrator unless the brief requires dialogue.\n\n\
non_diegetic_music:\n\
Soft cinematic underscore, unobtrusive.\n"
    );
    if let Some(leak) = per_scene_ir_board_leak(&ir) {
        eprintln!(
            "[h3 ref2va] warning: live-action IR contains {leak:?}; Eros will print a board"
        );
    }
    ir
}

/// Video-editing IR: source plate is `<Video 1>`, replacement still is `<Picture 1>`.
/// Official task type is `[video editing + reference generation]`.
pub(crate) fn compile_edit_ir(brief: &str, duration_s: f64, mask_prompt: &str) -> String {
    let duration_s = duration_s.max(0.5);
    let brief = brief.trim();
    let intent = {
        let mut t = if brief.is_empty() {
            "replace only the masked region with the reference still".to_string()
        } else {
            brief.to_string()
        };
        if !t.ends_with('.') && !t.ends_with('!') && !t.ends_with('?') {
            t.push('.');
        }
        t
    };
    let region = {
        let p = mask_prompt.trim();
        if p.is_empty() {
            "the tracked region".to_string()
        } else {
            format!("the tracked '{p}' region")
        }
    };
    format!(
        "subject_definitions:\n\
- <Picture 1>: the replacement reference still. Use this identity, face, hair, and clothing inside {region}.\n\
- <Video 1>: the source video. Preserve camera motion, framing, timing, environment, lighting, unmasked people, and the performance {region} is attached to.\n\n\
summary:\n\
[video editing + reference generation] The target is an edited version of <Video 1> ({duration_s:.0}s). {intent} Change only {region}. Keep every other element of <Video 1> unchanged.\n\n\
retention_analysis:\n\
- <Video 1> — fully_preserved: camera path, framing, timing, environment, lighting, unmasked people, and soundtrack.\n\
- <Picture 1> — attribute_transfer: identity, face, hair, and clothing inside {region} follow this still.\n\
- {region} — motion and performance from <Video 1>; appearance from <Picture 1>. Do not keep the source hair or wardrobe inside {region}.\n\n\
detailed_description:\n\
One continuous shot. {intent} The person in {region} is the person in <Picture 1>, including hair. Do not redesign unmasked faces, bodies, backgrounds, or timing.\n\n\
overall_soundscape:\n\
Reuse the original soundtrack of <Video 1>.\n\n\
non_diegetic_music:\n\
N/A\n"
    )
}

/// Eros treats these as an instruction to *draw* a board. Negation does not help.
pub(crate) fn per_scene_ir_board_leak(ir: &str) -> Option<&'static str> {
    let l = ir.to_ascii_lowercase();
    for (needle, label) in [
        ("storyboard", "storyboard"),
        ("contact sheet", "contact sheet"),
        ("split screen", "split screen"),
        ("split-screen", "split-screen"),
        ("[shot", "[Shot"),
        ("panel border", "panel border"),
        ("comic", "comic"),
        ("grid", "grid"),
    ] {
        if l.contains(needle) {
            return Some(label);
        }
    }
    if contains_scene_n_of_m(&l) {
        return Some("scene N of M");
    }
    None
}

pub(crate) fn contains_scene_n_of_m(lower: &str) -> bool {
    let mut rest = lower;
    while let Some(i) = rest.find("scene ") {
        let after = &rest[i + 6..];
        let digits = after.chars().take_while(|c| c.is_ascii_digit()).count();
        if digits > 0 {
            let after_n = &after[digits..];
            if after_n.starts_with(" of ") {
                let after_of = &after_n[4..];
                if after_of.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                    return true;
                }
            }
        }
        rest = &rest[i + 6..];
    }
    false
}
