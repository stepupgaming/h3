"""Canvas / resolution preset unit tests."""

from __future__ import annotations

from minimax_h3.canvas import (
    ASPECT_RATIOS,
    BASE_SHORT_EDGE,
    MAX_PIXELS,
    adapt_canvas,
    canvas_from_megapixels,
    canvas_from_aspect_short,
    list_mp_ladder,
    list_official_short_edge_table,
    resolve_canvas,
)


def test_adapt_preview_keeps_480p():
    w, h = adapt_canvas(864, 480, mode="preview")
    assert (w, h) == (864, 480)


def test_adapt_official_forces_768_short():
    w, h = adapt_canvas(864, 480, mode="official")
    assert min(w, h) == BASE_SHORT_EDGE
    assert w % 32 == 0 and h % 32 == 0
    assert w * h <= MAX_PIXELS


def test_official_16x9_native():
    c = canvas_from_aspect_short("16:9", 768, mode="official")
    assert (c.width, c.height) == (1344, 768)
    assert c.latent_hw == (48, 84)


def test_official_9x16_native():
    c = canvas_from_aspect_short("9:16", 768, mode="official")
    assert (c.width, c.height) == (768, 1344)


def test_official_1x1():
    c = canvas_from_aspect_short("1:1", 768, mode="official")
    assert (c.width, c.height) == (768, 768)


def test_mp_ladder_0p4_16x9():
    c = canvas_from_megapixels("16:9", 0.4, clamp_base=True)
    assert c.width % 32 == 0 and c.height % 32 == 0
    # Comfy-ish ~832x480 (not owner 864x480)
    assert c.height == 480
    assert 800 <= c.width <= 864


def test_mp_above_base_clamped():
    c = canvas_from_megapixels("16:9", 2.0, clamp_base=True)
    assert c.clamped
    assert c.width * c.height <= MAX_PIXELS


def test_all_named_aspects_legal():
    for name in ASPECT_RATIOS:
        c = canvas_from_aspect_short(name, 768, mode="official")
        assert c.width % 32 == 0 and c.height % 32 == 0
        assert c.width * c.height <= MAX_PIXELS


def test_resolve_aspect_beats_missing_wh():
    c = resolve_canvas(aspect="9:16", short_edge=768, mode="official")
    assert (c.width, c.height) == (768, 1344)


def test_resolve_default_native():
    c = resolve_canvas()
    assert (c.width, c.height) == (1344, 768)


def test_tables_nonempty():
    assert len(list_official_short_edge_table()) >= 6
    assert len(list_mp_ladder("16:9")) >= 10
