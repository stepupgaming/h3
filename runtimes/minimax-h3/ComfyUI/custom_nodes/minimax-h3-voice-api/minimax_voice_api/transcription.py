"""OpenAI-style response shaping for timestamped Parakeet results."""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

WORD_PATTERN = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)?", re.IGNORECASE)


@dataclass(frozen=True)
class TimedWord:
    word: str
    start: float
    end: float
    char_start: int
    char_end: int


def timestamped_words(result, duration: float) -> list[TimedWord]:
    """Map Parakeet subword timestamps onto readable word boundaries."""
    tokens = result.tokens or []
    timestamps = result.timestamps or []
    matches = list(WORD_PATTERN.finditer(result.text))
    if not tokens or not timestamps or not matches:
        return []

    joined = "".join(tokens)
    base = joined.find(result.text)
    if base < 0:
        base = len(joined) - len(joined.lstrip())
    token_ends: list[int] = []
    length = 0
    for token in tokens:
        length += len(token)
        token_ends.append(length)

    starts: list[float] = []
    for match in matches:
        index = min(
            bisect.bisect_right(token_ends, base + match.start()),
            len(timestamps) - 1,
        )
        starts.append(max(0.0, float(timestamps[index])))

    words: list[TimedWord] = []
    for index, match in enumerate(matches):
        start = starts[index]
        final_token = min(
            bisect.bisect_right(token_ends, base + match.end() - 1),
            len(timestamps) - 1,
        )
        estimate = 0.10 + min(0.55, len(match.group(0)) * 0.045)
        natural_end = float(timestamps[final_token]) + estimate
        if index + 1 < len(matches):
            end = max(start + 0.04, min(starts[index + 1] - 0.02, natural_end))
        else:
            end = max(start + 0.08, natural_end)
            if duration > 0:
                end = min(duration, end)
        words.append(TimedWord(
            word=match.group(0), start=round(start, 3), end=round(end, 3),
            char_start=match.start(), char_end=match.end(),
        ))
    return words


def transcription_segments(text: str, words: list[TimedWord],
                           max_seconds: float = 8.0) -> list[dict]:
    """Group timestamped words into subtitle-sized sentence segments."""
    if not words:
        return []
    segments: list[dict] = []
    first = 0
    for index, word in enumerate(words):
        next_word = words[index + 1] if index + 1 < len(words) else None
        between = text[word.char_end:next_word.char_start] if next_word else text[word.char_end:]
        sentence_end = bool(re.search(r"[.!?]", between))
        time_limit = word.end - words[first].start >= max_seconds
        if next_word is None or sentence_end or time_limit:
            end_char = next_word.char_start if next_word else len(text)
            segment_text = text[words[first].char_start:end_char].strip()
            segments.append({
                "id": len(segments),
                "start": words[first].start,
                "end": word.end,
                "text": segment_text,
            })
            first = index + 1
    return segments


def verbose_transcription(result, duration: float,
                          model: str) -> dict:
    words = timestamped_words(result, duration)
    return {
        "task": "transcribe",
        "language": "english",
        "duration": round(duration, 3),
        "text": result.text.strip(),
        "model": model,
        "segments": transcription_segments(result.text, words),
        "words": [
            {"word": word.word, "start": word.start, "end": word.end}
            for word in words
        ],
    }


def _subtitle_time(seconds: float, separator: str) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def subtitles(segments: list[dict], format_name: str) -> str:
    """Render timestamped segments as SRT or WebVTT."""
    if format_name not in {"srt", "vtt"}:
        raise ValueError(f"unsupported subtitle format: {format_name}")
    separator = "," if format_name == "srt" else "."
    blocks: list[str] = []
    for index, segment in enumerate(segments, 1):
        start = _subtitle_time(float(segment["start"]), separator)
        end = _subtitle_time(float(segment["end"]), separator)
        prefix = f"{index}\n" if format_name == "srt" else ""
        blocks.append(f"{prefix}{start} --> {end}\n{segment['text']}")
    body = "\n\n".join(blocks)
    return f"WEBVTT\n\n{body}\n" if format_name == "vtt" else f"{body}\n"
