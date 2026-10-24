from __future__ import annotations

import hashlib
import re

SENTENCE_END_RE = re.compile(r"[.!?]+(?=\s)")
FILE_DOT_RE = re.compile(r"(?<=\w\w)\.(?=[A-Za-z][A-Za-z0-9]{1,5}\b)")


def to_speech(markdown: str | None, max_chars: int = 1400) -> str:
    if not markdown:
        return ""
    text = re.sub(r"```.*?```", " (code block) ", markdown, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"^\s*\|.*\|\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_\n]+)__", r"\1", text)
    text = re.sub(r"\*([^*\n]+)\*", r"\1", text)
    text = re.sub(r"[#>~`|]", "", text)
    text = re.sub(r"^\s*[-+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + ". There is more on screen."
    return text


def speech_chunks(text: str, max_chars: int) -> list[str]:
    """Split text for synthesis, ending each chunk at a sentence end where one exists."""
    if not text:
        return []
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    chunks = []
    remaining = text.strip()
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        split_at = _split_point(remaining, max_chars)
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    return chunks


def _split_point(remaining: str, max_chars: int) -> int:
    """After the last sentence end past the half-way mark, else at the last space, else hard."""
    window = remaining[: max_chars + 1]
    sentence_ends = [match.end() for match in SENTENCE_END_RE.finditer(window)]
    if sentence_ends and sentence_ends[-1] >= max_chars // 2:
        return sentence_ends[-1]
    space = window[:max_chars].rfind(" ")
    return space if space >= max_chars // 2 else max_chars


def spoken_form(text: str) -> str:
    """Text as the voice should say it: a dot inside a file or domain name is spoken as "dot"."""
    return FILE_DOT_RE.sub(" dot ", text)


def to_gist(text: str | None, gist_chars: int = 240) -> str:
    if not text:
        return ""
    if len(text) <= gist_chars + 40:
        return text
    gist = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if not gist:
            gist = sentence
        elif len(gist) + 1 + len(sentence) <= gist_chars:
            gist += " " + sentence
        else:
            break
    return gist.strip() or text[:gist_chars].rsplit(" ", 1)[0]


def signature(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
