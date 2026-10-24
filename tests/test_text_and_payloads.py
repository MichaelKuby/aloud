from __future__ import annotations

import json
from pathlib import Path

from aloud.text import speech_chunks, spoken_form, to_gist, to_speech
from aloud.transcripts import assistant_text_from_payload, last_assistant_text


def test_text_cleaning_removes_code_tables_links_and_caps_length():
    markdown = """# Done

Here is [the result](https://example.com).

```python
print("secret")
```

| a | b |
| - | - |
| 1 | 2 |

Use `foo`.
"""

    spoken = to_speech(markdown, max_chars=80)

    assert "Done" in spoken
    assert "the result" in spoken
    assert "code block" in spoken
    assert "https://" not in spoken
    assert "foo" in spoken
    assert "|" not in spoken


def test_inline_code_preserves_commands_and_filenames():
    spoken = to_speech("Run `pytest tests/test_attention_events.py` and edit `src/aloud/hooks.py`.")

    assert "pytest tests/test_attention_events.py" in spoken
    assert "src/aloud/hooks.py" in spoken


def test_gist_uses_first_sentences_with_cap():
    text = "First sentence. Second sentence fits. Third sentence should not fit. " * 12

    assert to_gist(text, gist_chars=38) == "First sentence. Second sentence fits."


def test_payload_text_wins_before_transcript_parsing(tmp_path: Path):
    transcript = tmp_path / "claude.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "fallback"}]},
            }
        )
        + "\n"
    )

    payload = {
        "last_assistant_message": "payload **text**",
        "transcript_path": str(transcript),
    }

    assert assistant_text_from_payload(payload) == "payload **text**"


def test_transcript_fallback_understands_claude_and_codex(tmp_path: Path):
    claude = tmp_path / "claude.jsonl"
    claude.write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Claude text"}]},
            }
        )
        + "\n"
    )
    codex = tmp_path / "codex.jsonl"
    codex.write_text(
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Codex text"}],
                },
            }
        )
        + "\n"
    )

    assert last_assistant_text(claude) == "Claude text"
    assert last_assistant_text(codex) == "Codex text"


FOUR_SENTENCES = (
    "First sentence of the reply here. Second sentence of the reply here! "
    "Third sentence of the reply here? Fourth sentence ends the reply here."
)


def test_chunks_end_at_sentence_ends_and_rejoin_to_the_text():
    chunks = speech_chunks(FOUR_SENTENCES, 40)  # each sentence fits, two do not

    assert len(chunks) == 4
    assert all(chunk[-1] in ".!?" for chunk in chunks)
    assert all(chunk[0].isupper() for chunk in chunks)
    assert " ".join(chunks) == FOUR_SENTENCES


def test_a_sentence_end_right_at_the_limit_still_counts():
    first = "A sentence that is exactly at the limit."
    text = first + " And a second one."

    assert speech_chunks(text, len(first)) == [first, "And a second one."]


def test_a_very_long_sentence_splits_at_a_space_and_never_inside_a_word():
    words = " ".join(f"word{index}" for index in range(40))

    chunks = speech_chunks(words, 50)

    assert len(chunks) > 1
    assert " ".join(chunks) == words
    assert all(not chunk.startswith(".") for chunk in chunks)


def test_a_dotted_file_or_domain_name_is_spoken_with_dot():
    assert spoken_form("Edit daemon.py and config.json, then open github.com.") == (
        "Edit daemon dot py and config dot json, then open github dot com."
    )
    assert spoken_form("See settings.json.bak-1 and test_playback.py.") == (
        "See settings dot json dot bak-1 and test_playback dot py."
    )


def test_versions_abbreviations_and_sentences_keep_their_dots():
    text = "Version 2.1.310, e.g. the U.S. build. Next sentence."

    assert spoken_form(text) == text
