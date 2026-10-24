"""Streamed playback: the first chunk plays at once, later chunks follow without a gap, a stop
or a higher-priority utterance takes effect immediately, and a repeat replays from cache."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

from aloud.attention import normalize_attention_event
from aloud.config import Config
from aloud.daemon import Daemon
from aloud.registry import Registry
from aloud.text import signature, speech_chunks

CONFIG = Config(max_chars=40, reply_speech="full")
LONG_REPLY = (
    "First sentence of the reply here. Second sentence of the reply here. "
    "Third sentence of the reply here. Fourth sentence ends the reply here."
)
LONG_SPEECH = "Claude aloud completed. " + LONG_REPLY
CHUNKS = speech_chunks(LONG_SPEECH, CONFIG.max_chars)


class RecordingSynth:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text: str, output_path: Path) -> bool:
        self.calls.append(text)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"wav")
        return True


class GatedPlayer:
    """A player whose current chunk only ends when the test lets it, or on stop."""

    def __init__(self) -> None:
        self.played: list[str] = []
        self.stops = 0
        self.active = False
        self._ended = threading.Event()

    def play(self, path: Path) -> str:
        self.active = True
        self.played.append(Path(path).name)
        self._ended.clear()
        return Path(path).name

    def wait(self, process: str | None = None) -> None:
        self._ended.wait(timeout=5)

    def stop(self) -> None:
        self.active = False
        self.stops += 1
        self._ended.set()

    def is_playing(self) -> bool:
        return self.active

    def finish_current(self) -> None:
        self._ended.set()


def wait_until(condition: Callable[[], bool], seconds: float = 2.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.005)


def _played(player: GatedPlayer, count: int) -> Callable[[], bool]:
    return lambda: len(player.played) == count


def _synthesised(synth: RecordingSynth, count: int) -> Callable[[], bool]:
    return lambda: len(synth.calls) == count


def _chunk_names(count: int) -> list[str]:
    return ["last.wav", *(f"last-{index}.wav" for index in range(2, count + 1))]


def _play_through(daemon: Daemon, player: GatedPlayer, already: int, count: int) -> None:
    """Release the chunks of one utterance one by one until all `count` have played."""
    for played in range(already + 2, already + count + 1):
        player.finish_current()
        wait_until(_played(player, played))
    daemon.wait_idle()


def _daemon(isolated_env) -> tuple[Daemon, RecordingSynth, GatedPlayer, Registry]:
    registry = Registry(isolated_env, CONFIG)
    synth = RecordingSynth()
    player = GatedPlayer()
    daemon = Daemon(
        paths=isolated_env, config=CONFIG, registry=registry, synthesizer=synth, player=player
    )
    return daemon, synth, player, registry


def _record(registry: Registry, session_id: str, text: str) -> None:
    event = normalize_attention_event(
        {
            "source": "Claude",
            "hook_event_name": "Stop",
            "session_id": session_id,
            "cwd": "/repo/aloud",
            "last_assistant_message": text,
        },
        CONFIG,
    )
    assert event
    registry.record_attention(event)


def test_the_fixture_text_needs_several_chunks():
    assert len(CHUNKS) >= 3


def test_first_chunk_plays_at_once_and_later_chunks_follow_in_order(isolated_env):
    daemon, synth, player, registry = _daemon(isolated_env)
    _record(registry, "SID-A", LONG_REPLY)

    daemon.speak("SID-A")

    assert player.played == ["last.wav"]
    wait_until(_synthesised(synth, 2))  # chunk two is ready while chunk one still plays
    assert player.played == ["last.wav"]
    _play_through(daemon, player, 0, len(CHUNKS))
    assert player.played == _chunk_names(len(CHUNKS))
    assert synth.calls == CHUNKS
    assert isolated_env.signature.read_text() == signature(LONG_SPEECH)


def test_stop_during_the_first_chunk_cancels_the_rest(isolated_env):
    daemon, synth, player, registry = _daemon(isolated_env)
    _record(registry, "SID-A", LONG_REPLY)
    daemon.speak("SID-A")
    wait_until(_synthesised(synth, 2))

    daemon.handle("stop")
    daemon.wait_idle()

    assert player.played == ["last.wav"]
    assert len(synth.calls) == 2
    assert daemon.current_priority is None
    assert not isolated_env.signature.exists()


def test_a_higher_priority_utterance_interrupts_a_running_sequence(isolated_env):
    daemon, synth, player, registry = _daemon(isolated_env)
    _record(registry, "SID-A", LONG_REPLY)
    daemon.speak("SID-A")
    wait_until(_synthesised(synth, 2))
    permission = normalize_attention_event(
        {
            "source": "Claude",
            "hook_event_name": "PermissionRequest",
            "tool_name": "Bash",
            "session_id": "SID-B",
            "cwd": "/repo/b",
            "command": "ls",
        },
        CONFIG,
    )
    assert permission
    registry.record_attention(permission)

    daemon.speak("SID-B")
    permission_chunks = speech_chunks(permission.speech_text, CONFIG.max_chars)
    _play_through(daemon, player, 1, len(permission_chunks))  # one chunk played before

    assert player.stops >= 1
    assert synth.calls[2:] == permission_chunks
    assert "Third sentence" not in " ".join(synth.calls)


def test_a_lower_priority_reply_waits_while_a_question_plays(isolated_env):
    daemon, synth, player, registry = _daemon(isolated_env)
    _record(registry, "SID-A", "Short reply. Which one do I use?")  # one chunk at 40 chars
    daemon.speak("SID-A")
    question_calls = len(synth.calls)
    _record(registry, "SID-B", "Another reply that finished without a question.")

    daemon.speak("SID-B")

    assert len(synth.calls) == question_calls
    assert player.played == _chunk_names(question_calls)


def test_repeat_of_the_same_text_replays_from_cache(isolated_env):
    daemon, synth, player, registry = _daemon(isolated_env)
    _record(registry, "SID-A", LONG_REPLY)
    daemon.speak("SID-A")
    _play_through(daemon, player, 0, len(CHUNKS))
    synthesised = len(synth.calls)

    daemon.repeat()
    _play_through(daemon, player, len(CHUNKS), len(CHUNKS))

    assert len(synth.calls) == synthesised
    assert player.played[len(CHUNKS) :] == _chunk_names(len(CHUNKS))
