from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from aloud.config import Config, load_config
from aloud.paths import AppPaths, default_paths
from aloud.registry import Registry
from aloud.text import signature, speech_chunks, spoken_form, to_gist


class Synthesizer(Protocol):
    def synthesize(self, text: str, output_path: os.PathLike[str] | str) -> bool: ...


class Player(Protocol):
    def play(self, path: os.PathLike[str] | str) -> Any: ...

    def stop(self) -> None: ...


@dataclass
class KokoroSynthesizer:
    config: Config

    def __post_init__(self) -> None:
        import numpy as np
        import soundfile as sf
        from kokoro import KPipeline

        self._np = np
        self._sf = sf
        self._pipeline = KPipeline(lang_code="a")

    def synthesize(self, text: str, output_path: os.PathLike[str] | str) -> bool:
        chunks = [
            audio
            for _, _, audio in self._pipeline(
                spoken_form(text),
                voice=self.config.voice,
                speed=self.config.speed,
            )
        ]
        if not chunks:
            return False
        output = os.fspath(output_path)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        self._sf.write(output, self._np.concatenate(chunks), self.config.sample_rate)
        return True


class AfplayPlayer:
    def __init__(self) -> None:
        self._process: subprocess.Popen[bytes] | None = None

    def play(self, path: os.PathLike[str] | str) -> subprocess.Popen[bytes]:
        self._process = subprocess.Popen(["afplay", os.fspath(path)])
        return self._process

    def stop(self) -> None:
        subprocess.run(
            ["killall", "afplay"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        self._process = None

    def wait(self, process: subprocess.Popen[bytes] | None = None) -> None:
        """Block until `process` (or the current playback) has ended, however it ended."""
        target = process or self._process
        if target is None:
            return
        target.wait()
        if self._process is target:
            self._process = None

    def is_playing(self) -> bool:
        return self._process is not None and self._process.poll() is None


class Daemon:
    """Turns registry entries into speech, one utterance at a time.

    Long texts are split into chunks: the first one is synthesised and started right away,
    the rest are synthesised on a worker thread while the previous chunk plays, so playback
    has no gaps and the command loop stays free to stop or replace it at any moment.
    """

    def __init__(
        self,
        paths: AppPaths | None = None,
        config: Config | None = None,
        registry: Registry | None = None,
        synthesizer: Synthesizer | None = None,
        player: Player | None = None,
        notifier: Callable[[str], None] | None = None,
    ):
        self.paths = paths or default_paths()
        self.config = config or load_config(self.paths)
        self.registry = registry or Registry(self.paths, self.config)
        self.synthesizer = synthesizer or KokoroSynthesizer(self.config)
        self.player = player or AfplayPlayer()
        self.notifier = notifier or notify
        self.current_priority: int | None = None
        self._generation = 0
        self._worker: threading.Thread | None = None
        self._synth_lock = threading.Lock()

    def handle(self, command: str) -> None:
        parts = command.strip().split(maxsplit=1)
        if not parts:
            return
        verb = parts[0]
        arg = parts[1] if len(parts) > 1 else ""
        if verb == "speak" and arg:
            self.speak(arg)
        elif verb in ("full", "play"):
            self.full()
        elif verb == "repeat":
            self.repeat()
        elif verb == "stop":
            self.stop_playback()
            self.current_priority = None
        elif verb == "forget":
            self.forget()
        elif verb == "last":
            self.last(arg or None)
        elif verb == "mute":
            self.set_muted(not self.muted if arg == "toggle" else True)
        elif verb == "unmute":
            self.set_muted(False)

    def full(self) -> None:
        self._play_text(self.registry.spoken_target().text, priority=0)

    def repeat(self) -> None:
        self._play_text(self.registry.spoken_attention_target().text, priority=0)

    def last(self, session_id: str | None = None) -> None:
        """Read a session's last finished reply in full, or the most recent one of any session."""
        session_id = session_id or self.registry.latest_session()
        text = self.registry.text_for(session_id)
        if text:
            self.registry.note_spoken(session_id)
        self._play_text(text, priority=0)

    @property
    def muted(self) -> bool:
        return self.paths.muted.exists()

    def set_muted(self, muted: bool) -> None:
        """Silence or resume automatic speech; explicit reads keep working either way."""
        if muted:
            self.stop_playback()
            self.current_priority = None
            self.paths.muted.parent.mkdir(parents=True, exist_ok=True)
            self.paths.muted.touch()
        else:
            with suppress(OSError):
                self.paths.muted.unlink()
        self.notifier("Aloud muted" if muted else "Aloud unmuted")

    def speak(self, session_id: str) -> None:
        full_text = self.registry.text_for(session_id)
        if not full_text or self.muted:
            return
        priority = self.registry.priority_for(session_id)
        attention_text = self.registry.attention_for(session_id)
        if not self._should_play(priority):
            return
        self.registry.note_spoken(session_id)
        self._play_text(
            attention_text or to_gist(full_text, self.config.gist_chars),
            quiet=True,
            priority=priority,
        )

    def forget(self) -> None:
        self.stop_playback()
        self.current_priority = None
        with suppress(OSError):
            for path in self.paths.cache_home.glob("last*.wav"):
                path.unlink()
            self.paths.signature.unlink()

    def stop_playback(self) -> None:
        """Cancel the chunk worker, if any, and silence the player."""
        self._generation += 1
        self.player.stop()

    def is_busy(self) -> bool:
        """True while a chunk worker is alive or the player reports sound."""
        worker = self._worker
        if worker is not None and worker.is_alive():
            return True
        is_playing = getattr(self.player, "is_playing", None)
        return bool(is_playing()) if callable(is_playing) else False

    def wait_idle(self) -> None:
        """Block until the chunk worker has finished; for tests and the self-test."""
        worker = self._worker
        if worker is not None:
            worker.join()

    def poll_transcripts(self) -> None:
        for session_id in self.registry.record_armed_transcript_events():
            if self.registry.is_armed(session_id):
                self.speak(session_id)

    def _play_text(self, text: str, quiet: bool = False, priority: int = 4) -> None:
        if not self._should_play(priority):
            return
        self.stop_playback()
        self.current_priority = priority
        if not text:
            if not quiet:
                self.notifier("No reply to read yet.")
            return
        chunks = speech_chunks(text, self.config.max_chars)
        outputs = [self._chunk_path(index) for index in range(len(chunks))]
        sig = signature(text)
        cached = self._read_signature() == sig and all(path.exists() for path in outputs)
        if not cached:
            self._clear_signature()
            if not self._synthesize(chunks[0], outputs[0]):
                return
        process = self.player.play(outputs[0])
        if len(chunks) == 1:
            if not cached:
                self._write_signature(sig)
            return
        self._worker = threading.Thread(
            target=self._continue,
            args=(chunks, outputs, cached, sig, self._generation, process),
            daemon=True,
        )
        self._worker.start()

    def _continue(
        self,
        chunks: list[str],
        outputs: list[Path],
        cached: bool,
        sig: str,
        generation: int,
        process: Any,
    ) -> None:
        previous = process
        for chunk, out in zip(chunks[1:], outputs[1:], strict=True):
            if generation != self._generation:
                return
            if not cached and not self._synthesize(chunk, out):
                return
            self._wait_for(previous)
            if generation != self._generation:
                return
            previous = self.player.play(out)
        if not cached:
            self._write_signature(sig)

    def _synthesize(self, text: str, out: Path) -> bool:
        with self._synth_lock:
            return self.synthesizer.synthesize(text, out)

    def _wait_for(self, process: Any) -> None:
        wait = getattr(self.player, "wait", None)
        if callable(wait):
            wait(process)

    def _chunk_path(self, index: int) -> Path:
        if index == 0:
            return self.paths.wav
        return self.paths.cache_home / f"last-{index + 1}.wav"

    def _should_play(self, priority: int) -> bool:
        if not self.is_busy() or self.current_priority is None:
            return True
        return priority <= self.current_priority

    def _read_signature(self) -> str | None:
        try:
            return self.paths.signature.read_text().strip()
        except OSError:
            return None

    def _write_signature(self, sig: str) -> None:
        self.paths.signature.parent.mkdir(parents=True, exist_ok=True)
        self.paths.signature.write_text(sig)

    def _clear_signature(self) -> None:
        with suppress(OSError):
            self.paths.signature.unlink()


def notify(message: str) -> None:
    subprocess.run(
        ["osascript", "-e", f'display notification "{message}" with title "Aloud"'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def serve(paths: AppPaths | None = None) -> None:
    paths = paths or default_paths()
    paths.ensure_runtime_dirs()
    if paths.socket.exists():
        paths.socket.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(paths.socket))
    server.listen(8)
    sys.stderr.write("aloud daemon ready\n")
    sys.stderr.flush()
    daemon = Daemon(paths=paths)
    threading.Thread(target=_poll_transcripts, args=(daemon,), daemon=True).start()
    while True:
        conn, _ = server.accept()
        try:
            data = conn.recv(256).decode("utf-8", "ignore")
            daemon.handle(data)
        finally:
            conn.close()


def _poll_transcripts(daemon: Daemon) -> None:
    while True:
        with suppress(Exception):
            daemon.poll_transcripts()
        time.sleep(1)
