from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest

import aloud.hooks as hooks_module
from aloud.config import Config
from aloud.hooks import event_hook, prompt_hook, stop_hook
from aloud.registry import Registry


def test_registry_target_follows_spoken_session_not_newest(isolated_env):
    registry = Registry(isolated_env, Config())

    registry.record_stop({"session_id": "SID-A", "last_assistant_message": "Alpha reply."})
    registry.arm("SID-A")
    time.sleep(0.01)
    registry.record_stop({"session_id": "SID-B", "last_assistant_message": "Beta reply."})
    registry.note_spoken("SID-A")

    assert registry.spoken_target().text == "Alpha reply."
    assert registry.resolve_target().text == "Beta reply."


def test_session_ids_cannot_escape_session_directory(isolated_env):
    registry = Registry(isolated_env, Config())
    sid = "../bad/session"

    assert registry.record_stop({"session_id": sid, "last_assistant_message": "Safe."}) == sid

    files = list(isolated_env.sessions.glob("*.json"))
    assert len(files) == 1
    assert files[0].parent == isolated_env.sessions
    assert registry.text_for(sid) == "Safe."


def test_prompt_hook_blocks_control_phrase_and_writes_marker(isolated_env):
    decision = prompt_hook({"session_id": "SID-A", "prompt": "/aloud-on"}, isolated_env)

    assert decision and decision["decision"] == "block"
    assert (isolated_env.armed / "SID-A").exists()


def test_stop_hook_sends_speak_only_for_armed_session(isolated_env):
    short_socket = Path("/tmp") / f"aloud-test-{os.getpid()}.sock"
    isolated_env = type(isolated_env)(
        app_home=isolated_env.app_home,
        cache_home=isolated_env.cache_home,
        log_home=isolated_env.log_home,
        socket=short_socket,
        config=isolated_env.config,
        sessions=isolated_env.sessions,
        armed=isolated_env.armed,
        latest=isolated_env.latest,
        spoken=isolated_env.spoken,
        wav=isolated_env.wav,
        signature=isolated_env.signature,
        log=isolated_env.log,
        launch_agent=isolated_env.launch_agent,
        muted=isolated_env.muted,
    )
    isolated_env.socket.parent.mkdir(parents=True, exist_ok=True)
    received: list[str] = []
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(isolated_env.socket))
    server.listen(1)
    server.settimeout(3)

    def accept_once():
        conn, _ = server.accept()
        with conn:
            received.append(conn.recv(256).decode("utf-8"))

    thread = threading.Thread(target=accept_once)
    thread.start()
    try:
        prompt_hook({"session_id": "SID-A", "prompt": "aloud on"}, isolated_env)
        sid = stop_hook(
            {"session_id": "SID-A", "last_assistant_message": "Ready."},
            isolated_env,
        )
        thread.join(3)
    finally:
        server.close()
        isolated_env.socket.unlink(missing_ok=True)

    assert sid == "SID-A"
    assert received == ["speak SID-A"]
    assert json.loads((isolated_env.sessions / "SID-A.json").read_text())["text"] == "Ready."


def _stop(session_id: str, text: str) -> dict:
    return {
        "source": "Claude",
        "hook_event_name": "Stop",
        "session_id": session_id,
        "cwd": "/repo/aloud",
        "last_assistant_message": text,
    }


def _recorder(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    sent: list[str] = []
    monkeypatch.setattr(hooks_module, "send_command", lambda command, *a, **k: sent.append(command))
    return sent


def test_unarmed_sessions_keep_their_last_reply_without_speaking(isolated_env, monkeypatch):
    sent = _recorder(monkeypatch)
    registry = Registry(isolated_env, Config())

    assert event_hook(_stop("SID-Q", "Quiet reply. Nothing spoken."), isolated_env) is None

    assert registry.text_for("SID-Q") == "Quiet reply. Nothing spoken."
    assert registry.resolve_target().text == "Quiet reply. Nothing spoken."
    assert registry.latest_session() == "SID-Q"
    assert sent == []


def test_unarmed_sessions_do_not_keep_permission_requests(isolated_env, monkeypatch):
    _recorder(monkeypatch)
    payload = {
        "source": "Claude",
        "hook_event_name": "PermissionRequest",
        "tool_name": "Bash",
        "session_id": "SID-P",
        "cwd": "/repo/aloud",
        "command": "ls",
    }

    assert event_hook(payload, isolated_env) is None
    assert Registry(isolated_env, Config()).text_for("SID-P") == ""


@pytest.mark.parametrize(
    ("prompt", "command", "reason"),
    [
        ("aloud last", "last SID-A", "last reply"),
        ("/aloud-last", "last SID-A", "last reply"),
        ("aloud mute", "mute", "muted"),
        ("aloud unmute", "unmute", "unmuted"),
    ],
)
def test_prompt_hook_relays_last_mute_and_unmute(
    isolated_env, monkeypatch, prompt, command, reason
):
    sent = _recorder(monkeypatch)

    decision = prompt_hook({"session_id": "SID-A", "prompt": prompt}, isolated_env)

    assert decision and decision["decision"] == "block"
    assert reason in decision["reason"]
    assert sent == [command]
    assert not (isolated_env.armed / "SID-A").exists()
