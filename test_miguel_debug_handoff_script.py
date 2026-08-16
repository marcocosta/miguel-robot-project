"""Process-level safety tests for the two-phase Miguel debug handoff."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import pytest


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "week3" / "tools" / "miguel_debug_last.sh"


@pytest.fixture
def handoff_env(tmp_path):
    logs = tmp_path / "logs"
    handoffs = tmp_path / "handoffs"
    logs.mkdir()
    (logs / "latest.jsonl").write_text('{"event_type":"session_end"}\n')
    invocation_log = tmp_path / "codex-invocations.log"
    fake_codex = tmp_path / "fake-codex"
    fake_codex.write_text(
        """#!/usr/bin/env bash
set -u
printf '%s\\n' "$*" >>"$FAKE_CODEX_INVOCATIONS"
output=""
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "-o" ]]; then output="$2"; shift 2; continue; fi
    shift
done
if [[ -n "$output" ]]; then
    printf 'Findings\\nSafe fake analysis.\\nExact files likely to change\\nnone\\n' >"$output"
fi
cat >/dev/null
"""
    )
    fake_codex.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "MIGUEL_CONVERSATION_LOG_DIR": str(logs),
            "MIGUEL_DEBUG_HANDOFF_DIR": str(handoffs),
            "MIGUEL_CODEX_BIN": str(fake_codex),
            "FAKE_CODEX_INVOCATIONS": str(invocation_log),
            "MIGUEL_DEBUG_APPROVAL_TIMEOUT_SECONDS": "1",
            "DISPLAY": "",
            "WAYLAND_DISPLAY": "",
        }
    )
    return env, handoffs, invocation_log


def run_handoff(env, *args, input_text=None):
    return subprocess.run(
        [str(SCRIPT), *args],
        cwd=ROOT,
        env=env,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=15,
    )


def handoff_id(output: str) -> str:
    match = re.search(r"Handoff: (\S+)", output)
    assert match, output
    return match.group(1)


def invocations(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.exists() else []


def test_analysis_prompt_and_process_are_strictly_read_only(handoff_env):
    env, handoffs, calls = handoff_env
    result = run_handoff(env, "--analyze-only")
    assert result.returncode == 0, result.stderr
    prompt = next(handoffs.glob("*_analysis_prompt.md")).read_text()
    assert "ANALYZE ONLY" in prompt
    assert "must not edit files, apply patches, commit, push, merge, create a PR" in prompt
    assert "Return proposed changes only" in prompt
    assert len(invocations(calls)) == 1
    assert "-s read-only -a never" in invocations(calls)[0]
    assert "workspace-write" not in invocations(calls)[0]
    metadata = json.loads(next(handoffs.glob("*_metadata.json")).read_text())
    assert metadata["analysis_completed"] is True


def test_prepare_only_launches_no_codex_and_keeps_report_capability(handoff_env):
    env, handoffs, calls = handoff_env
    result = run_handoff(env, "--prepare-only")
    assert result.returncode == 0
    assert list(handoffs.glob("*_analysis_prompt.md"))
    assert list(handoffs.glob("*_metadata.json"))
    assert invocations(calls) == []


@pytest.mark.parametrize("response", [None, "", "no\n", "n\n", "yes\n", "invalid\n"])
def test_no_or_missing_explicit_approval_never_invokes_writer(handoff_env, response):
    env, _handoffs, calls = handoff_env
    result = run_handoff(env, input_text=response)
    assert result.returncode == 0
    assert len(invocations(calls)) == 1
    assert all("workspace-write" not in call for call in invocations(calls))


def test_explicit_apply_is_the_only_path_to_writer_and_occurs_second(handoff_env):
    if shutil.which("script") is None:
        pytest.skip("util-linux script is required for a pseudo-TTY approval test")
    env, _handoffs, calls = handoff_env
    command = f"{SCRIPT}"
    result = subprocess.run(
        ["script", "-q", "-c", command, "/dev/null"],
        cwd=ROOT,
        env=env,
        input="APPLY\n",
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    recorded = invocations(calls)
    assert len(recorded) == 2
    assert "-s read-only -a never" in recorded[0]
    assert "-s workspace-write -a on-request" in recorded[1]


def test_detached_voice_path_can_only_start_read_only_analysis(handoff_env):
    env, _handoffs, calls = handoff_env
    result = run_handoff(env, "--voice")
    assert result.returncode == 0
    assert "No implementation will start automatically" in result.stdout
    deadline = time.time() + 5
    while time.time() < deadline and not invocations(calls):
        time.sleep(0.05)
    assert len(invocations(calls)) == 1
    assert "-s read-only -a never" in invocations(calls)[0]
    assert "workspace-write" not in invocations(calls)[0]


def test_stale_analysis_blocks_writer_before_approval(handoff_env):
    env, handoffs, calls = handoff_env
    analyzed = run_handoff(env, "--analyze-only")
    identifier = handoff_id(analyzed.stdout)
    metadata_path = next(handoffs.glob("*_metadata.json"))
    metadata = json.loads(metadata_path.read_text())
    metadata["head_sha"] = "stale-head"
    metadata_path.write_text(json.dumps(metadata))

    applied = run_handoff(env, "--apply", identifier)
    assert applied.returncode != 0
    assert "STALE ANALYSIS" in applied.stderr
    assert len(invocations(calls)) == 1
    assert all("workspace-write" not in call for call in invocations(calls))


def test_shell_control_places_writer_after_freshness_and_approval_checks():
    source = SCRIPT.read_text()
    review = source.index("review_and_maybe_apply()")
    body = source[review:source.index("\n}\n", review)]
    assert body.index("state_is_fresh") < body.index("request_approval") < body.index("run_implementation")
    assert '[[ "$response" != "APPLY" ]]' in source
    assert "nohup \"$SCRIPT_PATH\" --analyze-only" in source
    assert "nohup \"$CODEX_BIN\" exec" not in source
