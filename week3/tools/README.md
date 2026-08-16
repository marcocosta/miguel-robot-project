# Miguel Tools

## Debug the latest robot session

`miguel-debug-last` finds the newest conversation JSONL log and creates a
two-phase handoff under `week3/debug_handoffs/`. Codex first analyzes the log
in a technically read-only sandbox. The report is shown to the developer, and
workspace-write access is launched only after the developer types the exact
token `APPLY`. The default, EOF, timeout, and every other response mean no.

Install the convenient command once:

```bash
ln -s /home/marquinho/robot-project/week3/tools/miguel_debug_last.sh /home/marquinho/bin/miguel-debug-last
```

Run it after stopping Miguel with Ctrl+C:

```bash
miguel-debug-last
```

Confirmed voice shutdowns launch the same read-only analysis automatically
after the final session log event is written. If a GUI terminal is available,
it shows the proposal and approval prompt without blocking robot shutdown. If
there is no GUI/TTY, analysis runs detached and implementation cannot start;
the launcher prints a handoff-specific command for later review:

```bash
miguel-debug-last --apply HANDOFF_ID
```

Useful safe modes:

```bash
miguel-debug-last --prepare-only
miguel-debug-last --analyze-only
```

`--prepare-only` launches no Codex process. `--analyze-only` always uses the
read-only sandbox and stops after saving the report. `--apply` still displays
the saved proposal, checks that branch/HEAD/worktree state are unchanged, and
requires `APPLY`; it is not a force mode.
