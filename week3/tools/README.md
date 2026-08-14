# Miguel Tools

## Debug the latest robot session

`miguel-debug-last` finds the newest conversation JSONL log, writes a timestamped prompt under `week3/debug_handoffs/`, and opens Codex with instructions to analyze, implement, and verify fixes.

Install the convenient command once:

```bash
ln -s /home/marquinho/robot-project/week3/tools/miguel_debug_last.sh /home/marquinho/bin/miguel-debug-last
```

Run it after stopping Miguel with Ctrl+C:

```bash
miguel-debug-last
```

Confirmed voice shutdowns launch the same handoff automatically after the final session log event is written. Use `--prepare-only` to generate a prompt without launching Codex.
