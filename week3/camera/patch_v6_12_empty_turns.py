from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

# Add empty turn counter in run loop.
text = text.replace(
    "last_reply_time = 0",
    "last_reply_time = 0\n    empty_followup_count = 0"
)

# Reset counter when real text is captured.
text = text.replace(
    'print(f"[TRANSCRIPT] {user_text}")',
    'empty_followup_count = 0\n                    print(f"[TRANSCRIPT] {user_text}")'
)

# Patch no-speech follow-up branches.
text = text.replace(
    '''                    if not user_text:
                        print("[AUDIO] No speech captured during follow-up grace.")
                        continue
''',
    '''                    if not user_text:
                        empty_followup_count += 1
                        print("[AUDIO] No speech captured during follow-up grace.")

                        if empty_followup_count >= 2:
                            print("[IDLE] Too many empty follow-up turns. Returning to wake mode.")
                            last_reply_time = 0
                            empty_followup_count = 0

                        continue
'''
)

path.write_text(text)
print("Patched V6.12 empty follow-up handling.")
