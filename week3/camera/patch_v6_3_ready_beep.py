from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

# Insert beep helper before run_v6_threaded_conversation.
marker = "def run_v6_threaded_conversation():"
idx = text.find(marker)

if idx == -1:
    raise SystemExit("Could not find run_v6_threaded_conversation().")

beep_block = r'''
def ready_beep():
    """Short audible cue: Miguel is ready for the user to speak."""
    try:
        beep_path = Path(tempfile.gettempdir()) / "miguel_ready_beep.wav"

        # Generate a short beep using Python only.
        import wave as _wave
        import math as _math
        import struct as _struct

        sample_rate = 22050
        duration = 0.16
        frequency = 880
        volume = 0.22

        n_samples = int(sample_rate * duration)

        with _wave.open(str(beep_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)

            for i in range(n_samples):
                t = i / sample_rate
                # Tiny fade in/out to avoid click.
                fade = min(i / 800, (n_samples - i) / 800, 1.0)
                value = int(32767 * volume * fade * _math.sin(2 * _math.pi * frequency * t))
                wf.writeframes(_struct.pack("<h", value))

        subprocess.run(["aplay", "-q", "-D", SPEAKER_DEVICE, str(beep_path)], check=False)

    except Exception as e:
        print("[BEEP] Could not play ready beep:", e)


'''

if "def ready_beep():" not in text:
    text = text[:idx] + beep_block + "\n" + text[idx:]
else:
    print("ready_beep already exists; skipping insertion.")

# Add beep before every capture_user_turn() in run loop.
# We keep it close to the actual listen call so it means "talk now".
text = text.replace(
    'print(f"[READY] Familiar person present: {friendly_person}. Start talking now.")\n                    user_text = capture_user_turn()',
    'print(f"[READY] Familiar person present: {friendly_person}. Start talking now.")\n                    ready_beep()\n                    user_text = capture_user_turn()'
)

text = text.replace(
    'print("[READY] Start talking now.")\n\n                    local_state = get_cached_local_state()\n                    user_text = capture_user_turn()',
    'print("[READY] Start talking now.")\n                    ready_beep()\n\n                    local_state = get_cached_local_state()\n                    user_text = capture_user_turn()'
)

path.write_text(text)
print(f"Patched V6.3 ready beep into: {path}")
