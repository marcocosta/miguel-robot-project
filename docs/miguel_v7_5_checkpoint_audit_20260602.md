# Miguel V7.5 Checkpoint Audit - 2026-06-02

## Scope

This audit was triggered after the USB speaker change exposed a voice-mode regression in `start-robot-cloud-v7-5`.

The launch path is:

- `/home/marquinho/bin/start-robot-cloud-v7-5`
- `week3/camera/robot_cloud_brain_v7_5_queue.py`
- V7.5 delegates TTS and several legacy primitives through `robot_cloud_brain_v7_full.py -> robot_cloud_brain_v6_threaded.py`

## Findings

- The startup greeting was not the problem. It is still enabled in V7.5:
  `v6.speak("I am Miguel, Marquinho's robot project. Camera and face recognition are online.")`
- The USB speaker regression was caused by direct ALSA playback fighting PulseAudio for the new USB speaker.
- The voice-mode regression was not caused by the speaker change. Git history shows the unsupported voice branch entered in commit `2b0bfe28` (`Polish Miguel V7.15 command and voice routing`, 2026-05-28).
- The older working voice-mode implementation was still present in `robot_memory.py`. V7.5 was shadowing it with a local router that treated deep, robot, story, and friendly voices as unsupported.
- Recent logs confirm the bad behavior in `week3/memory/conversation_logs/20260602_203156.jsonl`.

## Fixes Applied

- V7.5 voice-mode routing now delegates to `robot_memory.handle_voice_mode_command()`.
- Voice modes verified locally:
  - robot voice -> `robot_voice`
  - natural voice -> `natural_voice`
  - friendly voice -> `friendly_voice`
  - deep voice -> `deep_voice`
  - story voice -> `story_voice`
- Current saved voice mode after audit: `deep_voice`.
- Speaker output now uses PulseAudio with the USB2.0 speaker sink:
  - `MIGUEL_SPEAKER_DEVICE="pulse"`
  - `MIGUEL_PULSE_SINK="alsa_output.usb-Generic_USB2.0_Device_20121120222012-00.analog-stereo"`

## Capability Audit

Static/code-path checks:

- V7.5 queue runner still starts CameraManager, FaceWorker, SpeechWorker, BrainWorker, and AudioWorker.
- Camera and identity routes still call CameraManager-backed helpers.
- Safety guard routing is still present through `SafetyGuard`, `should_run_safety_guard()`, and output safety in speech worker.
- Cloud replies still route through V6/OpenAI helpers.
- Conversation modes and depth modes remain in V7.5.
- Conversation logs are active through `robot_memory.py`.
- V6 TTS config still maps `natural_voice`, `friendly_voice`, `deep_voice`, `story_voice`, and `robot_voice`.

Runtime evidence from logs:

- Startup greeting logged on 2026-06-01 and 2026-06-02.
- Creative/story modes produced long story outputs on 2026-06-01.
- The voice-mode unsupported reply appeared in the 2026-06-02 log and was traced to V7.5 local routing.

## Verification

Passed:

- `python3 -m py_compile` for V6, V7 Full, V7.5, memory, camera intents, safety classifier, and safety guard.
- `git diff --check`
- Local voice-mode handler checks.
- USB2.0 speaker playback through PulseAudio sink.

Not run:

- Pytest suite. `pytest` is not installed in the system Python or project venv.
- Full live robot regression test. Requires running `start-robot-cloud-v7-5` and speaking commands.

## Residual Risk

- The repo still has pre-existing uncommitted work in `robot_cloud_brain_v7_5_queue.py`, `robot_memory.py`, and `v7/camera_intents.py`. Those changes appear related to long-story routing, conversation logs, and camera intent improvements, not the speaker fix.
- V7.5 has several local routers that can shadow older V6/V7 behavior. Voice modes were one example. Future regressions should be checked by comparing V7.5 local routes against the original V6/V7 implementations.

## Recommended Live Smoke Test

After restarting `start-robot-cloud-v7-5`, test:

1. Startup greeting plays through the USB2.0 speaker.
2. `Hi Miguel, how are you?`
3. `Miguel, use deep voice.`
4. `What voice mode are you using?`
5. `Miguel, use story voice.`
6. `Miguel, what do you see?`
7. `Miguel, who am I?`
8. `Miguel, go to creative mode.`
9. `Miguel, tell a short story about Marco, Marquinho, and Miguel.`
10. A normal safety/cloud question.
