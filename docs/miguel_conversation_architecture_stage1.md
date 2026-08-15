# Miguel Conversation Architecture — Stage 0/1

## Runtime audit

- Production runtime: `week3/camera/robot_cloud_brain_v7_5_queue.py`, the latest
  queue runtime referenced by the V7.5 checkpoint and debug tooling. The root
  `start_robot.sh` now launches this queue runtime as well, so the normal
  executable launcher and documented production entry point agree.
- Audio: `AudioWorker` calls the inherited
  `robot_cloud_brain_v6_threaded.capture_user_turn()`. That function owns one
  `arecord` raw PCM stream (or `DualMicStream` selecting chunks from two
  configured PulseAudio XVF sources), adaptive RMS speech detection, WAV
  capture, and final transcription.
- Wake: wake/name phrases are detected from the returned transcript in
  `_has_v7_5_wake_phrase()` and `_strip_wake_phrase()`. The older inherited
  `listen_for_wake()` has its own Vosk lifecycle but is not used by AudioWorker.
- Vosk: the model is process-global. Wake listening creates/reset a recognizer
  within that legacy function. Stage 1 creates one recognizer per authoritative
  user capture and feeds it the same PCM chunks for partials; no second audio
  stream exists. Existing final transcription remains unchanged (cloud when
  available, local Vosk fallback).
- Previous endpoint: adaptive RMS starts speech, then a fixed 0.9-second silence
  timeout (1.05 seconds for speech shorter than 1.25 seconds) ends capture. A
  process-wide maximum also ended active speech; Stage 1 retains it only as a
  no-input guard. Partial transcript semantics did not participate.
- TTS: legacy `v6.speak` is wrapped into `ReplyEvent`s. `SpeechWorker` alone
  prepares cloud/espeak audio and plays it, sequentially. Stop commands clear
  queued replies; the active playback backend is explicitly logged as not yet
  interruptible.
- Queues/threads: `AudioWorker` produces `UserTurnEvent`; `BrainWorker` routes;
  `SpeechWorker` consumes `ReplyEvent`; `FaceWorker` owns identity refresh;
  `CameraManager` owns OAK queues. Stage 1 makes conversation queues bounded.
- UI: `set_interaction_state()` and `notify_face_status()` debounce and map
  runtime state into the optional pygame face controller. END_CANDIDATE does
  not call the UI thinking state, so short pauses remain visually listening.
- Logging: `robot_memory.start_conversation_log_session()` creates a retained
  JSONL session in `week3/memory/conversation_logs`; append failures are logged.
  Turn playback already carried capture/route/TTS metrics. Stage 0 adds the
  monotonic endpoint timeline and derived EOT metrics.
- Configuration: runtime settings use `MIGUEL_*` environment variables via
  local typed helpers. `ConversationConfig.from_env()` follows that convention
  and centrally owns all Stage-1 thresholds.

## Existing routing after commit

The Conversation Manager ends at accepted completed-turn delivery. BrainWorker
continues to route, in its established order, wake/session handling, Teacher
Mode and times tables, story control and Long Story generation, enrollment and
identity, language and voice modes, timers, shutdown confirmation, repeat,
safety, camera requests, local utilities, and finally cloud conversation.

## Stage-1 flow

`IDLE -> ENGAGED -> LISTENING -> END_CANDIDATE -> PREPARING -> SPEAKING`

Speech resumption transitions `END_CANDIDATE -> LISTENING`. Floor ownership is
explicitly `NONE`, `HUMAN`, or `MIGUEL`; SpeechWorker must obtain Miguel's floor
grant before playback. Engagement is a continuously decaying score reinforced
by wake, accepted turns, and robot questions. A turn is addressable with an
explicit wake, sufficiently strong engagement, or an outstanding expected
reply. Existing wake-safe routing remains the fallback while idle.

## Hardware/AEC status at implementation time

USB enumeration found two Seeed Studio reSpeaker XVF3800 arrays (`2886:001a`)
and ALSA/PulseAudio exposed XVF capture and playback endpoints. PyUSB and the
upstream-compatible `libusb-package` backend are installed in the project venv.
The USB device nodes are currently owned by `root:root` with no write access for
the runtime user, so firmware/DoA/VAD control transfers report `EACCES` and
startup safely reports `available=false`. Install
`week3/config/99-miguel-xvf3800.rules` as a udev rule and replug the arrays to
enable live reads. The generic XKX USB soundbar is a separate
PulseAudio sink; no software loopback from that sink to an XVF playback/AEC
reference was visible. This is routing evidence, not a completed acoustic A/B/C
test. `miguel_aec_diagnostic.py` now records RMS, XVF VAD ratio, and DoA samples
for explicit manual A/B/C runs, saving raw audio only with `--save-wav`.

The default endpoint thresholds are 300 ms candidate, 500 ms strong completion,
750 ms normal completion, 1400 ms ambiguous/no-partial fallback, 2200 ms repair
consideration, and 3000 ms silence-only repair fallback. Active speech has no
total-duration cutoff. Set `MIGUEL_ENABLE_ADAPTIVE_ENDPOINT=false` to run the
legacy endpoint with the same Stage-0 monotonic instrumentation for comparison.

## XVF deployment

Install optional control dependencies with:

```bash
week3/camera/venv/bin/pip install -r week3/camera/requirements-conversation.txt
```

Install the scoped udev rule once, then replug both arrays:

```bash
sudo install -m 0644 week3/config/99-miguel-xvf3800.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=usb
```
