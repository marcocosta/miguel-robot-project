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

## Stage 2A characterization and capture infrastructure

The XVF3800 control interface was subsequently validated on hardware at USB
VID `0x2886`, PID `0x001a`, firmware 2.0.6. `VERSION`, `DOA_VALUE`,
`AEC_AZIMUTH_VALUES`, and `AEC_SPENERGY_VALUES` are supported. Basic VAD and
DoA characterization passed. Persistent-connection physical calibration is
approximately front 177 degrees, robot-left 270 degrees, and robot-right 75
degrees. DoA collected while XVF VAD is false is diagnostic telemetry, not
valid human-direction evidence.

The self-speech/AEC gate failed: human-only speech produced VAD ratio 1.00;
Miguel-only TTS through the actual USB soundbar produced 0.97; and a human
speaking over Miguel produced 1.00. Miguel-only TTS had RMS mean 1348.2, p95
4179, peak 6017, with stable DoA around 269--270 degrees. Human-only speech had
RMS mean 1566.37, p95 4682, peak 13726, around 176--177 degrees. Combined speech
had RMS mean 2230.62, p95 5845, peak 12920, shifting toward 178--187 degrees.

Automatic barge-in is therefore explicitly blocked. A likely explanation is
that the separate USB soundbar does not give the XVF hardware an adequate
digital reference for AEC; this is a hypothesis, not an established fact.
Stage 2B may later use XVF VAD/DoA as passive evidence while Miguel is not
speaking, but no XVF signal changes conversation behavior in Stage 2A. Stage
2C barge-in requires a separate, validated self-speech suppression solution
and gate.

V7.5 now owns conversation-aware capture through `miguel_audio_capture.py`.
That component owns the PCM loop, RMS voice events, available Vosk partials,
adaptive endpoint decisions, and final-ASR handoff to `ConversationManager`.
It reuses established low-level device/transcription helpers but never calls
or adds parameters to historical V6 `capture_user_turn()`.

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

## Stage 2B: shadow spatial-audio evidence

Stage 2B adds an optional, independent `XVFWorker` and
`miguel_spatial_audio.py`. The worker polls the persistent XVF control monitor
at 10 Hz by default and retains only VAD-positive DoA samples in a rolling
0.7-second window. It computes a circular mean and resultant-vector magnitude,
so directions across the 0/360-degree boundary are handled correctly. At
least three samples and a resultant of 0.85 are required before direction is
classified as stable. These calibration values are configurable with
`MIGUEL_XVF_POLL_HZ`, `MIGUEL_XVF_WINDOW_SECONDS`,
`MIGUEL_XVF_MIN_STABLE_SAMPLES`, and `MIGUEL_XVF_STABLE_RESULTANT`.

Front calibration defaults to the measured 177 degrees and can be set with
`MIGUEL_XVF_FRONT_AZIMUTH_DEG`. Relative angle is
`((raw - front + 180) % 360) - 180`. The current installation therefore
observes front near 0, physical left near +93, and physical right near -102
degrees. Sign is not yet mapped to OAK-D camera coordinates. Stable evidence
within 15 degrees is `FRONT_STRONG`, within 30 degrees is `FRONT_POSSIBLE`,
and wider angles are `OFF_AXIS`; the gates are configurable.

VAD-false samples are `NO_SPEECH`: their retained firmware DoA is never added
to the rolling window or treated as a person. While Miguel owns the floor or
TTS is active, samples are marked `ROBOT_SPEAKING_SUPPRESSED`, the human DoA
window is cleared, and a modest configurable post-TTS settling guard applies.
This is required because Stage 2A found Miguel-only VAD at 0.97. The unresolved
self-speech/AEC failure continues to block automatic barge-in.

`ConversationManager` owns only the latest immutable snapshot. It emits one
`ADDRESSEE_SHADOW` diagnostic when the existing addressee decision is made,
but spatial evidence cannot change that decision, engagement, floor ownership,
endpointing, wake handling, ASR, routing, or TTS. USB/PyUSB failure degrades to
`UNKNOWN`; PCM capture and conversation continue independently. Sensor/error
logs and reconnects are rate-limited.

Both identical `2886:001a` devices are enumerated in deterministic bus/address
order. `MIGUEL_XVF_DEVICE_INDEX` selects an index (default 0), while
`MIGUEL_XVF_SERIAL` can select a unique serial when one is readable. Startup
and session-end diagnostics record count, index, bus, address, serial,
firmware, and whether selection remains ambiguous. The ambiguity is acceptable
only because this release is shadow mode. Further calibration may refine these
gates; Stage 3, not Stage 2B, owns future OAK-D/audio spatial fusion. Stage 2C
remains the separate controlled-barge-in stage and is not implemented here.

The Stage 2B hardware smoke successfully used index 0 at bus 1, address 17,
serial `114993701262100545`. Because USB address/index can change when two
identical devices are attached, follow-up calibration should pin this unit with
`MIGUEL_XVF_SERIAL=114993701262100545`; the serial is an operator setting, not
a source-code default. Front speech in that smoke appeared around 189--195
degrees rather than the earlier 177-degree calibration. The default remains
177 and configurable until several front utterances are measured with the
pinned device; placement, device orientation, and test geometry are still
possible explanations.

Instantaneous evidence and turn evidence have separate lifetimes. Every
`begin_listening()` clears the new turn latch. While LISTENING or END_CANDIDATE,
the manager retains the first and last stable, VAD-positive, unsuppressed
snapshots. Later `NO_SPEECH` does not erase them. The final latched snapshot is
copied into the queued `UserTurnEvent` and user-turn JSON diagnostics, so the
next capture cannot reattribute evidence asynchronously. Suppressed and
post-TTS-settling evidence is never eligible for the latch. This remains
diagnostic-only and cannot change the baseline addressee result.
