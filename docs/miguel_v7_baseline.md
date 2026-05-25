# Miguel V7 Baseline

Last updated: 2026-05-23

## Official Run Commands

Run Miguel V7 Full:

```bash
start-robot-cloud-v7-full
```

Run Miguel V7 Full with the display face disabled:

```bash
MIGUEL_FACE_ENABLED=0 start-robot-cloud-v7-full
```

Direct Python execution is only documented for syntax checking:

```bash
python3 -m py_compile /home/marquinho/robot-project/week3/camera/robot_cloud_brain_v7_full.py
```

## High-Level Architecture

Miguel V7 Full is the current voice-and-vision robot runtime. The active entry point is:

- `/home/marquinho/robot-project/week3/camera/robot_cloud_brain_v7_full.py`

The runtime starts the OAK-D Lite camera through DepthAI, uses one `CameraManager` as the only owner of camera frames, runs face recognition through V6/InsightFace helpers, listens through the ReSpeaker microphone, transcribes user turns, speaks through OpenAI TTS or local `espeak`, routes camera questions to live camera truth, uses safety checks for normal conversation, and uses cloud reasoning for general replies.

V7 Full reuses `robot_cloud_brain_v6_threaded.py` as a library of stable primitives. V6 still provides audio capture, transcription, TTS, InsightFace recognition helpers, cloud conversation, scene description helpers, memory update helpers, and several compatibility paths.

## Current Robot Capabilities

Miguel can currently:

- Wake/respond to phrases like `Miguel`, `Hey Miguel`, `Hey me go`, and `Mission Control`.
- Recognize known people from camera embeddings under `week3/face_embeddings/`.
- Recognize enrolled people such as `marco` and `marquinho`.
- Enter conversation-ready mode when a familiar face is visible.
- Require a wake phrase when no familiar face is recognized.
- Answer identity questions such as "who do you see", "do you recognize me", and "who am I" from live face-recognition state only.
- Answer scene questions such as "what do you see", "describe the scene", and "camera view" from a fresh camera frame and cloud vision.
- Avoid claiming camera truth from stale conversation memory.
- Speak with different voice modes: robot, natural, friendly, deep, and story.
- Switch personality modes: mission control, creative, teacher, engineer, and quiet.
- Enter sleep mode and wake up later.
- Stop the robot program after shutdown confirmation while keeping the Jetson on.
- Answer local skills: time, date, weather, internet/network status, project status, and simple calculations.
- Remember profile notes and preferences.
- Save, resume, list, and forget long-term topics.
- Keep short conversational context for follow-ups like "yes", "ok", and "continue".
- Use safety guardrails for user input and spoken output.
- Show an optional animated pygame display face that follows robot state.

## Active Runtime Files

### `week3/camera/robot_cloud_brain_v7_full.py`

Current main runtime and V7 Full loop.

Important functions:

- `init_optional_face()` starts the pygame face display unless `MIGUEL_FACE_ENABLED=0`.
- `set_face_once()` prevents repeated duplicate face mode updates.
- `face_idle()`, `face_listening()`, `face_thinking()`, `face_speaking()`, `face_happy()`, `face_confused()`, `face_error()`, and `face_sleeping()` send state changes to the display face.
- `has_wake_phrase()` checks Miguel wake phrases.
- `install_output_safety()` wraps V6 `speak()` so spoken output is safety-checked.
- `build_scene_reply()` saves a fresh frame and asks cloud vision to describe it.
- `build_identity_reply()` answers identity questions from fresh face state.
- `face_worker()` repeatedly runs face recognition and stores results in `CameraManager`.
- `is_local_robot_control_request()` identifies commands like voice, sleep, shutdown, status, and time.
- `handle_v7_local_utility()` handles small V7-only utilities, currently local time.
- `route_camera_intent_now()` routes camera, identity, and scene requests before safety/cloud brain.
- `handle_user_turn()` is the main per-turn router.
- `create_camera_manager_from_live_pipeline()` builds and starts the DepthAI camera pipeline.
- `run_v7_full()` is the main program loop.

### `week3/camera/robot_cloud_brain_v6_threaded.py`

Older V6 threaded runtime, still reused by V7 Full as stable primitives.

Important function groups:

- Audio/TTS: `speak()`, `speak_with_openai_tts()`, `speak_with_espeak()`, `capture_user_turn()`, `transcribe_audio_openai()`, `listen_for_wake()`.
- Face recognition: older LBP helpers plus the newer InsightFace override via `detect_face_state()`.
- Cloud brain: `ask_cloud_brain()` sends structured local state, memory, and conversation context to OpenAI.
- Conversation routing: `handle_user_turn_with_cached_state()` processes robot modes, memory, enrollment, identity, local skills, and cloud brain.
- Vision truth firewall: `sanitize_camera_memory_state()`, `safe_reply_after_camera_firewall()`, `is_camera_truth_state()`.
- Scene vision: `capture_fresh_camera_frame()`, `describe_scene_with_openai()`, `describe_scene_now()`.
- V6 standalone loop: `run_v6_threaded_conversation()`.

### `week3/camera/v7/camera_manager.py`

Single owner of OAK-D Lite camera frames.

Main pieces:

- `FrameSnapshot`: stores a frame and timestamp.
- `LatestFrameMessage`: adapter exposing `getCvFrame()` so old V6 helpers can consume V7 frames.
- `CameraManager`: owns the DepthAI queue and latest-frame state.
- `start()` and `stop()` manage the camera thread.
- `_run()` continuously stores the newest frame.
- `_get_newest_raw_msg()` drains camera queues and keeps the newest frame.
- `get_latest_frame()` returns a fresh frame copy.
- `get_latest_message()`, `get()`, `tryGet()`, and `tryGetAll()` provide queue-like compatibility.
- `save_latest_frame()` writes debug snapshots.
- `update_face_state()` stores face recognition results.
- `get_face_state()` returns recent face state or a stale/empty state.

### `week3/camera/v7/camera_intents.py`

Routes camera-related language.

Functions:

- `normalize_text()` lowercases and strips input.
- `is_identity_camera_request()` catches "who am I", "do you recognize me", and similar requests.
- `is_scene_camera_request()` catches "what do you see", "describe the scene", and similar requests.
- `is_any_camera_request()` provides a broad camera/vision fallback.
- `classify_camera_intent()` returns `identity_camera`, `scene_camera`, `camera_generic`, or `none`.

### `week3/camera/v7/safety_guard.py`

Safety policy router.

Main pieces:

- `SafetyDecision`: normalized allow/block result.
- `SafetyGuard`: combines semantic classifier, moderation hard-block signals, and output guard.
- `evaluate_user_text()` checks user input.
- `evaluate_assistant_reply()` checks spoken output.
- `_route_semantic_decision()` maps classifier results into final actions.
- `_moderation_hard_block()` checks OpenAI moderation hard categories.
- `_safe_redirect()` creates short family-safe refusal text.

### `week3/camera/v7/safety_classifier.py`

Semantic safety classifier using OpenAI.

Main pieces:

- `SemanticSafetyResult`: raw classifier result.
- `SemanticSafetyClassifier`: classifies intent rather than isolated words.
- `classify()` applies fast local allows, model classification, then fallback.
- `_classify_with_model()` asks OpenAI for JSON safety decision.
- `_is_obviously_safe_smalltalk()` fast-allows greetings/simple conversation.
- `_is_harmless_meaning_question()` allows phrase meaning questions.
- `_is_robot_control()` allows robot commands.
- `_is_obviously_benign_project_context()` allows Miguel debugging/project talk.
- `_contains_direct_harmful_request()` detects direct unsafe instruction requests.
- `_fallback_intent_check()` is the conservative fallback if classifier fails.

## Skills And Memory

### `week3/camera/robot_skills.py`

Local skills handled before cloud conversation.

Functions:

- `get_time_text()` answers current local time.
- `get_date_text()` answers today's date.
- `get_project_status_text()` summarizes Miguel system status.
- `check_internet_text()` checks connectivity to `8.8.8.8`.
- `get_weather_text()` fetches weather from `wttr.in`, defaulting to `MIGUEL_LOCATION` or Los Gatos.
- `safe_calculate_text()` evaluates simple arithmetic after filtering input.
- `maybe_handle_local_skill()` dispatches time, date, weather, network, status, and calculation requests.

### `week3/camera/robot_memory.py`

Persistent JSON memory at:

- `/home/marquinho/robot-project/week3/memory/miguel_memory.json`

Function groups:

- Storage: `load_memory()`, `save_memory()`.
- Modes: `get_robot_mode()`, `set_robot_mode()`, `get_personality_mode()`, `set_personality_mode()`.
- Shutdown: `set_pending_shutdown()`, `get_pending_shutdown()`.
- Profiles: `add_profile_note()`, `add_preference()`, `get_memory_context()`.
- Long-term topics: `create_or_update_long_term_topic()`, `set_active_long_term_topic()`, `forget_long_term_topic()`, `list_long_term_topics()`, `append_turn_to_active_topic()`, `handle_long_term_topic_command()`.
- Enrollment security: `unlock_enrollment()`, `clear_enrollment_unlock()`, `enrollment_is_unlocked()`, `handle_enrollment_security_command()`.
- Voice modes: `normalize_voice_mode()`, `handle_voice_mode_command()`.
- Robot/personality modes: `handle_robot_mode_command()` handles sleep, wake, shutdown confirmation, personality modes, memory commands, and mode questions.

## Display Face

### `week3/face/face_controller.py`

Small API used by Miguel runtime.

- `FaceController.start()` and `FaceController.stop()` manage the display thread.
- `set_mode()` sends a `FaceEvent`.
- `idle()`, `listening()`, `thinking()`, `speaking()`, `happy()`, `confused()`, `sleeping()`, and `error()` are convenience methods.

### `week3/face/face_state.py`

Display state definitions.

- `FaceMode`: enum of face modes.
- `FaceEvent`: dataclass carrying mode, optional text, and mouth level.

### `week3/face/face_thread.py`

Pygame renderer.

- `FaceThread.start()`, `stop()`, and `join()` manage rendering.
- `_run()` owns the pygame loop.
- `_render()` draws current mode.
- `_eye_color()`, `_draw_eyes()`, `_draw_mouth()`, `_draw_sleeping()`, `_draw_text()`, `_draw_glow_oval()`, and `_blink_amount()` create the animated robot face.

## Face Enrollment And Recognition Tools

### `week3/camera/enroll_face_insight_headless.py`

Headless enrollment tool for InsightFace embeddings. It takes a person name argument, captures camera frames, extracts embeddings, and saves `.npy` files plus preview JPGs.

### `week3/camera/recognize_face_insight_headless.py`

Headless recognition test. It loads embeddings, watches the camera, and prints recognition attempts.

### Other enrollment tools

- `week3/camera/enroll_face.py`
- `week3/camera/enroll_face_insight.py`

These are older or alternate enrollment utilities.

## Camera Test Utilities

- `week3/camera/oak_capture.py`: captures one OAK-D Lite image to `oak_first_capture.jpg`.
- `week3/camera/oak_preview.py`: live OpenCV preview window.
- `week3/camera/oak_face_detect.py`: live Haar-cascade face detection preview.
- `week3/camera/oak_face_tracking.py`: OAK face tracking/debug utility.
- `week3/camera/test_depthai_v7_queue.py`: DepthAI queue behavior test for V7 camera handling.

## Older Robot Entrypoints

These are older generations or alternate experiments:

- `week3/camera/robot_cloud_brain.py`
- `week3/camera/robot_cloud_brain_v2.py`
- `week3/camera/robot_cloud_brain_v3.py`
- `week3/camera/robot_cloud_brain_v4.py`
- `week3/camera/robot_cloud_brain_v5.py`
- `week3/camera/robot_cloud_brain_v7.py`

The current active version is `robot_cloud_brain_v7_full.py`. V6 threaded remains important because V7 imports it for audio, TTS, face recognition, scene description, memory update helpers, and cloud conversation.

## Historical And Patch Files

Most files named `patch_*`, `fix_*`, and `*_backup_before_*` are migration scripts or snapshots from previous changes. They are useful for history, but they are not the clean current runtime path.

Examples:

- `patch_v6_*`: incremental V6 behavior changes.
- `patch_v7_*`: V7 camera, safety, or routing updates.
- `fix_*`: one-off repair scripts.
- `*_backup_before_*`: saved copies before patches.

## Data And Runtime Folders

- `week3/audio/`: recorded test and runtime WAV files.
- `week3/debug_snapshots/`: saved camera frames for scene description/debugging.
- `week3/faces/`: older face sample PNGs.
- `week3/face_embeddings/`: current InsightFace `.npy` embeddings.
- `week3/face_embedding_previews/`: preview images saved during embedding enrollment.
- `week3/memory/miguel_memory.json`: persistent Miguel memory.
- `week3/logs/`: hardware and integration test logs.
- `week3/models/vosk-model-small-en-us-0.15/`: local Vosk speech model.
- `week3/camera/venv/`: Python environment.
