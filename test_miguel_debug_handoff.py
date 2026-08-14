"""Regression tests for Miguel's post-shutdown debug handoff."""

from __future__ import annotations

import threading
import time
import unittest
from unittest import mock
import queue
import numpy as np
from week3.camera.dual_respeaker import DualMicStream, adaptive_rms_threshold, strongest_stereo_channel

from test_v7_5_portuguese_story import load_v7_5_module


class DebugHandoffTests(unittest.TestCase):
    def test_portuguese_acknowledgement_stays_local_and_grounded(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        spoken = []
        q.v6.speak = spoken.append

        self.assertTrue(q._route_fast_local_reply("entendi.", state))
        self.assertEqual(spoken, ["Entendi."])

    def test_portuguese_volume_command_changes_default_sink_relatively(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        spoken = []
        q.v6.speak = spoken.append

        with mock.patch.object(q.subprocess, "run") as run:
            run.return_value.returncode = 0
            run.return_value.stderr = ""
            run.return_value.stdout = ""
            self.assertTrue(q._route_system_volume_local_reply("aumente a taxa de som, Miguel", state))

        run.assert_called_once_with(
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+5%"],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        self.assertEqual(spoken, ["Aumentei o volume."])

    def test_dual_respeaker_selects_cleaner_non_clipping_ear(self) -> None:
        class FakeStdout:
            def __init__(self, chunk):
                self.chunk = chunk

            def read(self, _size):
                return self.chunk

        class FakeStream:
            def __init__(self, chunk):
                self.stdout = FakeStdout(chunk)

        quiet = np.full((160, 2), 500, dtype=np.int16).tobytes()
        clear = np.full((160, 2), 4000, dtype=np.int16).tobytes()
        stream = DualMicStream([FakeStream(quiet), FakeStream(clear)])

        self.assertEqual(stream.read(len(clear)), clear)
        self.assertEqual(stream.selected_chunks, [0, 1])

    def test_dual_respeaker_survives_one_disconnected_ear(self) -> None:
        class FakeStdout:
            def __init__(self, chunk):
                self.chunk = chunk

            def read(self, _size):
                return self.chunk

        class FakeStream:
            def __init__(self, chunk):
                self.stdout = FakeStdout(chunk)

        live = np.full((80, 2), 2500, dtype=np.int16).tobytes()
        stream = DualMicStream([FakeStream(b""), FakeStream(live)])

        self.assertEqual(stream.read(len(live)), live)
        self.assertEqual(stream.selected_chunks, [0, 1])

    def test_dual_respeaker_locks_one_array_for_the_utterance(self) -> None:
        class SequenceStdout:
            def __init__(self, chunks):
                self.chunks = iter(chunks)

            def read(self, _size):
                return next(self.chunks)

        class FakeStream:
            def __init__(self, chunks):
                self.stdout = SequenceStdout(chunks)

        chunk = lambda level: np.full((160, 2), level, dtype=np.int16).tobytes()
        left = FakeStream([chunk(1500), chunk(1200), chunk(1100)])
        right = FakeStream([chunk(700), chunk(5000), chunk(6000)])
        stream = DualMicStream([left, right], speech_lock_rms=900)

        self.assertEqual(stream.read(640), chunk(1500))
        self.assertEqual(stream.read(640), chunk(1200))
        self.assertEqual(stream.read(640), chunk(1100))
        self.assertEqual(stream.selected_chunks, [3, 0])

    def test_stereo_conversion_preserves_stronger_channel(self) -> None:
        stereo = np.column_stack(
            (np.full(160, 2000, dtype=np.int16), np.full(160, 300, dtype=np.int16))
        )

        mono_bytes, rms = strongest_stereo_channel(stereo.tobytes())

        self.assertTrue(np.array_equal(np.frombuffer(mono_bytes, dtype=np.int16), stereo[:, 0]))
        self.assertAlmostEqual(rms, 2000.0)

    def test_adaptive_speech_threshold_tracks_noise_but_stays_bounded(self) -> None:
        self.assertEqual(adaptive_rms_threshold([]), 500.0)
        self.assertEqual(adaptive_rms_threshold([200, 220, 240]), 660.0)
        self.assertEqual(adaptive_rms_threshold([900, 950]), 1200.0)

    def test_logged_friend_recognition_phrasings_enter_guarded_enrollment(self) -> None:
        q = load_v7_5_module()
        examples = (
            "hey miguel, i want you to recognize someone",
            "please refresh your camera and recognize my grandma as a friend, miguel",
            "can you meet me, make a friend, make friends",
        )
        for text in examples:
            with self.subTest(text=text):
                self.assertTrue(q._is_enrollment_request_text(text))

        named = "please fresh your camera and see her as a friend named ana"
        self.assertTrue(q._is_enrollment_request_text(named))
        self.assertEqual(q._extract_enrollment_name(named), "Ana")

    def test_latest_logged_new_user_requests_enter_guarded_enrollment(self) -> None:
        q = load_v7_5_module()
        named = "miguel, let's registrar a new user, elvira is marquinho's mother."
        camera_request = "hey miguel, i want you to register with your camera a new user."

        self.assertTrue(q._is_enrollment_request_text(named))
        self.assertEqual(q._extract_enrollment_name(named), "Elvira")
        self.assertTrue(q._is_enrollment_request_text(camera_request))
        self.assertIsNone(q._extract_enrollment_name(camera_request))

        class NoFaceCamera:
            @staticmethod
            def get_face_state(max_age_seconds=2.0):
                return {"face_detected": False, "recognized_person": None}

        state = q.RobotRuntimeState(stop_event=threading.Event())
        spoken = []
        q.v6.speak = spoken.append
        self.assertTrue(q._route_enrollment(named, state, NoFaceCamera()))
        self.assertEqual(state.enrollment_state, "requested")
        self.assertEqual(state.enrollment_target_name, "elvira")
        self.assertIsNone(state.enrollment_approved_by)
        self.assertIn("needs approval", spoken[-1].lower())

    def test_pending_enrollment_name_followup_stays_in_local_guarded_flow(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        state.enrollment_state = "awaiting_name"
        state.last_prompt_type = "enrollment_name"

        self.assertTrue(q._enrollment_followup_pending(state))
        self.assertEqual(
            q._extract_enrollment_name("Okay, her name is Ana, and you're gonna make friends with her."),
            "Ana",
        )

    def test_logged_owner_approval_variants_are_recognized_but_remain_face_gated(self) -> None:
        q = load_v7_5_module()
        variants = (
            "approve.",
            "hey miguel, enrollment is approved by me, marco.",
            "hey miguel, marco approves enrollment of your virus.",
        )
        for text in variants:
            with self.subTest(text=text):
                self.assertTrue(q._is_enrollment_approval_text(text))

        class NoOwnerCamera:
            @staticmethod
            def get_face_state(max_age_seconds=2.0):
                return {"face_detected": True, "recognized_person": None}

        state = q.RobotRuntimeState(stop_event=threading.Event())
        state.enrollment_state = "requested"
        state.enrollment_target_name = "elvira"
        spoken = []
        q.v6.speak = spoken.append

        self.assertTrue(q._route_enrollment("approve.", state, NoOwnerCamera()))
        self.assertEqual(state.enrollment_state, "requested")
        self.assertIsNone(state.enrollment_approved_by)
        self.assertIn("only marco or marquinho", spoken[-1].lower())

        state.recognized_person = "marco"
        self.assertTrue(
            q._route_enrollment(
                "hey miguel, marco approves enrollment of your virus.",
                state,
                NoOwnerCamera(),
            )
        )
        self.assertEqual(state.enrollment_state, "approved_pending_subject")
        self.assertEqual(state.enrollment_target_name, "elvira")
        self.assertEqual(state.enrollment_approved_by, "marco")

    def test_pending_enrollment_does_not_intercept_unrelated_camera_question(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        state.enrollment_state = "requested"
        state.enrollment_target_name = "elvira"
        state.last_prompt_type = "enrollment_approval"

        self.assertFalse(q._should_route_enrollment_followup("hey miguel, can you see me?", state))
        self.assertTrue(q._should_route_enrollment_followup("approve.", state))

    def test_logged_enrollment_cancellation_resets_before_camera_access(self) -> None:
        q = load_v7_5_module()

        class CameraMustNotBeRead:
            @staticmethod
            def get_face_state(*_args, **_kwargs):
                raise AssertionError("cancellation must not wait for camera state")

        state = q.RobotRuntimeState(stop_event=threading.Event())
        state.enrollment_state = "requested"
        state.enrollment_target_name = "olvira"
        state.last_prompt_type = "enrollment_approval"
        state.last_prompt_text = q._enrollment_approval_prompt("olvira")
        spoken = []
        q.v6.speak = spoken.append

        logged_text = "hey miguel, cancel the enrollment."
        self.assertTrue(q._is_enrollment_cancel_text(logged_text))
        self.assertTrue(q._should_route_enrollment_followup(logged_text, state))
        self.assertTrue(q._route_enrollment(logged_text, state, CameraMustNotBeRead()))
        self.assertEqual(state.enrollment_state, "idle")
        self.assertIsNone(state.enrollment_target_name)
        self.assertIsNone(state.last_prompt_type)
        self.assertEqual(spoken, ["Enrollment canceled."])

    def test_enrollment_dialog_gives_explicit_next_actions(self) -> None:
        q = load_v7_5_module()

        name_prompt = q._enrollment_name_prompt().lower()
        approval_prompt = q._enrollment_approval_prompt("elvira").lower()
        subject_prompt = q._enrollment_subject_prompt("elvira").lower()

        self.assertIn("step one", name_prompt)
        self.assertNotIn("elvira", name_prompt)
        self.assertIn("their name is, followed by the name", name_prompt)
        self.assertIn("correct enrollment name to", name_prompt)
        self.assertIn("step two", approval_prompt)
        self.assertIn("stand in front of my camera", approval_prompt)
        self.assertIn("say: marco approves enrolling elvira", approval_prompt)
        self.assertIn("step three", subject_prompt)
        self.assertIn("owner should move out of view", subject_prompt)
        self.assertIn("put only elvira", subject_prompt)
        self.assertIn("say: elvira is here", subject_prompt)

    def test_enrollment_name_correction_resets_approval_without_reading_camera(self) -> None:
        q = load_v7_5_module()

        class CameraMustNotBeRead:
            @staticmethod
            def get_face_state(*_args, **_kwargs):
                raise AssertionError("a name correction must not access the camera")

        state = q.RobotRuntimeState(stop_event=threading.Event())
        state.enrollment_state = "approved_pending_subject"
        state.enrollment_target_name = "misheard"
        state.enrollment_approved_by = "marco"
        state.enrollment_approved_at = time.time()
        spoken = []
        q.v6.speak = spoken.append

        command = "Miguel, correct the enrollment name to Daniela."
        self.assertEqual(q._extract_enrollment_name_correction(command), "Daniela")
        self.assertTrue(q._should_route_enrollment_followup(command, state))
        self.assertTrue(q._route_enrollment(command, state, CameraMustNotBeRead()))
        self.assertEqual(state.enrollment_target_name, "daniela")
        self.assertEqual(state.enrollment_state, "requested")
        self.assertIsNone(state.enrollment_approved_by)
        self.assertEqual(state.enrollment_approved_at, 0.0)
        self.assertIn("the enrollment name is daniela", spoken[-1].lower())
        self.assertIn("needs approval", spoken[-1].lower())

    def test_startup_announcement_is_not_logged_as_a_conversation_turn(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        state.conversation_log_session_id = "test"
        captured = []
        q._append_log_event = (
            lambda _state, event_type, **payload: captured.append((event_type, payload))
        )

        q._log_assistant_reply_event(
            state,
            "I am Miguel. Camera and face recognition are online.",
            "startup",
            latency_override={"reply_context": "startup", "log_user_text": ""},
        )

        self.assertEqual(captured[0][0], "startup_announcement")
        self.assertEqual(captured[0][1]["user_text"], "")
        self.assertEqual(state.last_completed_user_turn_at, 0.0)

    def test_startup_enqueue_context_is_snapshotted(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        replies = queue.Queue()
        original_speak = q.v6.speak
        try:
            q.install_speech_queue(replies, object(), state)
            q._speak_with_enqueue_context(
                "Miguel online.",
                {"reply_context": "startup", "log_user_text": ""},
            )
            event = replies.get_nowait()
        finally:
            q.v6.speak = original_speak

        self.assertEqual(event.context, "startup")
        self.assertEqual(event.latency["log_user_text"], "")

    def test_startup_announcement_logs_tts_latency_without_completing_a_user_turn(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        state.conversation_log_session_id = "test"
        captured = []
        q._append_log_event = (
            lambda _state, event_type, **payload: captured.append((event_type, payload))
        )

        q._log_assistant_reply_event(
            state,
            "Miguel online.",
            "startup",
            latency_override={
                "reply_context": "startup",
                "log_user_text": "",
                "turn_started_at": 10.0,
                "route_done_at": 10.0,
                "reply_queued_at": 10.01,
                "tts_prepare_ms": 125.4,
                "playback_start_ms": 135.2,
                "playback_ms": 2200.7,
                "total_ms": 2335.9,
            },
        )

        self.assertEqual(captured[0][0], "startup_announcement")
        self.assertEqual(
            captured[0][1]["latency_ms"],
            {
                "route": 0,
                "reply_queue": 10,
                "tts_prepare": 125,
                "playback_start": 135,
                "playback": 2201,
                "total": 2336,
            },
        )
        self.assertEqual(state.last_completed_user_turn_at, 0.0)

    def test_logged_portuguese_times_table_range_is_complete_locally(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        spoken = []
        q.v6.speak = spoken.append

        handled = q._route_portuguese_times_table_reply(
            "hey miguel, ensina a tabuada do 5, 5 vezes 1 a 5 vezes 12.", state
        )

        self.assertTrue(handled)
        self.assertIn("5 vezes 1 e 5", spoken[0])
        self.assertIn("5 vezes 12 e 60", spoken[0])
        self.assertEqual(spoken[0].count(" vezes "), 12)

    def test_logged_portuguese_continue_range_understands_por(self) -> None:
        q = load_v7_5_module()
        self.assertEqual(
            q._parse_portuguese_times_table_request(
                "continue a tabuada no cinco por seis ate o cinco por doze"
            ),
            (5, 6, 12),
        )

    def test_affirmative_can_accept_pending_times_table_offer(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        state.pending_times_table = (5, 6, 12)
        spoken = []
        q.v6.speak = spoken.append

        self.assertTrue(q._route_portuguese_times_table_reply("sim, Miguel!", state))
        self.assertIn("5 vezes 6 e 30", spoken[0])
        self.assertIn("5 vezes 12 e 60", spoken[0])
        self.assertIsNone(state.pending_times_table)

    def test_logged_affirmative_accepts_range_from_previous_robot_offer(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        state.last_robot_text = "Quer que eu continue a tabuada do 5 do 5x6 ate o 5x12?"
        spoken = []
        q.v6.speak = spoken.append

        self.assertTrue(q._route_portuguese_times_table_reply("sim, Miguel!", state))
        self.assertEqual(spoken[0].count(" vezes "), 7)
        self.assertIn("5 vezes 12 e 60", spoken[0])

    def test_startup_resets_persisted_personality_to_normal_conversation(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        state.current_mode = "mission_control"
        state.conversation_mode = "robot_control"
        state.response_length_mode = "terse"
        state.response_depth_mode = "long_explanation"

        with mock.patch.object(q.robot_memory, "set_personality_mode", create=True) as set_mode:
            q._initialize_startup_conversation_mode(state)

        set_mode.assert_called_once_with("normal")
        self.assertEqual(state.current_mode, "normal")
        self.assertEqual(state.conversation_mode, "wake_required")
        self.assertEqual(state.response_length_mode, "normal")
        self.assertEqual(state.response_depth_mode, "normal")

    def test_drop_mission_control_activates_normal_conversation(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        state.current_mode = "mission_control"
        state.conversation_mode = "robot_control"
        state.response_length_mode = "terse"
        spoken = []
        q.v6.speak = spoken.append

        text = "Hey Miguel, drop the mission control and let's have a normal mode conversation."

        self.assertTrue(q._is_normal_conversation_mode_request(text))
        self.assertTrue(q._route_response_depth_mode(text, state))
        self.assertEqual(state.current_mode, "normal")
        self.assertEqual(state.conversation_mode, "general")
        self.assertEqual(state.response_length_mode, "normal")
        self.assertEqual(state.response_depth_mode, "normal")
        self.assertEqual(
            spoken,
            ["Normal conversation mode on. I'll use natural, complete replies."],
        )

    def test_activate_normal_mode_conversation_is_not_a_short_mode_ack(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        spoken = []
        q.v6.speak = spoken.append

        self.assertTrue(
            q._route_response_depth_mode("No, activate your normal mode conversation.", state)
        )
        self.assertNotIn("short", spoken[0].lower())

    def test_capability_inventory_does_not_activate_teacher_mode(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        spoken = []
        q.v6.speak = spoken.append
        text = (
            "I'm pretty sure you have modes like story modes, creative modes, "
            "teacher mode, and other functions. Describe all the modes and functions."
        )

        self.assertTrue(q._is_capabilities_request(text))
        self.assertFalse(q._route_teacher_mode_local_reply(text, state, partner="marco"))
        self.assertTrue(q._route_capabilities_local_reply(text, state))
        self.assertFalse(state.teacher_controller.session.active)
        self.assertIn("commands I can actually handle", spoken[0])

    def test_logged_teacher_exit_phrasings_leave_teacher_mode(self) -> None:
        q = load_v7_5_module()

        for text in (
            "Hey Miguel, just teacher mode off now.",
            "Miguel, I want you to turn off the teacher mode.",
            "Miguel, go to normal mode.",
        ):
            with self.subTest(text=text):
                state = q.RobotRuntimeState(stop_event=threading.Event())
                state.teacher_controller.session.active = True
                state.conversation_mode = "teacher"
                spoken = []
                q.v6.speak = spoken.append

                self.assertTrue(q._route_teacher_mode_local_reply(text, state, partner="marco"))
                self.assertFalse(state.teacher_controller.session.active)
                self.assertEqual(state.conversation_mode, "general")
                self.assertIn("Teacher mode is off", spoken[0])

    def test_voice_commands_select_all_graphical_face_expressions(self) -> None:
        q = load_v7_5_module()
        for expression in ("normal", "happy", "angry", "sad", "scared", "concerned", "motivated"):
            with self.subTest(expression=expression):
                state = q.RobotRuntimeState(stop_event=threading.Event())
                spoken = []
                q.v6.speak = spoken.append

                self.assertTrue(
                    q._route_face_expression_local_reply(
                        f"Miguel, change to {expression} face", state
                    )
                )
                self.assertEqual(state.face_expression, expression)
                self.assertEqual(state.face_expression_source, "voice")
                self.assertEqual(spoken, [f"{expression.title()} face selected."])

    def test_face_expression_and_joke_compound_request_fulfills_both(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        spoken = []
        q.v6.speak = spoken.append

        handled = q._route_face_expression_local_reply(
            "miguel, go to happy face and tell a funny joke.", state
        )

        self.assertTrue(handled)
        self.assertEqual(state.face_expression, "happy")
        self.assertEqual(len(spoken), 1)
        self.assertTrue(spoken[0].startswith("Happy face selected. "))
        self.assertIn(spoken[0].split(". ", 1)[1], q.LOCAL_JOKES)

    def test_bare_wake_greeting_is_logged_before_reply(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        captured = []
        q._remember_accepted_turn = lambda _state, text: captured.append(("remember", text))
        q._log_user_turn_event = lambda _state, text, route_hint="", partner=None: captured.append(
            ("log", text, route_hint, partner)
        )
        q.v6.speak = lambda text: captured.append(("speak", text))
        event = q.UserTurnEvent(
            text="hello, miguel.",
            recognized_person="marco",
            authorized=True,
            authorization_source="wake_phrase",
            stripped_text="",
        )

        self.assertTrue(q.handle_queued_turn(event, None, mock.Mock(), state))
        self.assertEqual(
            captured,
            [
                ("remember", "hello, miguel."),
                ("log", "hello, miguel.", "greeting", "marco"),
                ("speak", "Here."),
            ],
        )
    def test_face_expression_command_requires_command_and_face_words(self) -> None:
        q = load_v7_5_module()
        self.assertIsNone(q._requested_face_expression("I read a sad story"))
        self.assertIsNone(q._requested_face_expression("change the story to a happy ending"))
        self.assertEqual(q._requested_face_expression("show a neutral expression"), "normal")
        self.assertEqual(q._requested_face_expression("miguel, go for happyface"), "happy")

    def test_face_emotion_questions_and_comments_are_not_commands(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        state.face_expression = "normal"
        state.face_expression_source = "voice"
        spoken = []
        q.v6.speak = spoken.append

        for text in (
            "okay, now, miguel, that is a good angry face.",
            "hey miguel, why are you angry?",
            "hey miguel, i'm asking the reason for you to be angry.",
            "miguel, are you angry?",
            "hey miguel, are you happy?",
        ):
            with self.subTest(text=text):
                self.assertFalse(q._route_face_expression_local_reply(text, state))

        self.assertEqual(state.face_expression, "normal")
        self.assertEqual(spoken, [])
        self.assertTrue(q._route_face_expression_local_reply("angry face", state))
        self.assertEqual(state.face_expression, "angry")

    def test_angry_face_requests_select_the_angry_expression_locally(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        spoken = []
        q.v6.speak = spoken.append

        for text in ("go to angry face", "go to super angry face, please"):
            with self.subTest(text=text):
                spoken.clear()
                self.assertTrue(q._route_face_expression_local_reply(text, state))
                self.assertEqual(spoken, ["Angry face selected."])
                self.assertEqual(state.face_expression, "angry")
                self.assertEqual(state.current_turn_latency["reply_context"], "face_expression")

    def test_face_inventory_is_answered_from_supported_expressions(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        spoken = []
        q.v6.speak = spoken.append

        self.assertTrue(q._route_face_expression_local_reply("describe how many faces do you have", state))
        self.assertIn("seven selectable face expressions", spoken[0])
        self.assertIn("angry", spoken[0])
        self.assertIn("motivated", spoken[0])

    def test_face_expression_followups_repeat_locally_without_cloud_promises(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        state.face_expression = "scared"
        state.face_expression_source = "voice"
        spoken = []
        q.v6.speak = spoken.append

        self.assertTrue(q._route_face_expression_local_reply("can you please do it again?", state))
        self.assertEqual(spoken.pop(), "Scared face selected.")
        self.assertTrue(q._route_face_expression_local_reply("scary, scary!", state))
        self.assertEqual(spoken.pop(), "Scared face selected.")

        self.assertTrue(q._route_face_expression_local_reply("big!", state))
        self.assertIn("cannot resize", spoken.pop())

    def test_affirmative_face_prompt_executes_current_expression(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        state.face_expression = "scared"
        state.face_expression_source = "voice"
        state.last_robot_question_text = "Do you mean switching to the scared face expression?"
        state.last_robot_question_at = time.time()
        spoken = []
        q.v6.speak = spoken.append

        self.assertTrue(q._route_face_expression_local_reply("yes?", state))
        self.assertEqual(spoken, ["Scared face selected."])

    def test_detailed_capability_inventory_uses_detailed_speech_budget(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        spoken = []
        q.v6.speak = spoken.append

        self.assertTrue(q._route_capabilities_local_reply(
            "describe all capabilities and functions you have", state
        ))
        self.assertEqual(state.current_turn_latency["response_length_mode"], "detailed")
        self.assertIn("shutdown with explicit confirmation", spoken[0])

    def test_normal_capability_inventory_trims_at_a_complete_sentence(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        spoken = []
        q.v6.speak = spoken.append

        self.assertTrue(q._route_capabilities_local_reply("what are your capabilities", state))
        shaped = q.make_robot_reply_concise(
            spoken[0], context="capabilities", response_length_mode="normal"
        )

        self.assertNotIn("...", shaped)
        self.assertTrue(shaped.endswith("."))
        self.assertLessEqual(q._word_len(shaped), q._response_word_limit("normal"))

    def test_sleep_and_wake_routes_log_the_user_side_before_reply(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        logged = []
        q.v6.speak = lambda _text: None
        q._log_user_turn_event = lambda _state, text, route_hint="", partner=None: logged.append(
            (text, route_hint, partner)
        )

        sleep_event = q.UserTurnEvent(
            "hey miguel, go to silent mode.", None, True, "owner_session",
            "hey miguel go to silent mode", "go to silent mode",
        )
        self.assertTrue(q.handle_queued_turn(sleep_event, None, object(), state))
        self.assertEqual(logged[-1][1], "sleep")

        wake_event = q.UserTurnEvent(
            "mission control?", None, True, "owner_session", "mission control", "",
        )
        self.assertTrue(q.handle_queued_turn(wake_event, None, object(), state))
        self.assertEqual(logged[-1][1], "wake")

    def test_automatic_face_rules_use_only_strong_cues(self) -> None:
        q = load_v7_5_module()
        examples = {
            "I am happy about the good news": "happy",
            "I am sad because my friend passed away": "sad",
            "I'm scared; this is an emergency": "scared",
            "I'm worried that the microphone is not working": "concerned",
            "Let's practice math; my goal is level five": "motivated",
        }
        for text, expected in examples.items():
            with self.subTest(text=text):
                self.assertEqual(q._automatic_face_expression(text), expected)
        self.assertIsNone(q._automatic_face_expression("Explain a sad character in a book"))

    def test_resting_expression_does_not_override_shutdown_confirmation(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        state.face_expression = "happy"
        state.shutdown_confirmation_pending = True
        face_state, text = q._face_status_payload(state, "idle", "")
        self.assertIn(face_state, {"confirm", "shutdown_pending"})
        self.assertEqual(text, "Confirm shutdown")

    def test_selected_expression_survives_listening_and_thinking_states(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        state.conversation_active = True
        state.face_expression = "sad"
        state.face_expression_source = "voice"

        state.audio_capture_active = True
        face_state, text = q._face_status_payload(state, "listening", "YOUR TURN")
        self.assertEqual((face_state, text), ("sad", "YOUR TURN"))

        state.audio_capture_active = False
        face_state, text = q._face_status_payload(state, "thinking", "Processing")
        self.assertEqual((face_state, text), ("sad", "Processing"))

    def test_normal_expression_keeps_listening_and_thinking_animations(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        state.conversation_active = True

        state.audio_capture_active = True
        face_state, text = q._face_status_payload(state, "listening", "YOUR TURN")
        self.assertEqual((face_state, text), ("listening", "YOUR TURN"))

        state.audio_capture_active = False
        face_state, text = q._face_status_payload(state, "thinking", "Processing")
        self.assertEqual((face_state, text), ("thinking", "Processing"))

    def test_thinking_priority_allows_selected_expression_to_be_emitted(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        state.conversation_active = True
        state.turn_processing_active = True
        state.interaction_state = "thinking"
        state.current_status_text = "Processing"
        state.face_expression = "motivated"
        emitted = []
        q.full.face_status = lambda face_state, text: emitted.append((face_state, text))

        q.notify_face_status(state, "thinking", "Processing")

        self.assertEqual(emitted, [("motivated", "Processing")])

    def test_natural_second_person_shutdown_request_requires_confirmation(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        spoken = []
        q.v6.speak = spoken.append

        result = q._route_shutdown_control("so, miguel, you can just turn off now.", state)

        self.assertIs(result, True)
        self.assertTrue(state.shutdown_pending)
        self.assertTrue(state.shutdown_confirmation_pending)
        self.assertEqual(spoken, ["Shutdown confirmation required."])
        self.assertFalse(state.stop_event.is_set())

    def test_turning_off_other_hardware_is_not_robot_shutdown(self) -> None:
        q = load_v7_5_module()

        self.assertFalse(q.is_explicit_robot_shutdown("turn off the camera now"))
        self.assertFalse(q.is_explicit_robot_shutdown("turn off the lights"))

    def test_length_complaints_request_detailed_answers(self) -> None:
        q = load_v7_5_module()

        examples = (
            "Can you be a little bit longer about the Spider-Man movies?",
            "Why are you cutting your answers? They are too short.",
            "Change your mode to be a little bit more on the answer.",
        )

        for text in examples:
            with self.subTest(text=text):
                self.assertEqual(q.infer_response_length_mode(text, "creative", "none"), "detailed")

    def test_detailed_reply_is_not_cut_at_normal_limit(self) -> None:
        q = load_v7_5_module()
        reply = (
            "Spider-Man begins as Peter Parker, a teenager balancing school and responsibility. "
            "His choices connect him to the Avengers, especially when larger threats pull his "
            "neighborhood heroism into the wider Marvel story. The Avengers become mentors and "
            "teammates, while Peter still has to decide what kind of hero he wants to be."
        )

        shaped = q.make_robot_reply_concise(reply, context="creative", response_length_mode="detailed")

        self.assertEqual(shaped, reply)
        self.assertGreater(q._word_len(shaped), q._response_word_limit("normal"))

    def test_compound_clarification_reaches_normal_conversation(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        state.last_answer_text_short = "Miguel's core systems are online."
        state.last_answer_at = time.time()

        handled = q._route_last_answer_clarification(
            "What do you mean? I was talking about the sword and its steel.", state
        )

        self.assertFalse(handled)

    def test_short_clarification_still_repeats_previous_answer(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        state.last_answer_text_short = "Miguel's core systems are online."
        state.last_answer_at = time.time()
        spoken = []
        q.v6.speak = spoken.append

        self.assertTrue(q._route_last_answer_clarification("What do you mean?", state))
        self.assertEqual(spoken, ["I meant this: Miguel's core systems are online."])

    def test_save_conversation_is_truthful_local_memory_reply(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        spoken = []
        q.v6.speak = spoken.append

        self.assertTrue(q._route_conversation_memory_local_reply("save the sword conversation", state))
        self.assertIn("already being saved", spoken[0])

    def test_user_turn_event_preserves_its_logging_snapshot(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        turns = queue.Queue()

        q._enqueue_user_turn(turns, state, "status", recognized_person="marquinho", authorized=True)
        event = turns.get_nowait()
        q._enqueue_user_turn(turns, state, "a later transcript", recognized_person="marquinho", authorized=True)

        self.assertEqual(event.latency["log_user_text"], "status")
        self.assertEqual(event.latency["log_person"], "marquinho")

    def test_story_session_freezes_initiating_log_attribution(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        state.conversation_partner = "marquinho"
        state.conversation_mode = "story"
        state.session_focus = "story: space adventure"
        session = q.StorySession(
            active=True,
            mode="story_continuous",
            title="Space Adventure",
            chapter_count=2,
            auto_continue=True,
        )
        state.story_session = session
        queued = []

        def generate(_session, _user_text, chapter_number):
            if chapter_number == 1:
                state.conversation_partner = None
                state.conversation_mode = "wake_required"
            return f"Chapter {chapter_number}."

        with (
            mock.patch.object(q, "_generate_story_chapter", side_effect=generate),
            mock.patch.object(q, "_wait_for_story_resume", return_value=True),
            mock.patch.object(q, "_wait_for_story_speech_slot", return_value=True),
            mock.patch.object(q, "_speak_with_enqueue_context", side_effect=lambda text, latency: queued.append((text, latency))),
        ):
            q._run_story_session("tell me a story", state, session)

        self.assertEqual([item[1]["log_person"] for item in queued], ["marquinho", "marquinho"])
        self.assertEqual([item[1]["log_conversation_mode"] for item in queued], ["story", "story"])
        self.assertEqual([item[1]["log_topic"] for item in queued], ["story: space adventure"] * 2)

    def test_story_topic_drops_spoken_duration_grammar(self) -> None:
        q = load_v7_5_module()

        topic = q._extract_long_story_topic_hint(
            "hey miguel, tell a story of twenty minutes duration about adventures and space"
        )

        self.assertEqual(topic, "adventures and space")

    def test_auto_story_latency_separates_generation_from_speech_slot_wait(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        state.session_focus = "story: space adventure"
        session = q.StorySession(
            active=True,
            mode="story_continuous",
            title="Space Adventure",
            chapter_count=1,
            auto_continue=True,
        )
        state.story_session = session
        queued = []

        with (
            mock.patch.object(q, "_generate_story_chapter", return_value="Chapter 1."),
            mock.patch.object(q, "_wait_for_story_resume", return_value=True),
            mock.patch.object(q, "_wait_for_story_speech_slot", return_value=True),
            mock.patch.object(q, "_speak_with_enqueue_context", side_effect=lambda text, latency: queued.append(latency)),
        ):
            q._run_story_session("tell me a story", state, session)

        self.assertGreaterEqual(queued[0]["story_generation_ms"], 0)
        self.assertGreaterEqual(queued[0]["story_speech_slot_wait_ms"], 0)
        self.assertGreaterEqual(queued[0]["turn_started_at"], queued[0]["route_done_at"])

    def test_routing_freezes_resolved_session_partner_for_reply_log(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        state.conversation_partner = "marco"
        state.conversation_active = True
        state.conversation_until = time.time() + 60.0
        event = q.UserTurnEvent(
            "hey miguel",
            None,
            True,
            "owner_session",
            "hey miguel",
            "",
            {"turn_started_at": time.monotonic(), "log_person": None},
        )
        q.v6.speak = lambda _text: None

        self.assertTrue(q.handle_queued_turn(event, None, None, state))
        self.assertEqual(state.current_turn_latency["log_person"], "marco")

    def test_user_turn_carries_capture_timing_once(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        turns = queue.Queue()
        state.last_audio_capture_timing = {
            "capture_total_ms": 2100.0,
            "endpoint_silence_ms": 905.0,
            "transcription_ms": 430.0,
        }

        q._enqueue_user_turn(turns, state, "status", recognized_person="marquinho", authorized=True)
        first = turns.get_nowait()
        q._enqueue_user_turn(turns, state, "time", recognized_person="marquinho", authorized=True)
        second = turns.get_nowait()

        self.assertEqual(first.latency["capture_total_ms"], 2100.0)
        self.assertEqual(first.latency["transcription_ms"], 430.0)
        self.assertNotIn("capture_total_ms", second.latency)

    def test_fixed_reply_requests_persistent_tts_cache(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        replies = queue.Queue()
        state.reply_queue = replies
        state.pending_reply_count = 1
        replies.put(q.ReplyEvent("Here.", {"reply_context": "greeting"}, "greeting"))
        prepared_calls = []
        played = []

        def prepare(text, cache=False):
            prepared_calls.append((text, cache))
            return {"text": text, "wav_path": "/tmp/fake.wav"}

        q.v6.prepare_speech_audio = prepare
        def play(prepared):
            played.append(prepared)
            state.stop_event.set()

        q.v6.play_prepared_speech = play
        safety = mock.Mock()
        safety.evaluate_assistant_reply.return_value = mock.Mock(allowed=True)

        q.speech_worker(mock.Mock(), safety, replies, state)

        self.assertEqual(prepared_calls, [("Here.", True)])
        self.assertEqual(played, [{"text": "Here.", "wav_path": "/tmp/fake.wav"}])

    def test_confirmed_voice_shutdown_requests_debug_handoff(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        state.shutdown_pending = True
        state.shutdown_confirmation_pending = True
        state.shutdown_confirmation_until = time.time() + 30.0

        result = q._route_shutdown_control("confirm shutdown", state)

        self.assertIs(result, False)
        self.assertTrue(state.stop_event.is_set())
        self.assertTrue(state.debug_handoff_requested)

    def test_shutdown_request_without_confirmation_does_not_request_handoff(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())

        result = q._route_shutdown_control("shutdown", state)

        self.assertIs(result, True)
        self.assertFalse(state.stop_event.is_set())
        self.assertFalse(state.debug_handoff_requested)

    def test_debug_handoff_launcher_is_detached(self) -> None:
        q = load_v7_5_module()

        with mock.patch.object(q.subprocess, "Popen") as popen:
            self.assertTrue(q._launch_debug_handoff())

        args, kwargs = popen.call_args
        self.assertEqual(args[0][-1], "--voice")
        self.assertTrue(args[0][0].endswith("week3/tools/miguel_debug_last.sh"))
        self.assertTrue(kwargs["start_new_session"])
        self.assertIs(kwargs["stdin"], q.subprocess.DEVNULL)

    def test_speech_worker_finishes_shutdown_confirmation_after_stop(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        replies = queue.Queue()
        state.reply_queue = replies
        state.pending_reply_count = 2
        state.stop_event.set()
        replies.put(q.ReplyEvent("obsolete reply", {"reply_context": "normal"}, "normal"))
        replies.put(q.ReplyEvent("Confirmed.", {"reply_context": "shutdown_confirm"}, "shutdown_confirm"))
        spoken = []
        safety = mock.Mock()
        safety.evaluate_assistant_reply.return_value = mock.Mock(allowed=True)

        q.speech_worker(spoken.append, safety, replies, state)

        self.assertEqual(spoken, ["Confirmed."])
        self.assertTrue(state.shutdown_acknowledged_event.is_set())
        self.assertEqual(state.pending_reply_count, 0)

    def test_shutdown_confirmation_survives_late_barge_in_flag(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        replies = queue.Queue()
        state.reply_queue = replies
        state.pending_reply_count = 1
        state.stop_event.set()
        replies.put(q.ReplyEvent("Confirmed.", {"reply_context": "shutdown_confirm"}, "shutdown_confirm"))
        spoken = []
        safety = mock.Mock()

        def evaluate(_text):
            state.stop_speech_event.set()
            return mock.Mock(allowed=True)

        safety.evaluate_assistant_reply.side_effect = evaluate

        q.speech_worker(spoken.append, safety, replies, state)

        self.assertEqual(spoken, ["Confirmed."])
        self.assertFalse(state.stop_speech_event.is_set())
        self.assertTrue(state.shutdown_acknowledged_event.is_set())
        self.assertEqual(state.pending_reply_count, 0)

    def test_shutdown_confirmation_is_enqueued_despite_barge_in_flag(self) -> None:
        q = load_v7_5_module()
        state = q.RobotRuntimeState(stop_event=threading.Event())
        replies = queue.Queue()
        state.current_turn_latency["reply_context"] = "shutdown_confirm"
        state.stop_speech_event.set()

        q.install_speech_queue(replies, mock.Mock(), state)
        q.v6.speak("Confirmed.")

        event = replies.get_nowait()
        self.assertEqual(event.text, "Confirmed.")
        self.assertEqual(event.context, "shutdown_confirm")
        self.assertFalse(state.stop_speech_event.is_set())
        self.assertEqual(state.pending_reply_count, 1)

    def test_shutdown_signal_handler_sets_stop_event_and_reason(self) -> None:
        q = load_v7_5_module()
        stop_event = threading.Event()
        reasons = []

        previous = q._install_shutdown_signal_handlers(stop_event, reasons.append)
        try:
            handler = q.signal.getsignal(q.signal.SIGTERM)
            self.assertTrue(callable(handler))
            handler(q.signal.SIGTERM, None)
        finally:
            q._restore_signal_handlers(previous)

        self.assertTrue(stop_event.is_set())
        self.assertEqual(reasons, ["sigterm"])


if __name__ == "__main__":
    unittest.main()
