import random
import tempfile
import unittest

from miguel_teacher import (
    CloudTeacherContent,
    MULTIPLICATION_LEVELS,
    MathTeacher,
    TeacherModeController,
    TeacherProgressStore,
    TeacherSession,
    extract_age,
    evaluate_mastery,
    extract_level_selection,
    extract_subject,
    extract_subtopic,
    get_level_progress,
    local_progress_recommendation,
)


class MockCloudTeacher:
    def __init__(self, creative=None, intro=None, explanation=None, raise_on_creative=False):
        self.creative = creative
        self.intro = intro
        self.explanation = explanation
        self.raise_on_creative = raise_on_creative

    def generate_lesson_intro(self, session):
        return self.intro

    def generate_creative_question(self, session):
        if self.raise_on_creative:
            raise RuntimeError("cloud unavailable")
        return self.creative

    def generate_hint(self, session):
        return CloudTeacherContent(
            speech="Hint ready.",
            hint="Think of equal robot cargo groups.",
            next_action="continue_lesson",
        )

    def generate_explanation(self, session, result):
        return self.explanation

    def generate_adaptive_review(self, session):
        return None

    def generate_recommendation(self, session, progress):
        return None


class MiguelTeacherModeTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def make_controller(self) -> TeacherModeController:
        return TeacherModeController(
            teachers={"math": MathTeacher(rng=random.Random(7))},
            progress_store=TeacherProgressStore(self.tempdir.name),
        )

    def start_level_one(self, controller: TeacherModeController) -> str:
        controller.handle("I am 9 years old teach me multiplication")
        return controller.handle("level 1")

    def test_detects_teacher_intent(self):
        controller = self.make_controller()
        self.assertTrue(controller.detects_teacher_intent("teacher mode"))
        self.assertTrue(controller.detects_teacher_intent("I want to learn math"))
        self.assertTrue(controller.detects_teacher_intent("I'm 9 teach me math multiplication"))

    def test_extracts_age_numeric_and_word_forms(self):
        self.assertEqual(extract_age("I am 9 years old"), 9)
        self.assertEqual(extract_age("I'm 9 teach me math"), 9)
        self.assertEqual(extract_age("I am nine"), 9)
        self.assertEqual(extract_age("Marquinho is 9"), 9)
        self.assertEqual(extract_age("he is nine years old"), 9)
        self.assertEqual(extract_age("teach me multiplication for a 9 years old boy level"), 9)

    def test_extracts_math_and_multiplication(self):
        self.assertEqual(extract_subject("teach me math multiplication"), "math")
        self.assertEqual(extract_subtopic("teach me multiplication"), "multiplication")
        self.assertEqual(extract_subtopic("practice times tables"), "multiplication")
        self.assertEqual(extract_subtopic("teach math table"), "multiplication")
        self.assertEqual(extract_subtopic("multiplikation"), "multiplication")
        self.assertEqual(extract_subtopic("multiplicação"), "multiplication")

    def test_multiplication_level_definitions_exist(self):
        self.assertEqual(len(MULTIPLICATION_LEVELS), 5)
        self.assertEqual(MULTIPLICATION_LEVELS[1].level_name, "Beginner Tables")
        self.assertTrue(MULTIPLICATION_LEVELS[5].cloud_creative_supported)

    def test_level_selection_aliases(self):
        self.assertEqual(extract_level_selection("level 1"), 1)
        self.assertEqual(extract_level_selection("start level 2"), 2)
        self.assertEqual(extract_level_selection("beginner"), 1)
        self.assertEqual(extract_level_selection("word problems"), 3)
        self.assertEqual(extract_level_selection("challenge mode"), 5)

    def test_start_flow_asks_for_level_when_topic_known(self):
        controller = self.make_controller()
        reply = controller.handle("I am 9 years old teach me multiplication")
        self.assertIn("choose Level 1", reply)
        self.assertIsNone(controller.session.current_question)

    def test_logged_math_table_request_reaches_multiplication_level_prompt(self):
        controller = self.make_controller()
        reply = controller.handle(
            "miguel, let's go to teacher mode and i wanna you to teach math table. "
            "hey miguel, can you go to teacher mode and teach me math?",
            user_id="marco",
            display_name="Marco",
        )

        self.assertIn("choose Level 1", reply)
        self.assertEqual(controller.session.subject, "math")
        self.assertEqual(controller.session.subtopic, "multiplication")
        self.assertEqual(controller.session.lesson_step, "ask_level")

    def test_active_teacher_accepts_asr_multiplikation_topic(self):
        controller = self.make_controller()
        controller.handle("teacher mode", user_id="marco", display_name="Marco")

        reply = controller.handle("multiplikation", user_id="marco", display_name="Marco")

        self.assertIn("choose Level 1", reply)
        self.assertEqual(controller.session.subject, "math")
        self.assertEqual(controller.session.subtopic, "multiplication")
        self.assertEqual(controller.session.lesson_step, "ask_level")

    def test_level_selection_commands_start_expected_levels(self):
        controller = self.make_controller()
        controller.handle("I am 9 years old teach me multiplication")
        self.assertIn("Today we will learn multiplication", controller.handle("level 1"))
        self.assertEqual(controller.session.level_number, 1)
        controller.handle("teach me multiplication")
        self.assertIn("Today we will learn multiplication", controller.handle("word problems"))
        self.assertEqual(controller.session.level_number, 3)
        controller.handle("teach me multiplication")
        self.assertIn("Today we will learn multiplication", controller.handle("challenge mode"))
        self.assertEqual(controller.session.level_number, 5)

    def test_starts_multiplication_lesson(self):
        controller = self.make_controller()
        controller.handle("I am 9 years old teach me multiplication")
        reply = controller.handle("level 1")
        self.assertIsNotNone(reply)
        self.assertIn("Today we will learn multiplication", reply)
        self.assertIn("what is", reply)
        self.assertTrue(controller.session.active)
        self.assertEqual(controller.session.subject, "math")
        self.assertEqual(controller.session.subtopic, "multiplication")
        self.assertEqual(controller.session.level_number, 1)
        self.assertIsNotNone(controller.session.current_question)

    def test_followup_flow_from_teacher_mode(self):
        controller = self.make_controller()
        self.assertIn("How old", controller.handle("teacher mode"))
        self.assertIn("What subject", controller.handle("9"))
        self.assertIn("practice", controller.handle("math"))
        reply = controller.handle("multiplication")
        self.assertIn("Level 1", reply)
        reply = controller.handle("level 1")
        self.assertIn("Today we will learn multiplication", reply)
        self.assertIsNotNone(controller.session.current_question)

    def test_log_phrase_starts_teacher_lesson(self):
        controller = self.make_controller()
        reply = controller.handle(
            "nice, miguel, you got it right. miguel, let's test your teacher mode. "
            "i wanna you teach multiplication for marquinho. he's a 9-years-old boy."
        )
        self.assertIn("Level 1", reply)
        reply = controller.handle("level 1")
        self.assertIn("Today we will learn multiplication", reply)
        self.assertEqual(controller.session.student_age, 9)
        self.assertEqual(controller.session.subtopic, "multiplication")
        self.assertIsNotNone(controller.session.current_question)

    def test_repeated_topic_request_during_lesson_restarts_lesson(self):
        controller = self.make_controller()
        self.start_level_one(controller)
        reply = controller.handle("Miguel, teach me multiplication.")
        self.assertIn("Today we will learn multiplication", reply)
        self.assertEqual(controller.session.score_total, 0)

    def test_log_followup_age_phrase_starts_lesson(self):
        controller = self.make_controller()
        controller.handle("miguel, i wanna you to go to teacher mode and make lesson for multiplication, math.")
        reply = controller.handle("okay, can you teach me multiplication for a 9 years old boy level?")
        self.assertIn("Level 1", reply)
        reply = controller.handle("level 1")
        self.assertIn("Today we will learn multiplication", reply)
        self.assertEqual(controller.session.student_age, 9)
        self.assertIsNotNone(controller.session.current_question)

    def test_basic_lesson_start_ignores_cloud_intro_language(self):
        cloud = MockCloudTeacher(
            intro=CloudTeacherContent(
                speech="Oi, Marquinho! Missão Miguel iniciando.",
                question="Quanto é 3 vezes 8?",
                expected_answer=24,
                answer_type="number",
                difficulty="normal",
                next_action="wait_for_answer",
            )
        )
        controller = TeacherModeController(
            teachers={"math": MathTeacher(rng=random.Random(7))},
            cloud_client=cloud,
            progress_store=TeacherProgressStore(self.tempdir.name),
        )
        controller.handle("I am 9 years old teach me multiplication")
        reply = controller.handle("level 1")
        self.assertIn("Today we will learn multiplication", reply)
        self.assertNotIn("Oi", reply)
        self.assertNotIn("Missão", reply)
        self.assertNotEqual(controller.session.current_question.get("source"), "cloud")

    def test_answer_feedback_stays_on_original_question(self):
        cloud = MockCloudTeacher(
            explanation=CloudTeacherContent(
                speech="Cloud explanation.",
                explanation="12 times 9 equals 108.",
                difficulty="normal",
                next_action="continue_lesson",
            )
        )
        controller = TeacherModeController(
            teachers={"math": MathTeacher(rng=random.Random(7))},
            cloud_client=cloud,
            progress_store=TeacherProgressStore(self.tempdir.name),
        )
        self.start_level_one(controller)
        controller.session.current_question = {
            "type": "multiplication",
            "a": 3,
            "b": 8,
            "expected_answer": 24,
        }
        reply = controller.handle("twenty")
        self.assertIn("3 times 8", reply)
        self.assertNotIn("12 times 9", reply)
        self.assertEqual(controller.session.score_total, 1)

    def test_generates_multiplication_question_with_expected_answer(self):
        teacher = MathTeacher(rng=random.Random(3))
        session = TeacherSession(active=True, student_age=9, subject="math", subtopic="multiplication", level_number=1)
        question = teacher.generate_question(session)
        self.assertEqual(question["expected_answer"], question["a"] * question["b"])
        self.assertGreaterEqual(question["a"], 2)
        self.assertLessEqual(question["a"], 12)

    def test_checks_correct_answer_and_tracks_score(self):
        controller = self.make_controller()
        self.start_level_one(controller)
        expected = controller.session.current_question["expected_answer"]
        reply = controller.handle(str(expected))
        self.assertIn("Great job", reply)
        self.assertEqual(controller.session.score_correct, 1)
        self.assertEqual(controller.session.score_total, 1)

    def test_checks_spoken_correct_answer(self):
        controller = self.make_controller()
        self.start_level_one(controller)
        controller.session.current_question = {
            "type": "multiplication",
            "a": 3,
            "b": 8,
            "expected_answer": 24,
        }
        reply = controller.handle("twenty four")
        self.assertIn("Great job", reply)
        self.assertEqual(controller.session.score_correct, 1)

    def test_checks_incorrect_answer_and_tracks_score(self):
        controller = self.make_controller()
        self.start_level_one(controller)
        expected = controller.session.current_question["expected_answer"]
        reply = controller.handle(str(expected + 1))
        self.assertIn("Good try", reply)
        self.assertIn("means", reply)
        self.assertEqual(controller.session.score_correct, 0)
        self.assertEqual(controller.session.score_total, 1)

    def test_saved_progress_loads_for_marquinho(self):
        store = TeacherProgressStore(self.tempdir.name)
        profile = store.load("marquinho", "Marquinho")
        profile["last_subject"] = "math"
        profile["last_topic"] = "multiplication"
        profile["last_level"] = 2
        progress = get_level_progress(profile, "math", "multiplication", 2)
        progress["attempted"] = 10
        progress["correct"] = 7
        store.save(profile)
        controller = self.make_controller()
        reply = controller.handle("Miguel, teacher mode", user_id="marquinho", display_name="Marquinho")
        self.assertIn("Marquinho", reply)
        self.assertIn("Level 2", reply)

    def test_saved_progress_is_separate_for_marco(self):
        store = TeacherProgressStore(self.tempdir.name)
        marquinho = store.load("marquinho", "Marquinho")
        marquinho["last_subject"] = "math"
        marquinho["last_topic"] = "multiplication"
        marquinho["last_level"] = 1
        store.save(marquinho)
        marco = store.load("marco", "Marco")
        marco["last_subject"] = "math"
        marco["last_topic"] = "multiplication"
        marco["last_level"] = 4
        store.save(marco)
        controller = self.make_controller()
        reply = controller.handle("teacher mode", user_id="marco", display_name="Marco")
        self.assertIn("Marco", reply)
        self.assertIn("Level 4", reply)

    def test_answer_updates_progress_for_known_user(self):
        controller = self.make_controller()
        controller.handle("teach me multiplication", user_id="marquinho", display_name="Marquinho")
        controller.handle("level 1")
        expected = controller.session.current_question["expected_answer"]
        controller.handle(str(expected))
        saved = TeacherProgressStore(self.tempdir.name).load("marquinho", "Marquinho")
        progress = get_level_progress(saved, "math", "multiplication", 1)
        self.assertEqual(progress["attempted"], 1)
        self.assertEqual(progress["correct"], 1)

    def test_incorrect_multiplication_facts_are_tracked_as_weak(self):
        controller = self.make_controller()
        controller.handle("teach me multiplication", user_id="marquinho", display_name="Marquinho")
        controller.handle("level 1")
        controller.session.current_question = {
            "type": "multiplication",
            "a": 7,
            "b": 8,
            "expected_answer": 56,
            "fact_key": "7x8",
        }
        controller.handle("55")
        saved = TeacherProgressStore(self.tempdir.name).load("marquinho", "Marquinho")
        progress = get_level_progress(saved, "math", "multiplication", 1)
        self.assertEqual(progress["weak_facts"]["7x8"], 1)

    def test_mastery_rule_requires_enough_attempts_and_accuracy(self):
        profile = {"attempted": 9, "correct": 9, "recent_results": [{"correct": True}] * 9}
        self.assertFalse(get_level_progress({"subjects": {}}, "math", "multiplication", 1).get("mastered"))
        self.assertFalse(local_progress_recommendation(profile, 1)["recommended_level"] == 2)
        profile = {"attempted": 10, "correct": 9, "recent_results": [{"correct": True}] * 9 + [{"correct": False}]}
        self.assertTrue(evaluate_mastery(profile))

    def test_recommendation_same_level_for_medium_accuracy(self):
        progress = {"attempted": 10, "correct": 7, "weak_facts": {"7x8": 2}, "recent_results": []}
        rec = local_progress_recommendation(progress, 2)
        self.assertEqual(rec["recommended_level"], 2)
        self.assertIn("7x8", rec["recommended_focus"])

    def test_recommendation_next_level_when_mastered(self):
        progress = {"attempted": 10, "correct": 9, "mastered": True, "weak_facts": {}, "recent_results": []}
        rec = local_progress_recommendation(progress, 2)
        self.assertEqual(rec["recommended_level"], 3)

    def test_score_command(self):
        controller = self.make_controller()
        self.start_level_one(controller)
        expected = controller.session.current_question["expected_answer"]
        controller.handle(str(expected))
        self.assertEqual(controller.handle("what is my score"), "You have answered 1 out of 1 correctly.")

    def test_make_easier_and_make_harder(self):
        controller = self.make_controller()
        self.start_level_one(controller)
        easier = controller.handle("make it easier")
        self.assertEqual(controller.session.difficulty, "easy")
        self.assertIn("make it easier", easier)
        self.assertLessEqual(controller.session.current_question["a"], 5)
        harder = controller.handle("make it harder")
        self.assertEqual(controller.session.difficulty, "hard")
        self.assertIn("make it harder", harder)
        self.assertGreaterEqual(controller.session.current_question["a"], 6)

    def test_exit_teacher_mode(self):
        controller = self.make_controller()
        controller.handle("I am 9 years old teach me multiplication")
        reply = controller.handle("exit teacher mode")
        self.assertEqual(reply, "Teacher mode is off. Great work today.")
        self.assertFalse(controller.session.active)

    def test_successful_cloud_creative_question(self):
        cloud = MockCloudTeacher(
            creative=CloudTeacherContent(
                speech="Mission challenge. Miguel has robot battery packs.",
                question="Miguel has 4 trays with 6 batteries each. How many batteries?",
                expected_answer=24,
                answer_type="number",
                difficulty="normal",
                next_action="wait_for_answer",
            )
        )
        controller = TeacherModeController(
            teachers={"math": MathTeacher(rng=random.Random(7))},
            cloud_client=cloud,
            progress_store=TeacherProgressStore(self.tempdir.name),
        )
        self.start_level_one(controller)
        reply = controller.handle("creative challenge")
        self.assertIn("Mission challenge", reply)
        self.assertEqual(controller.session.current_question["source"], "cloud")
        self.assertEqual(controller.session.current_question["expected_answer"], 24)

    def test_invalid_cloud_response_falls_back_to_local_question(self):
        cloud = MockCloudTeacher(
            creative=CloudTeacherContent(
                speech="Mission challenge.",
                question="Miguel has some trays. How many batteries?",
                expected_answer=None,
                answer_type="number",
                difficulty="normal",
                next_action="wait_for_answer",
            )
        )
        controller = TeacherModeController(
            teachers={"math": MathTeacher(rng=random.Random(7))},
            cloud_client=cloud,
            progress_store=TeacherProgressStore(self.tempdir.name),
        )
        self.start_level_one(controller)
        reply = controller.handle("creative challenge")
        self.assertIn("local mission", reply)
        self.assertNotEqual(controller.session.current_question.get("source"), "cloud")
        self.assertIsNotNone(controller.session.current_question["expected_answer"])

    def test_cloud_unavailable_falls_back_to_local_question(self):
        controller = TeacherModeController(
            teachers={"math": MathTeacher(rng=random.Random(7))},
            cloud_client=MockCloudTeacher(raise_on_creative=True),
            progress_store=TeacherProgressStore(self.tempdir.name),
        )
        self.start_level_one(controller)
        reply = controller.handle("mission challenge")
        self.assertIn("local mission", reply)
        self.assertNotEqual(controller.session.current_question.get("source"), "cloud")

    def test_local_score_tracking_after_cloud_generated_question(self):
        cloud = MockCloudTeacher(
            creative=CloudTeacherContent(
                speech="Space mission.",
                question="A rover carries 3 boxes with 8 sensors each. How many sensors?",
                expected_answer=24,
                answer_type="number",
                difficulty="normal",
                next_action="wait_for_answer",
            )
        )
        controller = TeacherModeController(
            teachers={"math": MathTeacher(rng=random.Random(7))},
            cloud_client=cloud,
            progress_store=TeacherProgressStore(self.tempdir.name),
        )
        self.start_level_one(controller)
        controller.handle("space challenge")
        reply = controller.handle("24")
        self.assertIn("Great job", reply)
        self.assertEqual(controller.session.score_correct, 1)
        self.assertEqual(controller.session.score_total, 1)

    def test_cloud_creative_exercise_rejects_missing_expected_answer(self):
        cloud = MockCloudTeacher(
            creative={
                "speech_intro": "Chief Engineer challenge.",
                "question": "Miguel has mystery parts. How many?",
                "expected_answer": None,
                "answer_type": "number",
                "hint": "Count the groups.",
                "explanation": "Use multiplication.",
                "skill_tags": ["multiplication"],
                "difficulty": "normal",
                "next_action": "wait_for_answer",
            }
        )
        controller = TeacherModeController(
            teachers={"math": MathTeacher(rng=random.Random(7))},
            cloud_client=cloud,
            progress_store=TeacherProgressStore(self.tempdir.name),
        )
        self.start_level_one(controller)
        reply = controller.handle("give me a challenge")
        self.assertIn("local mission", reply)
        self.assertNotEqual(controller.session.current_question.get("source"), "cloud")

    def test_portuguese_cloud_response_rejected_unless_explicitly_requested(self):
        cloud = MockCloudTeacher(
            creative=CloudTeacherContent(
                speech="Missão Miguel.",
                question="Quanto é 3 vezes 8?",
                expected_answer=24,
                answer_type="number",
                difficulty="normal",
                next_action="wait_for_answer",
            )
        )
        controller = TeacherModeController(
            teachers={"math": MathTeacher(rng=random.Random(7))},
            cloud_client=cloud,
            progress_store=TeacherProgressStore(self.tempdir.name),
        )
        self.start_level_one(controller)
        reply = controller.handle("chief engineer challenge")
        self.assertIn("local mission", reply)

        controller = TeacherModeController(
            teachers={"math": MathTeacher(rng=random.Random(7))},
            cloud_client=cloud,
            progress_store=TeacherProgressStore(self.tempdir.name),
        )
        controller.handle("I am 9 years old teach me multiplication", explicit_language="pt")
        controller.handle("level 1")
        reply = controller.handle("chief engineer challenge")
        self.assertIn("Missão", reply)
        self.assertEqual(controller.session.current_question.get("source"), "cloud")

    def test_teacher_progress_does_not_save_for_unknown_guest(self):
        controller = self.make_controller()
        self.start_level_one(controller)
        expected = controller.session.current_question["expected_answer"]
        controller.handle(str(expected))
        self.assertEqual(list(TeacherProgressStore(self.tempdir.name).base_dir.glob("*.json")), [])

    def test_continue_my_level_loads_last_level(self):
        store = TeacherProgressStore(self.tempdir.name)
        profile = store.load("marquinho", "Marquinho")
        profile["last_subject"] = "math"
        profile["last_topic"] = "multiplication"
        profile["last_level"] = 3
        store.save(profile)
        controller = self.make_controller()
        controller.handle("teacher mode", user_id="marquinho", display_name="Marquinho")
        reply = controller.handle("continue my level")
        self.assertIn("Today we will learn multiplication", reply)
        self.assertEqual(controller.session.level_number, 3)


if __name__ == "__main__":
    unittest.main()
