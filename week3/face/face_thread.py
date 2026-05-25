# ~/robot-project/week3/face/face_thread.py
# Last updated: 20260522

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import pygame

from face_state import FaceEvent, FaceMode


@dataclass
class FaceRuntimeState:
    mode: FaceMode = FaceMode.IDLE
    text: str = ""
    mouth_level: float = 0.0
    last_event_time: float = 0.0


class FaceThread:
    def __init__(
        self,
        event_queue: "queue.Queue[FaceEvent]",
        width: int = 1024,
        height: int = 600,
        fullscreen: bool = True,
        fps: int = 30,
    ) -> None:
        self.event_queue = event_queue
        self.width = width
        self.height = height
        self.fullscreen = fullscreen
        self.fps = fps

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.state = FaceRuntimeState()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._thread = threading.Thread(
            target=self._run,
            name="FaceThread",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    def _drain_events(self) -> None:
        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                return

            self.state.mode = event.mode
            self.state.text = event.text or ""
            self.state.mouth_level = max(0.0, min(1.0, event.mouth_level))
            self.state.last_event_time = time.time()
            print(f"[FACE UI] status={self.state.mode.value} text={self.state.text}".strip())

    def _run(self) -> None:
        pygame.init()
        pygame.display.set_caption("Miguel Face")

        flags = pygame.FULLSCREEN if self.fullscreen else 0
        screen = pygame.display.set_mode((self.width, self.height), flags)
        clock = pygame.time.Clock()

        start_time = time.time()

        while not self._stop_event.is_set():
            for pygame_event in pygame.event.get():
                if pygame_event.type == pygame.QUIT:
                    self._stop_event.set()
                elif pygame_event.type == pygame.KEYDOWN:
                    if pygame_event.key in (pygame.K_ESCAPE, pygame.K_q):
                        self._stop_event.set()

            self._drain_events()
            t = time.time() - start_time
            self._render(screen, t)

            pygame.display.flip()
            clock.tick(self.fps)

        pygame.quit()

    def _render(self, screen: pygame.Surface, t: float) -> None:
        screen.fill((0, 0, 0))

        mode = self.state.mode

        if mode == FaceMode.SLEEPING:
            self._draw_sleeping(screen, t)
            return

        eye_color = self._eye_color(mode)

        self._draw_eyes(screen, t, mode, eye_color)
        self._draw_mouth(screen, t, mode, eye_color)
        self._draw_status_overlay(screen, mode, eye_color)

        if self.state.text:
            if mode == FaceMode.WAKE_REQUIRED:
                self._draw_wake_required_text(screen, self.state.text)
            else:
                self._draw_text(screen, self.state.text)

    def _eye_color(self, mode: FaceMode) -> tuple[int, int, int]:
        if mode == FaceMode.ERROR:
            return (255, 80, 80)
        if mode == FaceMode.WAKE_REQUIRED:
            return (255, 235, 120)
        if mode == FaceMode.LISTENING:
            return (80, 180, 255)
        if mode == FaceMode.THINKING:
            return (180, 120, 255)
        if mode == FaceMode.SPEAKING:
            return (120, 255, 180)
        if mode == FaceMode.CONFIRM:
            return (255, 190, 80)
        if mode == FaceMode.HAPPY:
            return (255, 220, 100)
        if mode == FaceMode.CONFUSED:
            return (255, 170, 80)
        return (120, 220, 255)

    def _draw_eyes(
        self,
        screen: pygame.Surface,
        t: float,
        mode: FaceMode,
        color: tuple[int, int, int],
    ) -> None:
        center_y = int(self.height * 0.38)
        left_x = int(self.width * 0.34)
        right_x = int(self.width * 0.66)

        wobble_x = int(math.sin(t * 2.2) * 7)
        wobble_y = int(math.sin(t * 1.4) * 5)

        if mode == FaceMode.THINKING:
            wobble_x += 18
            wobble_y -= 10
        elif mode == FaceMode.CONFUSED:
            wobble_x += int(math.sin(t * 5.0) * 4)

        blink = self._blink_amount(t)

        eye_w = 145
        eye_h = max(18, int(88 * blink))

        if mode == FaceMode.HAPPY:
            eye_h = max(18, int(55 * blink))

        if mode == FaceMode.LISTENING:
            eye_w = 155
            eye_h = max(22, int(96 * blink))

        if mode == FaceMode.SPEAKING:
            eye_w = 142 + int(math.sin(t * 3.0) * 4)
            eye_h = max(20, int((82 + math.sin(t * 2.5) * 4) * blink))

        self._draw_glow_oval(screen, left_x + wobble_x, center_y + wobble_y, eye_w, eye_h, color)
        self._draw_glow_oval(screen, right_x + wobble_x, center_y + wobble_y, eye_w, eye_h, color)

    def _draw_mouth(
        self,
        screen: pygame.Surface,
        t: float,
        mode: FaceMode,
        color: tuple[int, int, int],
    ) -> None:
        mouth_x = int(self.width * 0.5)
        mouth_y = int(self.height * 0.68)

        if mode == FaceMode.SPEAKING:
            loop_level = 0.5 + 0.5 * math.sin(t * 12.0)
            level = max(self.state.mouth_level, loop_level * 0.75)
            mouth_h = int(20 + level * 65)
            mouth_w = 190
        elif mode == FaceMode.HAPPY:
            mouth_w = 210
            mouth_h = 34
        elif mode == FaceMode.CONFUSED:
            mouth_w = 120
            mouth_h = 18
            mouth_x += 20
        elif mode == FaceMode.THINKING:
            mouth_w = 80
            mouth_h = 12
        elif mode == FaceMode.LISTENING:
            mouth_w = 120 + int(math.sin(t * 5.0) * 8)
            mouth_h = 18
        elif mode == FaceMode.WAKE_REQUIRED:
            mouth_w = 110
            mouth_h = 14
        else:
            mouth_w = 140
            mouth_h = 16

        rect = pygame.Rect(
            mouth_x - mouth_w // 2,
            mouth_y - mouth_h // 2,
            mouth_w,
            mouth_h,
        )

        pygame.draw.rect(screen, color, rect, border_radius=max(1, mouth_h // 2))

    def _draw_sleeping(self, screen: pygame.Surface, t: float) -> None:
        color = (60, 90, 120)
        y = int(self.height * 0.42)

        for x in (int(self.width * 0.35), int(self.width * 0.65)):
            pygame.draw.line(screen, color, (x - 80, y), (x + 80, y), 14)

        font = pygame.font.SysFont("Arial", 52)
        text = font.render("z z z", True, color)
        screen.blit(text, (int(self.width * 0.43), int(self.height * 0.62)))
        self._draw_status_overlay(screen, FaceMode.SLEEPING, color)

    def _draw_text(self, screen: pygame.Surface, text: str) -> None:
        font = pygame.font.SysFont("Arial", 42, bold=True)
        display_text = text[:42].upper()
        rendered = font.render(display_text, True, (235, 245, 255))
        rect = rendered.get_rect(center=(self.width // 2, int(self.height * 0.9)))
        screen.blit(rendered, rect)

    def _draw_wake_required_text(self, screen: pygame.Surface, text: str) -> None:
        display_text = text.strip()[:42] or 'Say "Hey Miguel"'
        font_size = 70 if self.width >= 900 else 56
        font = pygame.font.SysFont("Arial", font_size, bold=True)
        rendered = font.render(display_text, True, (255, 255, 245))
        shadow = font.render(display_text, True, (0, 0, 0))
        rect = rendered.get_rect(center=(self.width // 2, int(self.height * 0.78)))
        shadow_rect = shadow.get_rect(center=(rect.centerx + 4, rect.centery + 4))
        bg_rect = rect.inflate(68, 36)
        bg = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 185))
        screen.blit(bg, bg_rect)
        screen.blit(shadow, shadow_rect)
        screen.blit(rendered, rect)

    def _draw_status_overlay(
        self,
        screen: pygame.Surface,
        mode: FaceMode,
        color: tuple[int, int, int],
    ) -> None:
        label = {
            FaceMode.LISTENING: "LISTENING",
            FaceMode.WAKE_REQUIRED: "WAKE",
            FaceMode.THINKING: "THINKING",
            FaceMode.SPEAKING: "SPEAKING",
            FaceMode.IDLE: "IDLE",
            FaceMode.SLEEPING: "SLEEPING",
            FaceMode.CONFIRM: "CONFIRM",
            FaceMode.ERROR: "ERROR",
            FaceMode.CONFUSED: "CONFUSED",
            FaceMode.HAPPY: "READY",
        }.get(mode, mode.value.upper())

        dot_x = int(self.width * 0.08)
        dot_y = int(self.height * 0.1)
        pygame.draw.circle(screen, color, (dot_x, dot_y), 18)

        font = pygame.font.SysFont("Arial", 28, bold=True)
        rendered = font.render(label, True, color)
        screen.blit(rendered, (dot_x + 34, dot_y - 17))

    def _draw_glow_oval(
        self,
        screen: pygame.Surface,
        x: int,
        y: int,
        w: int,
        h: int,
        color: tuple[int, int, int],
    ) -> None:
        glow_color = tuple(max(0, int(c * 0.25)) for c in color)

        for scale in (1.8, 1.45, 1.2):
            rect = pygame.Rect(
                x - int(w * scale) // 2,
                y - int(h * scale) // 2,
                int(w * scale),
                int(h * scale),
            )
            pygame.draw.ellipse(screen, glow_color, rect)

        rect = pygame.Rect(x - w // 2, y - h // 2, w, h)
        pygame.draw.ellipse(screen, color, rect)

    def _blink_amount(self, t: float) -> float:
        period = 4.0
        phase = t % period

        if phase < 0.10:
            return max(0.12, phase / 0.10)
        if 0.10 <= phase < 0.20:
            return max(0.12, 1.0 - ((phase - 0.10) / 0.10))
        return 1.0
