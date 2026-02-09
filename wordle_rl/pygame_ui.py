from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pygame

from .env import Feedback, WordleEnv


@dataclass(frozen=True)
class UIConfig:
    max_attempts: int = 6
    word_length: int = 5
    hard_mode: bool = False
    fps: int = 60
    tile_size: int = 64
    tile_gap: int = 8
    margin: int = 24


BACKGROUND = (18, 18, 19)
TEXT = (245, 245, 245)
MUTED = (155, 155, 160)
BORDER = (58, 58, 61)
TILE_EMPTY = (28, 28, 31)
TILE_GREEN = (83, 141, 78)
TILE_YELLOW = (181, 159, 59)
TILE_GREY = (58, 58, 60)
INPUT_BORDER = (86, 86, 90)
ALPHA_UNKNOWN = (70, 70, 74)
ALPHA_ABSENT = (58, 58, 60)
ALPHA_PRESENT = (181, 159, 59)
ALPHA_FIXED = (83, 141, 78)
KEY_TEXT_DARK = (32, 32, 33)


def _reason_to_message(reason: Optional[str]) -> str:
    if reason == "guess_not_in_allowed_list":
        return "Not in word list."
    if reason == "hard_mode_missing_green":
        return "Hard mode: keep revealed green letters fixed."
    if reason == "hard_mode_yellow_position_violation":
        return "Hard mode: yellow letters cannot stay in same spot."
    if reason == "hard_mode_missing_present_letter":
        return "Hard mode: use all revealed present letters."
    return "Invalid guess."


def _tile_color(code: int) -> Tuple[int, int, int]:
    if code == int(Feedback.GREEN):
        return TILE_GREEN
    if code == int(Feedback.YELLOW):
        return TILE_YELLOW
    if code == int(Feedback.GREY):
        return TILE_GREY
    return TILE_EMPTY


def _alpha_color(code: int) -> Tuple[int, int, int]:
    if code == 3:
        return ALPHA_FIXED
    if code == 2:
        return ALPHA_PRESENT
    if code == 1:
        return ALPHA_ABSENT
    return ALPHA_UNKNOWN


class WordlePygameApp:
    def __init__(self, config: UIConfig) -> None:
        self.cfg = config
        self.env = WordleEnv(
            max_attempts=config.max_attempts,
            word_length=config.word_length,
            hard_mode=config.hard_mode,
        )

        pygame.init()
        pygame.display.set_caption("Wordle RL UI")

        board_w = config.word_length * config.tile_size + (config.word_length - 1) * config.tile_gap
        board_h = config.max_attempts * config.tile_size + (config.max_attempts - 1) * config.tile_gap
        self.width = max(720, board_w + config.margin * 2)
        self.height = board_h + 320
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("arial", 36, bold=True)
        self.font_tile = pygame.font.SysFont("arial", 34, bold=True)
        self.font_text = pygame.font.SysFont("arial", 24)
        self.font_small = pygame.font.SysFont("arial", 18)
        self.font_key = pygame.font.SysFont("arial", 20, bold=True)

        self.current_input = ""
        self.history: List[Tuple[str, List[int]]] = []
        self.obs: Dict[str, object] = {}
        self.message = ""
        self.game_over = False
        self.game_win = False

        self._reset()

    def _reset(self) -> None:
        self.obs, _ = self.env.reset()
        self.current_input = ""
        self.history.clear()
        self.message = "Type letters and press Enter. Press R for a new game."
        self.game_over = False
        self.game_win = False

    def _submit_guess(self) -> None:
        if self.game_over:
            return
        if len(self.current_input) != self.cfg.word_length:
            self.message = f"Need {self.cfg.word_length} letters."
            return

        obs, _, terminated, truncated, info = self.env.step(self.current_input)
        self.obs = obs

        if not info["is_valid_guess"]:
            self.message = _reason_to_message(info.get("reason"))
            return

        feedback = info["feedback"].tolist()
        self.history.append((self.current_input, feedback))
        self.current_input = ""

        if terminated:
            answer = info["answer"]
            self.game_over = True
            self.game_win = bool(answer == self.history[-1][0])
            if self.game_win:
                self.message = f"Solved in {len(self.history)} guesses. Press R to restart."
            else:
                self.message = f"Out of guesses. Answer: {answer.upper()} (press R)"
            return

        if truncated:
            self.game_over = True
            answer = info.get("answer")
            self.message = f"Episode truncated. Answer: {str(answer).upper()} (press R)"
            return

        self.message = f"Guess {len(self.history) + 1}/{self.cfg.max_attempts}"

    def _on_keydown(self, event: pygame.event.Event) -> None:
        if event.key == pygame.K_ESCAPE:
            pygame.quit()
            sys.exit(0)

        if event.key == pygame.K_r:
            self._reset()
            return

        if self.game_over:
            return

        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._submit_guess()
            return

        if event.key == pygame.K_BACKSPACE:
            self.current_input = self.current_input[:-1]
            return

        if event.unicode and event.unicode.isalpha():
            if len(self.current_input) < self.cfg.word_length:
                self.current_input += event.unicode.lower()

    def _draw_board(self) -> None:
        cfg = self.cfg
        board_x = (self.width - (cfg.word_length * cfg.tile_size + (cfg.word_length - 1) * cfg.tile_gap)) // 2
        board_y = 90

        for row in range(cfg.max_attempts):
            for col in range(cfg.word_length):
                x = board_x + col * (cfg.tile_size + cfg.tile_gap)
                y = board_y + row * (cfg.tile_size + cfg.tile_gap)
                rect = pygame.Rect(x, y, cfg.tile_size, cfg.tile_size)

                letter = ""
                color = TILE_EMPTY
                border = BORDER

                if row < len(self.history):
                    guess, fb = self.history[row]
                    letter = guess[col].upper()
                    color = _tile_color(fb[col])
                    border = color
                elif row == len(self.history) and col < len(self.current_input):
                    letter = self.current_input[col].upper()
                    color = TILE_EMPTY
                    border = INPUT_BORDER

                pygame.draw.rect(self.screen, color, rect, border_radius=6)
                pygame.draw.rect(self.screen, border, rect, width=2, border_radius=6)
                if letter:
                    text = self.font_tile.render(letter, True, TEXT)
                    text_rect = text.get_rect(center=rect.center)
                    self.screen.blit(text, text_rect)

    def _draw_keyboard(self) -> None:
        rows = ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM"]
        key_w = 48
        key_h = 56
        gap = 6
        start_y = self.height - 200
        alphabet = self.obs.get("alphabet_status")
        alpha_status = alphabet.tolist() if hasattr(alphabet, "tolist") else [0] * 26

        for row_idx, row in enumerate(rows):
            row_w = len(row) * key_w + (len(row) - 1) * gap
            row_x = (self.width - row_w) // 2
            y = start_y + row_idx * (key_h + gap)
            for i, ch in enumerate(row):
                x = row_x + i * (key_w + gap)
                rect = pygame.Rect(x, y, key_w, key_h)
                code = alpha_status[ord(ch.lower()) - ord("a")]
                color = _alpha_color(int(code))
                pygame.draw.rect(self.screen, color, rect, border_radius=6)
                pygame.draw.rect(self.screen, BORDER, rect, width=1, border_radius=6)
                label_color = KEY_TEXT_DARK if color != ALPHA_UNKNOWN else TEXT
                text = self.font_key.render(ch, True, label_color)
                self.screen.blit(text, text.get_rect(center=rect.center))

    def _draw(self) -> None:
        self.screen.fill(BACKGROUND)
        title = self.font_title.render("WORDLE", True, TEXT)
        self.screen.blit(title, title.get_rect(center=(self.width // 2, 36)))

        subtitle_text = "Hard Mode ON" if self.cfg.hard_mode else "Hard Mode OFF"
        subtitle = self.font_small.render(subtitle_text, True, MUTED)
        self.screen.blit(subtitle, subtitle.get_rect(center=(self.width // 2, 64)))

        self._draw_board()
        self._draw_keyboard()

        msg = self.font_text.render(self.message, True, TEXT)
        self.screen.blit(msg, msg.get_rect(center=(self.width // 2, self.height - 230)))

        help_text = self.font_small.render("Enter=submit  Backspace=delete  R=restart  Esc=quit", True, MUTED)
        self.screen.blit(help_text, help_text.get_rect(center=(self.width // 2, self.height - 18)))

        pygame.display.flip()

    def run(self) -> None:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    self._on_keydown(event)
            self._draw()
            self.clock.tick(self.cfg.fps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Wordle with a Pygame UI.")
    parser.add_argument("--hard-mode", action="store_true", help="Enable hard mode.")
    parser.add_argument("--max-attempts", type=int, default=6, help="Number of attempts per game.")
    parser.add_argument("--word-length", type=int, default=5, help="Word length.")
    args = parser.parse_args()

    config = UIConfig(
        hard_mode=args.hard_mode,
        max_attempts=args.max_attempts,
        word_length=args.word_length,
    )
    app = WordlePygameApp(config)
    app.run()


if __name__ == "__main__":
    main()
