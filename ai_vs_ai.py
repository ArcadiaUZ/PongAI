"""
AI vs AI - Pong o'yini (4x kuchaytirilgan)
Yangi AI (chap) vs Eski AI (o'ng) - ikkala neyron tarmog'i vizualizatsiyasi
"""

import pygame
import sys
import random
import numpy as np
import torch
from ai_brain import AIBrain, device, INPUT_SIZE

pygame.init()

SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 700
GAME_WIDTH = 800
GAME_HEIGHT = 600
GAME_OFFSET_X = (SCREEN_WIDTH - GAME_WIDTH) // 2
GAME_OFFSET_Y = 50
FPS = 60

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
GRAY = (100, 100, 100)
DARK_GRAY = (30, 30, 40)
LIGHT_GRAY = (180, 180, 180)

class Ball:
    def __init__(self):
        self.reset()
        self.radius = 10

    def reset(self):
        self.x = GAME_WIDTH // 2
        self.y = GAME_HEIGHT // 2
        self.speed_x = random.choice([-5, 5])
        self.speed_y = random.choice([-3, -2, -1, 1, 2, 3])

    def move(self):
        self.x += self.speed_x
        self.y += self.speed_y
        if self.y <= 0 or self.y >= GAME_HEIGHT:
            self.speed_y *= -1

    def draw(self, screen):
        sx = int(self.x) + GAME_OFFSET_X
        sy = int(self.y) + GAME_OFFSET_Y
        pygame.draw.circle(screen, WHITE, (sx, sy), self.radius)

class Paddle:
    def __init__(self, x, color=WHITE):
        self.x = x
        self.y = GAME_HEIGHT // 2
        self.width = 15
        self.height = 90
        self.speed = 8
        self.color = color
        self.score = 0

    def move_up(self):
        self.y -= self.speed
        if self.y < 0:
            self.y = 0

    def move_down(self):
        self.y += self.speed
        if self.y > GAME_HEIGHT - self.height:
            self.y = GAME_HEIGHT - self.height

    def draw(self, screen):
        sx = self.x + GAME_OFFSET_X
        sy = self.y + GAME_OFFSET_Y
        pygame.draw.rect(screen, self.color, (sx, sy, self.width, self.height))

class AIVsAI:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("AI vs AI - 4x Kuchaytirilgan")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.big_font = pygame.font.Font(None, 64)
        self.small_font = pygame.font.Font(None, 16)
        self.tiny_font = pygame.font.Font(None, 14)

        self.ai_left = AIBrain(hidden_size=512)
        self.ai_left.load("pong_ai_model.pt")
        print("[OK] Chap AI yuklandi (hidden=512)")

        self.ai_right = AIBrain(hidden_size=512)
        self.ai_right.load("pong_ai_model.pt")
        print("[OK] O'ng AI yuklandi (hidden=512)")

        self.input_names = [
            "To'p X", "To'p Y", "Tez X", "Tez Y", "AI Y",
            "Raqib Y", "Masofa", "Vert Dif", "Kelyapti", "Last Act"
        ]
        self.output_names = ["Yuqori", "Pastga", "Turish"]

        self.left_q_values = [0, 0, 0]
        self.left_action = 2
        self.left_input_state = [0] * INPUT_SIZE
        self.right_q_values = [0, 0, 0]
        self.right_action = 2
        self.right_input_state = [0] * INPUT_SIZE

        self.reset_game()

    def reset_game(self):
        self.ball = Ball()
        self.left_paddle = Paddle(50, GREEN)
        self.right_paddle = Paddle(GAME_WIDTH - 65, BLUE)
        self.game_over = False
        self.winner = None
        self.rally_count = 0
        self.max_rally = 0
        self.round_num = 1
        self.total_rounds = 100
        self.max_steps = 600
        self.steps = 0

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and self.game_over:
                    self.reset_game()
                if event.key == pygame.K_ESCAPE:
                    return False
        return True

    def get_left_state(self):
        mirrored_ball_x = 1.0 - self.ball.x / GAME_WIDTH
        mirrored_ball_y = self.ball.y / GAME_HEIGHT
        mirrored_speed_x = -self.ball.speed_x / 10
        mirrored_speed_y = self.ball.speed_y / 10
        mirrored_paddle_y = self.left_paddle.y / GAME_HEIGHT
        opponent_y = self.right_paddle.y / GAME_HEIGHT
        mirrored_paddle_center = (self.left_paddle.y + self.left_paddle.height / 2) / GAME_HEIGHT
        mirrored_paddle_x_norm = (GAME_WIDTH - self.left_paddle.x - self.left_paddle.width) / GAME_WIDTH
        dist = mirrored_ball_x - mirrored_paddle_x_norm
        vert_diff = mirrored_ball_y - mirrored_paddle_center
        coming = 1.0 if self.ball.speed_x < 0 else 0.0
        return [
            mirrored_ball_x,
            mirrored_ball_y,
            mirrored_speed_x,
            mirrored_speed_y,
            mirrored_paddle_y,
            opponent_y,
            dist,
            vert_diff,
            coming,
            0.0,
        ]

    def get_right_state(self):
        paddle_center = (self.right_paddle.y + self.right_paddle.height / 2) / GAME_HEIGHT
        return [
            self.ball.x / GAME_WIDTH,
            self.ball.y / GAME_HEIGHT,
            self.ball.speed_x / 10,
            self.ball.speed_y / 10,
            self.right_paddle.y / GAME_HEIGHT,
            self.left_paddle.y / GAME_HEIGHT,
            (self.ball.x - self.right_paddle.x) / GAME_WIDTH,
            (self.ball.y / GAME_HEIGHT) - paddle_center,
            1.0 if self.ball.speed_x > 0 else 0.0,
            0.0,
        ]

    def ai_move_left(self):
        state_tensor = torch.FloatTensor(self.get_left_state()).unsqueeze(0).to(device)
        with torch.no_grad():
            q_vals = self.ai_left.brain(state_tensor).cpu().numpy()[0]
            self.left_q_values = q_vals.tolist()
            action = int(np.argmax(q_vals))
            self.left_action = action
            self.left_input_state = self.get_left_state()
        if action == 0:
            self.left_paddle.move_up()
        elif action == 1:
            self.left_paddle.move_down()

    def ai_move_right(self):
        state_tensor = torch.FloatTensor(self.get_right_state()).unsqueeze(0).to(device)
        with torch.no_grad():
            q_vals = self.ai_right.brain(state_tensor).cpu().numpy()[0]
            self.right_q_values = q_vals.tolist()
            action = int(np.argmax(q_vals))
            self.right_action = action
            self.right_input_state = self.get_right_state()
        if action == 0:
            self.right_paddle.move_up()
        elif action == 1:
            self.right_paddle.move_down()

    def check_collisions(self):
        if (self.ball.speed_x < 0 and
            self.ball.x - self.ball.radius <= self.left_paddle.x + self.left_paddle.width and
            self.ball.x - self.ball.radius >= self.left_paddle.x and
            self.left_paddle.y < self.ball.y < self.left_paddle.y + self.left_paddle.height):
            self.ball.speed_x *= -1
            self.ball.speed_y += random.uniform(-1, 1)
            self.ball.x = self.left_paddle.x + self.left_paddle.width + self.ball.radius + 1
            self.rally_count += 1
            self.max_rally = max(self.max_rally, self.rally_count)

        if (self.ball.speed_x > 0 and
            self.ball.x + self.ball.radius >= self.right_paddle.x and
            self.ball.x + self.ball.radius <= self.right_paddle.x + self.right_paddle.width and
            self.right_paddle.y < self.ball.y < self.right_paddle.y + self.right_paddle.height):
            self.ball.speed_x *= -1
            self.ball.speed_y += random.uniform(-1, 1)
            self.ball.x = self.right_paddle.x - self.ball.radius - 1
            self.rally_count += 1
            self.max_rally = max(self.max_rally, self.rally_count)

        if self.ball.x < -self.ball.radius:
            self.right_paddle.score += 1
            self.ball.reset()
            self.rally_count = 0
            self.round_num += 1
            self.steps = 0
        elif self.ball.x > GAME_WIDTH + self.ball.radius:
            self.left_paddle.score += 1
            self.ball.reset()
            self.rally_count = 0
            self.round_num += 1
            self.steps = 0

        if self.round_num > self.total_rounds:
            self.game_over = True
            if self.left_paddle.score > self.right_paddle.score:
                self.winner = f"CHAP AI YUTDI! {self.left_paddle.score}:{self.right_paddle.score}"
            elif self.right_paddle.score > self.left_paddle.score:
                self.winner = f"O'NG AI YUTDI! {self.right_paddle.score}:{self.left_paddle.score}"
            else:
                self.winner = f"DURANG! {self.left_paddle.score}:{self.right_paddle.score}"

    def update(self):
        if not self.game_over:
            self.ai_move_left()
            self.ai_move_right()
            self.ball.move()
            self.check_collisions()
            self.steps += 1
            if self.steps >= self.max_steps:
                self.round_num += 1
                self.steps = 0
                self.ball.reset()
                self.rally_count = 0

    def draw_neural_viz(self, vx, vy, vw, vh, title, input_state, q_values, action, color_scheme):
        pygame.draw.rect(self.screen, DARK_GRAY, (vx, vy, vw, vh))
        pygame.draw.rect(self.screen, color_scheme[0], (vx, vy, vw, vh), 2)

        title_text = self.small_font.render(title, True, color_scheme[0])
        self.screen.blit(title_text, (vx + 10, vy + 5))

        n_input = len(input_state)
        n_show = min(n_input, 6)
        cols = [60, 160, 260]
        input_y = [vy + 40 + i * 35 for i in range(n_show)]
        hidden_y = [vy + 30 + i * 40 for i in range(8)]
        output_y = [vy + 70 + i * 70 for i in range(3)]

        input_pos = [(vx + cols[0], iy) for iy in input_y]
        hidden_pos = [(vx + cols[1], hy) for hy in hidden_y]
        output_pos = [(vx + cols[2], oy) for oy in output_y]

        for i, ipos in enumerate(input_pos):
            for j, hpos in enumerate(hidden_pos):
                activation = abs(input_state[i]) if i < len(input_state) else 0
                c = max(0, min(255, int(80 + 175 * activation)))
                pygame.draw.line(self.screen, (c, c, 100), ipos, hpos, 1)

        for i, hpos in enumerate(hidden_pos):
            for j, opos in enumerate(output_pos):
                q_val = q_values[j] if j < len(q_values) else 0
                intensity = max(0, min(255, int(100 + 155 * q_val)))
                pygame.draw.line(self.screen, (intensity, intensity, 80), hpos, opos, 1)

        for i, (pos, val) in enumerate(zip(input_pos, input_state[:n_show])):
            activation = abs(val)
            r = max(0, min(255, int(255 * activation)))
            g = max(0, min(255, int(255 * (1 - activation))))
            pygame.draw.circle(self.screen, (r, g, 0), pos, 10)
            pygame.draw.circle(self.screen, WHITE, pos, 10, 1)
            name = self.tiny_font.render(self.input_names[i], True, GRAY)
            self.screen.blit(name, (pos[0] - 20, pos[1] - 18))
            val_text = self.tiny_font.render(f"{val:.2f}", True, WHITE)
            self.screen.blit(val_text, (pos[0] - 15, pos[1] + 13))

        for i, pos in enumerate(hidden_pos):
            pulse = 0.5 + 0.5 * (pygame.time.get_ticks() % 1000 / 1000)
            g = max(0, min(255, int(80 + 175 * pulse)))
            pygame.draw.circle(self.screen, (80, g, 255), pos, 9)
            pygame.draw.circle(self.screen, WHITE, pos, 9, 1)

        for i, (pos, q_val) in enumerate(zip(output_pos, q_values)):
            if i == action:
                pygame.draw.circle(self.screen, color_scheme[0], pos, 15)
            else:
                intensity = max(0, min(255, int(100 + 155 * q_val)))
                pygame.draw.circle(self.screen, (intensity, intensity, intensity), pos, 11)
            pygame.draw.circle(self.screen, WHITE, pos, 15 if i == action else 11, 2)
            name = self.tiny_font.render(self.output_names[i], True, WHITE)
            self.screen.blit(name, (pos[0] + 20, pos[1] - 7))
            q_text = self.tiny_font.render(f"Q:{q_val:.2f}", True, YELLOW)
            self.screen.blit(q_text, (pos[0] + 20, pos[1] + 8))

        y_info = vy + vh - 140
        pygame.draw.rect(self.screen, (40, 40, 50), (vx + 8, y_info, vw - 16, 28))
        act_text = self.small_font.render(f"HARAKAT: {self.output_names[action]}", True, color_scheme[0])
        self.screen.blit(act_text, (vx + 14, y_info + 6))

        y_bar = y_info + 35
        for i, (name, q_val) in enumerate(zip(self.output_names, q_values)):
            by = y_bar + i * 20
            nt = self.tiny_font.render(name[:3], True, GRAY)
            self.screen.blit(nt, (vx + 10, by))
            bar_w = vw - 120
            bar_h = 12
            bar_x = vx + 45
            pygame.draw.rect(self.screen, (40, 40, 50), (bar_x, by, bar_w, bar_h))
            nq = max(0, min(1, (q_val + 1.5) / 3))
            fill = int(bar_w * nq)
            bc = color_scheme[0] if i == action else color_scheme[1]
            pygame.draw.rect(self.screen, bc, (bar_x, by, fill, bar_h))
            pygame.draw.rect(self.screen, WHITE, (bar_x, by, bar_w, bar_h), 1)
            vt = self.tiny_font.render(f"{q_val:.2f}", True, YELLOW)
            self.screen.blit(vt, (bar_x + bar_w + 4, by))

    def draw(self):
        self.screen.fill(BLACK)

        gx = GAME_OFFSET_X
        gy = GAME_OFFSET_Y

        viz_width = (SCREEN_WIDTH - GAME_WIDTH) // 2 - 10
        viz_height = SCREEN_HEIGHT - 20

        self.draw_neural_viz(
            5, 10, viz_width, viz_height,
            "CHAP AI (YASHIL)",
            self.left_input_state, self.left_q_values, self.left_action,
            (GREEN, (0, 200, 100))
        )

        self.draw_neural_viz(
            gx + GAME_WIDTH + 5, 10, viz_width, viz_height,
            "O'NG AI (KOK)",
            self.right_input_state, self.right_q_values, self.right_action,
            (BLUE, (100, 150, 255))
        )

        pygame.draw.rect(self.screen, (15, 15, 25), (gx, gy, GAME_WIDTH, GAME_HEIGHT))
        pygame.draw.rect(self.screen, WHITE, (gx, gy, GAME_WIDTH, GAME_HEIGHT), 2)

        cx = gx + GAME_WIDTH // 2
        pygame.draw.line(self.screen, WHITE, (cx, gy), (cx, gy + GAME_HEIGHT), 2)
        pygame.draw.line(self.screen, WHITE, (cx - 8, gy + GAME_HEIGHT // 2 - 40),
                         (cx - 8, gy + GAME_HEIGHT // 2 + 40), 3)

        self.ball.draw(self.screen)
        self.left_paddle.draw(self.screen)
        self.right_paddle.draw(self.screen)

        ls = self.font.render(f"CHAP AI: {self.left_paddle.score}", True, GREEN)
        rs = self.font.render(f"O'NG AI: {self.right_paddle.score}", True, BLUE)
        self.screen.blit(ls, (gx + GAME_WIDTH // 4 - ls.get_width() // 2, gy + 10))
        self.screen.blit(rs, (gx + 3 * GAME_WIDTH // 4 - rs.get_width() // 2, gy + 10))

        rt = self.font.render(f"Raund: {min(self.round_num, self.total_rounds)}/{self.total_rounds}  Rally: {self.rally_count}  Max: {self.max_rally}", True, YELLOW)
        self.screen.blit(rt, (cx - rt.get_width() // 2, gy + 10))

        if self.game_over:
            wc = GREEN if "CHAP" in self.winner else BLUE
            wt = self.big_font.render(self.winner, True, wc)
            rxt = self.font.render("SPACE - Yangi o'yin | ESC - Chiqish", True, WHITE)
            self.screen.blit(wt, (gx + GAME_WIDTH // 2 - wt.get_width() // 2,
                                  gy + GAME_HEIGHT // 2 - 50))
            self.screen.blit(rxt, (gx + GAME_WIDTH // 2 - rxt.get_width() // 2,
                                   gy + GAME_HEIGHT // 2 + 50))
        else:
            ct = self.tiny_font.render("ESC - Chiqish", True, GRAY)
            self.screen.blit(ct, (gx + GAME_WIDTH // 2 - ct.get_width() // 2,
                                  gy + GAME_HEIGHT - 20))

        pygame.display.flip()

    def run(self):
        running = True
        print("\nAI vs AI boshlandi! (4x kuchaytirilgan)")
        print("Chap (Yashil) = Chap AI")
        print("O'ng (Kok) = O'ng AI")
        print("ESC - Chiqish\n")

        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = AIVsAI()
    game.run()
