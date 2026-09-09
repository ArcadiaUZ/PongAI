"""
O'yinchi vs AI - Pong o'yini
Siz bilan AI o'ynaydi
"""

import pygame
import sys
import random
import numpy as np
from ai_brain import AIBrain

# Pygame ni ishga tushirish
pygame.init()

# Oynaning o'lchamlari
SCREEN_WIDTH = 1100  # Kengaytirildi - vizualizatsiya uchun
GAME_WIDTH = 800     # O'yin kengligi
SCREEN_HEIGHT = 600
FPS = 60

# Vizualizatsiya ranglari
YELLOW = (255, 255, 0)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)

# Ranglar
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

class Ball:
    def __init__(self):
        self.reset()
        self.radius = 10
        
    def reset(self):
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2
        self.speed_x = random.choice([-5, 5])
        self.speed_y = random.choice([-3, -2, -1, 1, 2, 3])
        
    def move(self):
        self.x += self.speed_x
        self.y += self.speed_y
        
        if self.y <= 0 or self.y >= SCREEN_HEIGHT:
            self.speed_y *= -1
            
    def draw(self, screen):
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), self.radius)

class Paddle:
    def __init__(self, x, is_ai=False):
        self.x = x
        self.y = SCREEN_HEIGHT // 2
        self.width = 15
        self.height = 90
        self.speed = 8
        self.is_ai = is_ai
        self.score = 0
        
    def move_up(self):
        self.y -= self.speed
        if self.y < 0:
            self.y = 0
            
    def move_down(self):
        self.y += self.speed
        if self.y > SCREEN_HEIGHT - self.height:
            self.y = SCREEN_HEIGHT - self.height
                
    def draw(self, screen, color):
        pygame.draw.rect(screen, color, (self.x, self.y, self.width, self.height))

class PlayPong:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pong - Siz vs AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.big_font = pygame.font.Font(None, 72)
        
        # AI ni yuklash (har doim GPU modelini yuklaydi)
        self.ai_brain = AIBrain()
        if self.ai_brain.load("pong_ai_model.pt"):
            print("[OK] O'qitilgan AI yuklandi!")
        else:
            print("[!] AI modeli topilmadi, oddiy AI ishlatiladi")
            print("   Avval 'start_training.bat' ni ishga tushiring!")
            self.ai_brain = None
            
        self.reset_game()
        
        # Vizualizatsiya uchun
        self.input_names = ["To'p X", "To'p Y", "Tez X", "Tez Y", "AI Y"]
        self.output_names = ["Yuqori", "Pastga", "Turish"]
        self.q_values = [0, 0, 0]
        self.current_action = 0
        
    def reset_game(self):
        self.ball = Ball()
        self.player = Paddle(50)  # Chap taraf - o'yinchi
        self.ai = Paddle(GAME_WIDTH - 65, is_ai=True)  # O'ng taraf - AI
        self.game_over = False
        self.winner = None
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if self.game_over:
                        self.reset_game()
                if event.key == pygame.K_ESCAPE:
                    return False
        return True
        
    def handle_player_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.player.move_up()
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.player.move_down()

    def ai_move(self):
        """AI harakati"""
        if self.ai_brain:
            # O'qitilgan AI
            state = self.ai_brain.get_state(self.ball, self.ai)
            action = self.ai_brain.get_action(state)
            
            # Q qiymatlarini saqlash (vizualizatsiya uchun)
            import torch
            with torch.no_grad():
                q_vals = self.ai_brain.brain(state).cpu().numpy()[0]
                self.q_values = q_vals.tolist()
                self.current_action = action
            
            if action == 0:  # Yuqoriga
                self.ai.move_up()
            elif action == 1:  # Pastga
                self.ai.move_down()
            # action == 2: Turish
        else:
            # Oddiy AI
            if self.ball.y < self.ai.y + self.ai.height // 2:
                self.ai.move_up()
            elif self.ball.y > self.ai.y + self.ai.height // 2:
                self.ai.move_down()
            
    def check_collisions(self):
        # To'p va o'yinchi raketkasi (chap taraf)
        if (self.ball.speed_x < 0 and 
            self.ball.x - self.ball.radius <= self.player.x + self.player.width and
            self.ball.x - self.ball.radius >= self.player.x and
            self.player.y < self.ball.y < self.player.y + self.player.height):
            self.ball.speed_x *= -1
            self.ball.speed_y += random.uniform(-1, 1)
            # To'pni raketkasidan tashqariga chiqarish
            self.ball.x = self.player.x + self.player.width + self.ball.radius + 1
            
        # To'p va AI raketkasi (o'ng taraf)
        if (self.ball.speed_x > 0 and 
            self.ball.x + self.ball.radius >= self.ai.x and
            self.ball.x + self.ball.radius <= self.ai.x + self.ai.width and
            self.ai.y < self.ball.y < self.ai.y + self.ai.height):
            self.ball.speed_x *= -1
            self.ball.speed_y += random.uniform(-1, 1)
            # To'pni raketkasidan tashqariga chiqarish
            self.ball.x = self.ai.x - self.ball.radius - 1
            
        # Hisob (to'p chegaradan chiqib ketdi)
        if self.ball.x < -self.ball.radius:
            self.ai.score += 1
            self.ball.reset()
        elif self.ball.x > GAME_WIDTH + self.ball.radius:
            self.player.score += 1
            self.ball.reset()
            
        # O'yin tugashi
        if self.player.score >= 5 or self.ai.score >= 5:
            self.game_over = True
            self.winner = "Siz yutdingiz!" if self.player.score >= 5 else "AI yutti!"
            
    def update(self):
        if not self.game_over:
            self.handle_player_input()
            self.ai_move()
            self.ball.move()
            self.check_collisions()
            
    def draw_ai_viz(self):
        """AI neyron tarmoq vizualizatsiyasi - murakkab"""
        viz_x = GAME_WIDTH + 10
        viz_width = SCREEN_WIDTH - GAME_WIDTH - 20
        small_font = pygame.font.Font(None, 18)
        
        # Fon
        pygame.draw.rect(self.screen, (20, 20, 30), (viz_x, 0, viz_width, SCREEN_HEIGHT))
        
        # Sarlavha
        title = small_font.render("AI Neyron Tarmoq", True, WHITE)
        self.screen.blit(title, (viz_x + 10, 5))
        
        # Neyron tarmoq sxemasi
        input_state = [
            self.ball.x / GAME_WIDTH,
            self.ball.y / SCREEN_HEIGHT,
            self.ball.speed_x / 10,
            self.ball.speed_y / 10,
            self.ai.y / SCREEN_HEIGHT
        ]
        
        # Neyron pozitsiyalari
        input_pos = [(viz_x + 30, 60 + i * 50) for i in range(5)]
        hidden_pos = [(viz_x + 120, 40 + i * 45) for i in range(6)]
        output_pos = [(viz_x + 210, 80 + i * 60) for i in range(3)]
        
        # Input-hidden bog'lanishlar
        for i, ipos in enumerate(input_pos):
            for j, hpos in enumerate(hidden_pos):
                activation = abs(input_state[i])
                c = max(0, min(255, int(100 + 155 * activation)))
                color = (c, c, 100)
                pygame.draw.line(self.screen, color, ipos, hpos, 1)
        
        # Hidden-output bog'lanishlar
        for i, hpos in enumerate(hidden_pos):
            for j, opos in enumerate(output_pos):
                q_val = self.q_values[j] if j < len(self.q_values) else 0
                intensity = max(0, min(255, int(128 + 127 * q_val)))
                color = (intensity, intensity, 100)
                pygame.draw.line(self.screen, color, hpos, opos, 1)
        
        # Input neyronlari
        for i, (pos, val) in enumerate(zip(input_pos, input_state)):
            activation = abs(val)
            r = max(0, min(255, int(255 * activation)))
            g = max(0, min(255, int(255 * (1 - activation))))
            color = (r, g, 0)
            pygame.draw.circle(self.screen, color, pos, 12)
            pygame.draw.circle(self.screen, WHITE, pos, 12, 1)
            
            # Nomi
            name = small_font.render(self.input_names[i], True, GRAY)
            self.screen.blit(name, (pos[0] - 25, pos[1] - 20))
            
            # Qiymat
            val_text = small_font.render(f"{val:.2f}", True, WHITE)
            self.screen.blit(val_text, (pos[0] - 15, pos[1] + 15))
        
        # Hidden neyronlari
        for i, pos in enumerate(hidden_pos):
            pulse = 0.5 + 0.5 * (pygame.time.get_ticks() % 1000 / 1000)
            g = max(0, min(255, int(100 + 155 * pulse)))
            color = (100, g, 255)
            pygame.draw.circle(self.screen, color, pos, 10)
            pygame.draw.circle(self.screen, WHITE, pos, 10, 1)
        
        # Output neyronlari
        for i, (pos, q_val) in enumerate(zip(output_pos, self.q_values)):
            if i == self.current_action:
                color = GREEN
                size = 18
            else:
                intensity = max(0, min(255, int(128 + 127 * q_val)))
                color = (intensity, intensity, intensity)
                size = 14
            
            pygame.draw.circle(self.screen, color, pos, size)
            pygame.draw.circle(self.screen, WHITE, pos, size, 2)
            
            # Nomi
            name = small_font.render(self.output_names[i], True, WHITE)
            self.screen.blit(name, (pos[0] + 25, pos[1] - 8))
            
            # Q qiymati
            q_text = small_font.render(f"Q: {q_val:.2f}", True, YELLOW)
            self.screen.blit(q_text, (pos[0] + 25, pos[1] + 10))
        
        # Joriy harakat ko'rsatkichi
        y_harakat = 380
        pygame.draw.rect(self.screen, DARK_GRAY, (viz_x + 10, y_harakat, viz_width - 20, 40))
        action_text = small_font.render(f"Harakat: {self.output_names[self.current_action]}", True, GREEN)
        self.screen.blit(action_text, (viz_x + 20, y_harakat + 12))
        
        # Epsilon va GPU
        y_info = 430
        if self.ai_brain:
            eps_text = small_font.render(f"Epsilon: {self.ai_brain.epsilon:.4f}", True, GRAY)
            self.screen.blit(eps_text, (viz_x + 10, y_info))
        
        gpu_text = small_font.render("GPU: RTX 3060", True, GRAY)
        self.screen.blit(gpu_text, (viz_x + 10, y_info + 20))
        
        # Q qiymatlari diagrammasi
        y_diagram = 480
        diagram_title = small_font.render("Q Qiymatlari:", True, WHITE)
        self.screen.blit(diagram_title, (viz_x + 10, y_diagram))
        
        for i, (name, q_val) in enumerate(zip(self.output_names, self.q_values)):
            y = y_diagram + 25 + i * 35
            
            # Nomi
            name_text = small_font.render(name, True, GRAY)
            self.screen.blit(name_text, (viz_x + 10, y))
            
            # Indikator
            bar_width = 120
            bar_height = 18
            bar_x = viz_x + 80
            
            pygame.draw.rect(self.screen, DARK_GRAY, (bar_x, y, bar_width, bar_height))
            normalized_q = max(0, min(1, (q_val + 1) / 2))
            fill = int(bar_width * normalized_q)
            
            if i == self.current_action:
                color = GREEN
            else:
                color = BLUE
            
            pygame.draw.rect(self.screen, color, (bar_x, y, fill, bar_height))
            pygame.draw.rect(self.screen, WHITE, (bar_x, y, bar_width, bar_height), 1)
            
            # Qiymat
            val_text = small_font.render(f"{q_val:.2f}", True, YELLOW)
            self.screen.blit(val_text, (bar_x + bar_width + 5, y))
    
    def draw(self):
        self.screen.fill(BLACK)
        
        # O'yin maydonini chizish
        pygame.draw.line(self.screen, WHITE, (GAME_WIDTH // 2, 0), 
                        (GAME_WIDTH // 2, SCREEN_HEIGHT), 2)
        pygame.draw.line(self.screen, WHITE, (GAME_WIDTH // 2 - 10, SCREEN_HEIGHT // 2 - 50),
                        (GAME_WIDTH // 2 - 10, SCREEN_HEIGHT // 2 + 50), 3)
        
        # O'yin elementlarini chizish
        self.ball.draw(self.screen)
        self.player.draw(self.screen, WHITE)
        self.ai.draw(self.screen, GREEN)
        
        # Hisoblarni chizish
        player_score = self.font.render(f"Siz: {self.player.score}", True, WHITE)
        ai_score = self.font.render(f"AI: {self.ai.score}", True, GREEN)
        self.screen.blit(player_score, (GAME_WIDTH // 4 - player_score.get_width() // 2, 20))
        self.screen.blit(ai_score, (3 * GAME_WIDTH // 4 - ai_score.get_width() // 2, 20))
        
        # AI vizualizatsiyasi
        self.draw_ai_viz()
        
        # O'yin tugadi
        if self.game_over:
            winner_text = self.big_font.render(self.winner, True, GREEN)
            restart_text = self.font.render("Qatnashish uchun SPACE, chiqish uchun ESC", True, WHITE)
            self.screen.blit(winner_text, (GAME_WIDTH // 2 - winner_text.get_width() // 2, 
                                         SCREEN_HEIGHT // 2 - 50))
            self.screen.blit(restart_text, (GAME_WIDTH // 2 - restart_text.get_width() // 2, 
                                           SCREEN_HEIGHT // 2 + 50))
        else:
            # Ko'rsatmalar
            controls_text = self.font.render("W/Yuqoriga, S/Pastga | ESC/Chiqish", True, WHITE)
            self.screen.blit(controls_text, (GAME_WIDTH // 2 - controls_text.get_width() // 2, 
                                            SCREEN_HEIGHT - 40))
            
        pygame.display.flip()
        
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
            
        pygame.quit()
        sys.exit()

# O'yinni ishga tushirish
if __name__ == "__main__":
    game = PlayPong()
    game.run()
    def update(self):
        if not self.game_over:
            self.handle_player_input()
            self.ai_move()
            self.ball.move()
            self.check_collisions()
            self.player.move_up()
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.player.move_down()
                
    def draw(self, screen, color=WHITE):
        pygame.draw.rect(screen, color, (self.x, self.y, self.width, self.height))