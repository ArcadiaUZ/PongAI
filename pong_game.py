"""
Pong O'yini - Asosiy o'yin muhiti
Siz bilan AI o'ynash uchun tayyorlanmoqda
"""

import pygame
import sys
import random

# Pygame ni ishga tushirish
pygame.init()

# Oynaning o'lchamlari
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

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
        
        # Yuqori va pastki chegaralardan qaytish
        if self.y <= 0 or self.y >= SCREEN_HEIGHT:
            self.speed_y *= -1
            
    def draw(self, screen):
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), self.radius)
        
    def get_state(self):
        return {
            'x': self.x / SCREEN_WIDTH,
            'y': self.y / SCREEN_HEIGHT,
            'speed_x': self.speed_x / 10,
            'speed_y': self.speed_y / 10
        }
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
            
    def ai_move(self, ball):
        """Oddiy AI - to'pni kuzatadi"""
        if self.is_ai:
            if ball.y < self.y + self.height // 2:
                self.move_up()
            elif ball.y > self.y + self.height // 2:
                self.move_down()
                
    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, (self.x, self.y, self.width, self.height))
        
    def get_state(self):
        return {
            'y': self.y / SCREEN_HEIGHT,
            'height': self.height / SCREEN_HEIGHT
        }

class PongGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pong - AI bilan o'yin")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.big_font = pygame.font.Font(None, 72)
        
        self.reset_game()
        
    def reset_game(self):
        self.ball = Ball()
        self.player = Paddle(50)  # Chap taraf - o'yinchi
        self.ai = Paddle(SCREEN_WIDTH - 65, is_ai=True)  # O'ng taraf - AI
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
            
    def check_collisions(self):
        # To'p va raketka orasidagi to'qnashuv
        if (self.ball.x - self.ball.radius <= self.player.x + self.player.width and
            self.player.y < self.ball.y < self.player.y + self.player.height):
            self.ball.speed_x *= -1
            self.ball.speed_y += random.uniform(-1, 1)
            
        if (self.ball.x + self.ball.radius >= self.ai.x and
            self.ai.y < self.ball.y < self.ai.y + self.ai.height):
            self.ball.speed_x *= -1
            self.ball.speed_y += random.uniform(-1, 1)
            
        # Hisob
        if self.ball.x < 0:
            self.ai.score += 1
            self.ball.reset()
        elif self.ball.x > SCREEN_WIDTH:
            self.player.score += 1
            self.ball.reset()
            
        # O'yin tugashi
        if self.player.score >= 5 or self.ai.score >= 5:
            self.game_over = True
            self.winner = "Siz yutdingiz!" if self.player.score >= 5 else "AI yutti!"
            
    def update(self):
        if not self.game_over:
            self.handle_player_input()
            self.ai.ai_move(self.ball)
            self.ball.move()
            self.check_collisions()

    def draw(self):
        self.screen.fill(BLACK)
        
        # O'yin maydonini chizish
        pygame.draw.line(self.screen, WHITE, (SCREEN_WIDTH // 2, 0), 
                        (SCREEN_WIDTH // 2, SCREEN_HEIGHT), 2)
        pygame.draw.line(self.screen, WHITE, (SCREEN_WIDTH // 2 - 10, SCREEN_HEIGHT // 2 - 50),
                        (SCREEN_WIDTH // 2 - 10, SCREEN_HEIGHT // 2 + 50), 3)
        
        # O'yin elementlarini chizish
        self.ball.draw(self.screen)
        self.player.draw(self.screen)
        self.ai.draw(self.screen)
        
        # Hisoblarni chizish
        player_score = self.font.render(f"Siz: {self.player.score}", True, WHITE)
        ai_score = self.font.render(f"AI: {self.ai.score}", True, WHITE)
        self.screen.blit(player_score, (SCREEN_WIDTH // 4 - player_score.get_width() // 2, 20))
        self.screen.blit(ai_score, (3 * SCREEN_WIDTH // 4 - ai_score.get_width() // 2, 20))
        
        # O'yin tugadi
        if self.game_over:
            winner_text = self.big_font.render(self.winner, True, GREEN)
            restart_text = self.font.render("Qatnashish uchun SPACE, chiqish uchun ESC", True, WHITE)
            self.screen.blit(winner_text, (SCREEN_WIDTH // 2 - winner_text.get_width() // 2, 
                                         SCREEN_HEIGHT // 2 - 50))
            self.screen.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, 
                                           SCREEN_HEIGHT // 2 + 50))
        else:
            # Ko'rsatmalar
            controls_text = self.font.render("W/Yuqoriga, S/Pastga | ESC/Chiqish", True, WHITE)
            self.screen.blit(controls_text, (SCREEN_WIDTH // 2 - controls_text.get_width() // 2, 
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
    game = PongGame()
    game.run()