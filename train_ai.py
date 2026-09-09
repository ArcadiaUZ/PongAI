"""
AI ni o'qitish - Pong o'yini
AI o'z-o'zini o'rganadi
"""

import pygame
import sys
import random
import numpy as np
from ai_brain import AIBrain

# Pygame ni ishga tushirish
pygame.init()

# Oynaning o'lchamlari
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Ranglar
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

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

class Paddle:
    def __init__(self, x):
        self.x = x
        self.y = SCREEN_HEIGHT // 2
        self.width = 15
        self.height = 90
        self.speed = 8
        self.score = 0
        
    def move_up(self):
        self.y -= self.speed
        if self.y < 0:
            self.y = 0
            
    def move_down(self):
        self.y += self.speed
        if self.y > SCREEN_HEIGHT - self.height:
            self.y = SCREEN_HEIGHT - self.height


class TrainPong:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("AI O'qitish - Pong")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        self.ai_brain = AIBrain()
        self.model_file = "pong_ai_model.pt"
        self.checkpoint_file = "training_checkpoint.txt"
        
        # Checkpoint dan yuklash
        self.start_episode = self.load_checkpoint()
        self.max_episodes = self.start_episode + 100000
        
    def load_checkpoint(self):
        """Checkpoint dan o'qitishni davom etirish"""
        try:
            with open(self.checkpoint_file, 'r') as f:
                start_episode = int(f.read().strip())
                print(f"[CHECKPOINT] {start_episode} epizoddan davom etiladi...")
                # Modelni yuklash
                if self.ai_brain.load(self.model_file):
                    return start_episode
        except:
            pass
        print("[CHECKPOINT] Yangi o'qitish boshlanmoqda...")
        return 0
    
    def save_checkpoint(self, episode):
        """Checkpoint ni saqlash"""
        with open(self.checkpoint_file, 'w') as f:
            f.write(str(episode))
    
    def reset_game(self):
        self.ball = Ball()
        self.ai_paddle = Paddle(SCREEN_WIDTH - 65)
        self.opponent = Paddle(50)
        self.done = False
        self.steps = 0
        self.max_steps = 300
        
    def get_reward(self):
        """Mukofot tizimi - AI ni rag'batlantiruvchi"""
        # To'pni ushlab qolsa - katta mukofot
        if (self.ball.x + self.ball.radius >= self.ai_paddle.x and
            self.ai_paddle.y < self.ball.y < self.ai_paddle.y + self.ai_paddle.height):
            return +1.0
        
        # To'pni o'tkazib yuborsa - katta jarima
        if self.ball.x > SCREEN_WIDTH:
            return -1.0
        
        # To'pni kuzatish uchun mukofot
        # AI raketkasi markazidan to'p gacha bo'lgan masofa
        paddle_center = self.ai_paddle.y + self.ai_paddle.height // 2
        distance = abs(self.ball.y - paddle_center)
        
        # Normalizatsiya: 0 (yaqin) dan 1 (uzoq) gacha
        max_dist = SCREEN_HEIGHT
        normalized = min(distance / max_dist, 1.0)
        
        # Yaqin bo'lsa yaxshi, uzoq bo'sa yomon
        # Kichik qo'shimcha mukofot: to'p AI tomon kelyapsa, markazda turish yaxshi
        if self.ball.speed_x > 0:  # To'p AI tomon kelyapti
            # Markazda turish yaxshi tayyorgarlik uchun
            center_distance = abs(paddle_center - SCREEN_HEIGHT // 2)
            center_normalized = min(center_distance / (SCREEN_HEIGHT // 2), 1.0)
            return (1.0 - normalized) * 0.1 - center_normalized * 0.05
        
        return (1.0 - normalized) * 0.1
        
    def update_opponent(self):
        """Raqib (o'rtacha AI) harakati - AI ga qiyinroq bo'lish uchun"""
        # Raqib to'pni kuzatadi, lekin ba'zan xato qiladi
        paddle_center = self.opponent.y + self.opponent.height // 2
        
        # To'pning kelish yo'nalishini hisoblash
        if self.ball.speed_x < 0:  # To'p raqib tomon kelyapti
            # To'pning keladigan joyini bashorat qilish
            time_to_reach = abs(self.ball.x / self.ball.speed_x) if self.ball.speed_x != 0 else 0
            predicted_y = self.ball.y + self.ball.speed_y * time_to_reach
            
            # Chegaralardan qaytishni hisoblash
            while predicted_y < 0 or predicted_y > SCREEN_HEIGHT:
                if predicted_y < 0:
                    predicted_y = -predicted_y
                elif predicted_y > SCREEN_HEIGHT:
                    predicted_y = 2 * SCREEN_HEIGHT - predicted_y
            
            target_y = predicted_y
        else:
            # To'p uzoqqa ketmoqda, markazga qayt
            target_y = SCREEN_HEIGHT // 2
        
        # Raqib harakati (sekinroq - AI ga imkon berish uchun)
        if random.random() < 0.85:  # 85% daqiqlik
            if target_y < paddle_center - 5:
                self.opponent.move_up()
            elif target_y > paddle_center + 5:
                self.opponent.move_down()

    def check_collisions(self):
        # To'p va raqib raketkasi
        if (self.ball.x - self.ball.radius <= self.opponent.x + self.opponent.width and
            self.opponent.y < self.ball.y < self.opponent.y + self.opponent.height):
            self.ball.speed_x *= -1
            
        # To'p va AI raketkasi
        if (self.ball.x + self.ball.radius >= self.ai_paddle.x and
            self.ai_paddle.y < self.ball.y < self.ai_paddle.y + self.ai_paddle.height):
            self.ball.speed_x *= -1
            
        # Hisob
        if self.ball.x < 0:
            self.ai_paddle.score += 1
            self.ball.reset()
        elif self.ball.x > SCREEN_WIDTH:
            self.opponent.score += 1
            self.ball.reset()
            self.opponent.move_down()

    def train(self):
        """AI ni o'qitish"""
        print("AI o'qitish boshlandi...")
        print(f"Epizodlar: {self.start_episode} dan {self.max_episodes} gacha")
        
        for episode in range(self.start_episode, self.max_episodes):
            self.reset_game()
            state = self.ai_brain.get_state(self.ball, self.ai_paddle)
            total_reward = 0
            
            while not self.done and self.steps < self.max_steps:
                # AI harakatni tanlaydi
                action = self.ai_brain.get_action(state)
                
                # Harakatni bajarish
                if action == 0:  # Yuqoriga
                    self.ai_paddle.move_up()
                elif action == 1:  # Pastga
                    self.ai_paddle.move_down()
                # action == 2: Turish
                
                # O'yin holatini yangilash
                self.update_opponent()
                self.ball.move()
                self.check_collisions()
                self.steps += 1
                
                # Yangi holat va mukofot
                next_state = self.ai_brain.get_state(self.ball, self.ai_paddle)
                reward = self.get_reward()
                total_reward += reward
                
                # Xotiraga saqlash
                self.ai_brain.remember(state, action, reward, next_state, self.done)
                
                state = next_state
                
                # O'yin tugashi
                if self.ai_paddle.score >= 5 or self.opponent.score >= 5:
                    self.done = True
                    
            # Tajribadan o'rganish
            self.ai_brain.replay(64)
            
            # Natijalarni ko'rsatish
            if episode % 100 == 0:
                print(f"Epizod: {episode}/{self.max_episodes}, Mukofot: {total_reward:.2f}, Epsilon: {self.ai_brain.epsilon:.3f}")
                
            # Modelni va checkpoint ni saqlash
            if episode % 1000 == 0:
                self.ai_brain.save("pong_ai_model.pt")
                self.save_checkpoint(episode)
                print(f"Model va checkpoint saqlandi: {episode}")
                
        print("O'qitish tugadi!")
        self.ai_brain.save("pong_ai_model.pt")
        self.save_checkpoint(self.max_episodes)
        print("Yangi model saqlandi! Eski model almashtirildi.")

    def visualize_training(self):
        """O'qitishni ko'rsatish"""
        self.reset_game()
        state = self.ai_brain.get_state(self.ball, self.ai_paddle)
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        
            # AI harakat
            action = self.ai_brain.get_action(state)
            if action == 0:
                self.ai_paddle.move_up()
            elif action == 1:
                self.ai_paddle.move_down()
                
            # O'yin yangilash
            self.update_opponent()
            self.ball.move()
            self.check_collisions()
            
            # Chizish
            self.screen.fill(BLACK)
            pygame.draw.circle(self.screen, WHITE, (int(self.ball.x), int(self.ball.y)), self.ball.radius)
            pygame.draw.rect(self.screen, WHITE, (self.opponent.x, self.opponent.y, self.opponent.width, self.opponent.height))
            pygame.draw.rect(self.screen, GREEN, (self.ai_paddle.x, self.ai_paddle.y, self.ai_paddle.width, self.ai_paddle.height))
            
            # Hisob
            score_text = self.font.render(f"AI: {self.ai_paddle.score} | Raqib: {self.opponent.score}", True, WHITE)
            self.screen.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, 20))
            
            pygame.display.flip()
            self.clock.tick(FPS)
            
        pygame.quit()

if __name__ == "__main__":
    trainer = TrainPong()
    trainer.train()
    print("O'qitish tugadi! Endi 'play_with_ai.bat' ni ishga tushiring.")