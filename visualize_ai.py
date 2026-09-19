"""
AI Neyron Tarmoq Vizualizatsiyasi
AI nima ish qilayotganini real vaqtda ko'rish
"""

import pygame
import sys
import numpy as np
import torch
from ai_brain import AIBrain, device

# Pygame ni ishga tushirish
pygame.init()

# Oynaning o'lchamlari
WIDTH = 500
HEIGHT = 600
FPS = 60

# Ranglar
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)

class VisualizeAI:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("AI Neyron Tarmoq Vizualizatsiyasi")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        
        # AI Brain yuklash
        self.ai_brain = AIBrain()
        try:
            self.ai_brain.load("pong_ai_model.pt")
            self.model_loaded = True
        except:
            self.model_loaded = False
            
        # Vizualizatsiya uchun ma'lumotlar
        self.input_names = ["To'p X", "To'p Y", "Tezlik X", "Tezlik Y", "AI Y"]
        self.output_names = ["Yuqoriga", "Pastga", "Turish"]
        self.q_values = [0, 0, 0]
        self.current_action = 0
        self.current_state = [0.5, 0.5, 0, 0, 0.5]
        
    def draw_neural_network(self):
        """Neyron tarmoq sxemasini chizish"""
        # Input qatlam (5 ta neyron)
        input_positions = []
        for i in range(5):
            x = 80
            y = 100 + i * 80
            input_positions.append((x, y))
            
        # Hidden qatlam (8 ta neyron - vizual uchun)
        hidden_positions = []
        for i in range(8):
            x = 250
            y = 80 + i * 55
            hidden_positions.append((x, y))
            
        # Output qatlam (3 ta neyron)
        output_positions = []
        for i in range(3):
            x = 420
            y = 180 + i * 100
            output_positions.append((x, y))
        
        # Input-hidden bog'lanishlar
        for i, pos in enumerate(input_positions):
            for j, h_pos in enumerate(hidden_positions):
                intensity = max(0, min(255, int(128 + 127 * np.sin(i * j + pygame.time.get_ticks() * 0.001))))
                color = (intensity, intensity, intensity)
                pygame.draw.line(self.screen, color, pos, h_pos, 1)
                
        # Hidden-output bog'lanishlar
        for i, h_pos in enumerate(hidden_positions):
            for j, o_pos in enumerate(output_positions):
                intensity = max(0, min(255, int(128 + 127 * np.sin(i * j + pygame.time.get_ticks() * 0.001))))
                color = (intensity, intensity, intensity)
                pygame.draw.line(self.screen, color, h_pos, o_pos, 1)
        
        # Input neyronlari
        for i, pos in enumerate(input_positions):
            activation = abs(self.current_state[i])
            color = (int(255 * activation), int(255 * (1 - activation)), 0)
            pygame.draw.circle(self.screen, color, pos, 15)
            pygame.draw.circle(self.screen, WHITE, pos, 15, 2)
            
            name = self.small_font.render(self.input_names[i], True, WHITE)
            self.screen.blit(name, (pos[0] - 60, pos[1] - 8))
            
            value = self.small_font.render(f"{self.current_state[i]:.2f}", True, GRAY)
            self.screen.blit(value, (pos[0] - 60, pos[1] + 10))
            
        # Hidden neyronlari
        for i, pos in enumerate(hidden_positions):
            activation = 0.5 + 0.5 * np.sin(pygame.time.get_ticks() * 0.003 + i)
            color = (int(100 + 155 * activation), int(100 + 155 * activation), 255)
            pygame.draw.circle(self.screen, color, pos, 12)
            pygame.draw.circle(self.screen, WHITE, pos, 12, 1)
            
        # Output neyronlari
        for i, pos in enumerate(output_positions):
            q_val = self.q_values[i]
            if i == self.current_action:
                color = GREEN
            else:
                intensity = max(0, min(255, int(128 + 127 * q_val)))
                color = (intensity, intensity, intensity)
                
            pygame.draw.circle(self.screen, color, pos, 20)
            pygame.draw.circle(self.screen, WHITE, pos, 20, 2)
            
    
    def draw_info(self):
        """Ma'lumotlarni chizish"""
        title = self.font.render("AI Neyron Tarmoq", True, WHITE)
        self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 20))
        
        if self.model_loaded:
            status = self.font.render("Model yuklandi", True, GREEN)
        else:
            status = self.font.render("Model yuklanmadi", True, RED)
        self.screen.blit(status, (20, 50))
        
        action_text = self.font.render(f"Harakat: {self.output_names[self.current_action]}", True, GREEN)
        self.screen.blit(action_text, (20, HEIGHT - 100))
        
        eps_text = self.small_font.render(f"Epsilon: {self.ai_brain.epsilon:.4f}", True, GRAY)
        self.screen.blit(eps_text, (20, HEIGHT - 70))
        
        gpu_text = self.small_font.render(f"GPU: {torch.cuda.get_device_name(0)}", True, GRAY)
        self.screen.blit(gpu_text, (20, HEIGHT - 40))
        
    def draw_q_comparison(self):
        """Q qiymatlarini taqqoslash diagrammasi"""
        bar_x = 20
        bar_width = 100
        bar_height = 20
        start_y = HEIGHT - 200
        
        title = self.small_font.render("Q Qiymatlari:", True, WHITE)
        self.screen.blit(title, (bar_x, start_y - 25))
        
        for i, (name, q_val) in enumerate(zip(self.output_names, self.q_values)):
            y = start_y + i * 30
            
            pygame.draw.rect(self.screen, DARK_GRAY, (bar_x, y, bar_width, bar_height))
            
            normalized_q = max(0, min(1, (q_val + 1) / 2))
            fill_width = int(bar_width * normalized_q)
            
            if i == self.current_action:
                color = GREEN
            else:
                color = BLUE
                
            pygame.draw.rect(self.screen, color, (bar_x, y, fill_width, bar_height))
            pygame.draw.rect(self.screen, WHITE, (bar_x, y, bar_width, bar_height), 1)
            
            text = self.small_font.render(f"{name}: {q_val:.2f}", True, WHITE)
            self.screen.blit(text, (bar_x + bar_width + 10, y + 2))
    
    def update(self):
        """AI holatini yangilash"""
        if not hasattr(self, 'sim_timer'):
            self.sim_timer = 0
        self.sim_timer += 1
        
        if self.sim_timer % 30 == 0:
            ball_x = np.random.random()
            ball_y = np.random.random()
            speed_x = np.random.choice([-0.5, 0.5])
            speed_y = np.random.uniform(-0.5, 0.5)
            ai_y = np.random.random()
            
            self.current_state = [ball_x, ball_y, speed_x, speed_y, ai_y]
            
            state_tensor = torch.FloatTensor(self.current_state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_vals = self.ai_brain.brain(state_tensor).cpu().numpy()[0]
                self.q_values = q_vals.tolist()
                self.current_action = int(np.argmax(q_vals))
    
    def run(self):
        """Asosiy tsikl"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            self.screen.fill(BLACK)
            self.update()
            self.draw_neural_network()
            self.draw_info()
            self.draw_q_comparison()
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    viz = VisualizeAI()
    viz.run()