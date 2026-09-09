"""
AI Brain - Neyron tarmoq (PyTorch)
Pong o'yinini o'rganadigan AI
GPU (CUDA) yoki CPU da ishlaydi
"""

import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# Qurilmani tekshirish
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] Qurilma: {device}")
if torch.cuda.is_available():
    print(f"[GPU] {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
else:
    print("[CPU] GPU topilmadi, CPU ishlatiladi")

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Og'irliklarni boshlang'ich qiymat
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class AIBrain:
    def __init__(self, hidden_size=128):
        # Kirish: to'p pozitsiyasi, tezligi, AI raketkasi pozitsiyasi
        # Chiqish: harakat (yuqoriga, pastga, turish)
        self.hidden_size = hidden_size
        self.brain = NeuralNetwork(input_size=5, hidden_size=hidden_size, output_size=3).to(device)
        self.target_brain = NeuralNetwork(input_size=5, hidden_size=hidden_size, output_size=3).to(device)
        self.target_brain.load_state_dict(self.brain.state_dict())
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=200000)
        self.epsilon = 0.1  # Eksploratsiya (0.1 dan boshlanadi)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99997
        self.gamma = 0.95  # Discount factor
        self.target_update = 0
        
    def get_state(self, ball, ai_paddle):
        """O'yin holatini olish - PyTorch tensor"""
        state = np.array([
            ball.x / 800,  # To'p x pozitsiyasi (normalizatsiya)
            ball.y / 600,  # To'p y pozitsiyasi
            ball.speed_x / 10,  # To'p x tezligi
            ball.speed_y / 10,  # To'p y tezligi
            ai_paddle.y / 600  # AI raketkasi pozitsiyasi
        ], dtype=np.float32)
        return torch.FloatTensor(state).unsqueeze(0).to(device)
    
    def get_action(self, state):
        """Harakatni tanlash (bitta muhit) - int qaytaradi"""
        if random.random() < self.epsilon:
            return random.randint(0, 2)  # Tasodifiy harakat
        with torch.no_grad():
            q_values = self.brain(state)
            return q_values.argmax().item()
    
    def _explore_or_exploit(self, q_values):
        """Epsilon-greedy: batchni teskarisiz boshqaradi"""
        actions = q_values.argmax(dim=-1).cpu().numpy()
        # Tasodifiy tanlab olish (epsilon bo'yicha)
        rand = np.random.random(actions.shape[0])
        replace = rand < self.epsilon
        if replace.any():
            actions[replace] = np.random.randint(0, q_values.shape[-1], size=replace.sum())
        return actions
    
    def get_actions_batch(self, states):
        """Parallel muhitlar uchun harakatlarni olish (batch)"""
        with torch.no_grad():
            q_values = self.brain(states)
            return self._explore_or_exploit(q_values)
    
    def remember(self, state, action, reward, next_state, done):
        """Xotiraga saqlash (bitta)"""
        self.memory.append((state, action, reward, next_state, done))
    
    def remember_batch(self, states, actions, rewards, next_states, dones):
        """Parallel muhitlar tajribasini xotiraga saqlash"""
        states_t = torch.FloatTensor(states).to(device)
        next_states_t = torch.FloatTensor(next_states).to(device)
        for i in range(len(actions)):
            self.memory.append((states_t[i].unsqueeze(0), actions[i], rewards[i],
                                next_states_t[i].unsqueeze(0), bool(dones[i])))
    
    def replay(self, batch_size=128):
        """Tajribadan o'rganish - GPU da (Double DQN)"""
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.BoolTensor(dones).to(device)
        
        # Q qiymatlarini hisoblash
        current_q = self.brain(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            # Double DQN: brain tanlaydi, target_brain baholaydi
            next_actions = self.brain(next_states).argmax(1)
            next_q = self.target_brain(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            next_q[dones] = 0.0
            target_q = rewards + self.gamma * next_q
        
        # Loss va orqaga tarqalish
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Target network ni yangilash (har 100 qadamda)
        self.target_update += 1
        if self.target_update % 100 == 0:
            self.target_brain.load_state_dict(self.brain.state_dict())
    
    def decay_epsilon(self):
        """Epsilonni har bir epizodda kamaytirish (parallel o'qitish uchun)"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename):
        """Modelni saqlash"""
        torch.save({
            'brain': self.brain.state_dict(),
            'target_brain': self.target_brain.state_dict(),
            'epsilon': self.epsilon,
        }, filename)
        print(f"[GPU] PyTorch model saqlandi: {filename}")
    
    def load(self, filename):
        """Modelni yuklash"""
        try:
            checkpoint = torch.load(filename, map_location=device)
            self.brain.load_state_dict(checkpoint['brain'])
            self.target_brain.load_state_dict(checkpoint['target_brain'])
            self.epsilon = checkpoint['epsilon']
            print(f"[GPU] PyTorch model yuklandi: {filename} (epsilon={self.epsilon:.3f})")
            return True
        except Exception as e:
            print(f"[GPU] PyTorch model topilmadi: {filename} ({e})")
            return False