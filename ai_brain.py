"""
AI Brain - Neyron tarmoq (PyTorch) — 4x Kuchaytirilgan
Pong o'yinini mukammal o'rganadigan AI
GPU (CUDA) yoki CPU da ishlaydi
"""

import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] Qurilma: {device}")
if torch.cuda.is_available():
    print(f"[GPU] {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
else:
    print("[CPU] GPU topilmadi, CPU ishlatiladi")

INPUT_SIZE = 10
OUTPUT_SIZE = 3

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, output_size),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class AIBrain:
    def __init__(self, hidden_size=512):
        self.hidden_size = hidden_size
        self.brain = NeuralNetwork(input_size=INPUT_SIZE, hidden_size=hidden_size, output_size=OUTPUT_SIZE).to(device)
        self.target_brain = NeuralNetwork(input_size=INPUT_SIZE, hidden_size=hidden_size, output_size=OUTPUT_SIZE).to(device)
        self.target_brain.load_state_dict(self.brain.state_dict())
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.0003)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = deque(maxlen=1_000_000)
        self.epsilon = 0.5
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.99998
        self.gamma = 0.99
        self.target_update = 0

    def get_state(self, ball, ai_paddle, opponent_paddle=None):
        opponent_y = 0.5 if opponent_paddle is None else opponent_paddle.y / 600.0
        paddle_center = (ai_paddle.y + ai_paddle.height / 2) / 600.0
        state = np.array([
            ball.x / 800,
            ball.y / 600,
            ball.speed_x / 10,
            ball.speed_y / 10,
            ai_paddle.y / 600,
            opponent_y,
            (ball.x - ai_paddle.x) / 800,
            (ball.y / 600) - paddle_center,
            1.0 if ball.speed_x > 0 else 0.0,
            self._last_action_norm,
        ], dtype=np.float32)
        return torch.FloatTensor(state).unsqueeze(0).to(device)

    @property
    def _last_action_norm(self):
        return getattr(self, '_norm_val', 0.0)

    @_last_action_norm.setter
    def _last_action_norm(self, v):
        self._norm_val = v

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, OUTPUT_SIZE - 1)
        with torch.no_grad():
            q_values = self.brain(state)
            return q_values.argmax().item()

    def _explore_or_exploit(self, q_values):
        actions = q_values.argmax(dim=-1).cpu().numpy()
        rand = np.random.random(actions.shape[0])
        replace = rand < self.epsilon
        if replace.any():
            actions[replace] = np.random.randint(0, q_values.shape[-1], size=replace.sum())
        return actions

    def get_actions_batch(self, states):
        with torch.no_grad():
            q_values = self.brain(states)
            return self._explore_or_exploit(q_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def remember_batch(self, states, actions, rewards, next_states, dones):
        states_t = torch.FloatTensor(states).to(device)
        next_states_t = torch.FloatTensor(next_states).to(device)
        for i in range(len(actions)):
            self.memory.append((states_t[i].unsqueeze(0), actions[i], rewards[i],
                                next_states_t[i].unsqueeze(0), bool(dones[i])))

    def replay(self, batch_size=1024):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.BoolTensor(dones).to(device)

        current_q = self.brain(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.brain(next_states).argmax(1)
            next_q = self.target_brain(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            next_q[dones] = 0.0
            target_q = rewards + self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0)
        self.optimizer.step()

        self.target_update += 1
        if self.target_update % 500 == 0:
            self.target_brain.load_state_dict(self.brain.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        torch.save({
            'brain': self.brain.state_dict(),
            'target_brain': self.target_brain.state_dict(),
            'epsilon': self.epsilon,
        }, filename)
        print(f"[SAVE] Model saqlandi: {filename}")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename, map_location=device)
            self.brain.load_state_dict(checkpoint['brain'])
            self.target_brain.load_state_dict(checkpoint['target_brain'])
            self.epsilon = checkpoint['epsilon']
            print(f"[LOAD] Model yuklandi: {filename} (epsilon={self.epsilon:.4f})")
            return True
        except Exception as e:
            print(f"[!] Model topilmadi: {filename} ({e})")
            return False
