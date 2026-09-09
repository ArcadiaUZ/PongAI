"""
AI ni o'qitish - Pong o'yini (PARALLEL)
Bir vaqtda N ta o'yin muhiti ishlaydi, GPU dan samarali foydalanadi
"""

import sys
import random
import time
import numpy as np
import torch
from ai_brain import AIBrain, device

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

NUM_ENVS = 8
BATCH_SIZE = 256
replay_steps = 4          # Har qadamda nechta replay

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

class Env:
    """Bitta Pong muhiti"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.ball = Ball()
        self.ai_paddle = Paddle(SCREEN_WIDTH - 65)
        self.opponent = Paddle(50)
        self.done = False
        self.steps = 0
        self.max_steps = 500
        self.last_action = 2
        self.ai_wins = 0
        self.opp_wins = 0

    def get_state(self):
        return np.array([
            self.ball.x / SCREEN_WIDTH,
            self.ball.y / SCREEN_HEIGHT,
            self.ball.speed_x / 10,
            self.ball.speed_y / 10,
            self.ai_paddle.y / SCREEN_HEIGHT
        ], dtype=np.float32)

    def update_opponent(self):
        """Raqib - kuchli, barqaror (98%)"""
        paddle_center = self.opponent.y + self.opponent.height // 2
        if self.ball.speed_x < 0:
            time_to_reach = abs(self.ball.x / self.ball.speed_x) if self.ball.speed_x != 0 else 0
            predicted_y = self.ball.y + self.ball.speed_y * time_to_reach
            while predicted_y < 0 or predicted_y > SCREEN_HEIGHT:
                if predicted_y < 0:
                    predicted_y = -predicted_y
                elif predicted_y > SCREEN_HEIGHT:
                    predicted_y = 2 * SCREEN_HEIGHT - predicted_y
            target_y = predicted_y
        else:
            target_y = SCREEN_HEIGHT // 2

        if random.random() < 0.98:
            if target_y < paddle_center - 5:
                self.opponent.move_up()
            elif target_y > paddle_center + 5:
                self.opponent.move_down()

    def do_action(self, action):
        """Harakatni bajarib, (reward, next_state, done, info) qaytaradi"""
        if action == 0:
            self.ai_paddle.move_up()
        elif action == 1:
            self.ai_paddle.move_down()

        self.update_opponent()
        self.ball.move()
        self.steps += 1

        reward = self.get_reward(action)

        # To'qnashuvlar va hisob
        if (self.ball.x - self.ball.radius <= self.opponent.x + self.opponent.width and
            self.opponent.y < self.ball.y < self.opponent.y + self.opponent.height):
            self.ball.speed_x *= -1

        if (self.ball.x + self.ball.radius >= self.ai_paddle.x and
            self.ai_paddle.y < self.ball.y < self.ai_paddle.y + self.ai_paddle.height):
            self.ball.speed_x *= -1

        if self.ball.x < 0:
            self.ai_wins += 1
            self.ball.reset()
        elif self.ball.x > SCREEN_WIDTH:
            self.opp_wins += 1
            self.ball.reset()
            self.opponent.move_down()

        self.last_action = action

        done = False
        if self.ai_wins >= 5 or self.opp_wins >= 5:
            done = True
        elif self.steps >= self.max_steps:
            done = True

        return reward, self.get_state(), done

    def get_reward(self, action):
        """Mukofot tizimi"""
        paddle_center = self.ai_paddle.y + self.ai_paddle.height // 2
        distance = abs(self.ball.y - paddle_center)
        aligned = distance < 30

        if (self.ball.x + self.ball.radius >= self.ai_paddle.x and
            self.ai_paddle.y < self.ball.y < self.ai_paddle.y + self.ai_paddle.height):
            return +5.0

        if self.ball.x > SCREEN_WIDTH:
            return -5.0

        normalized = min(distance / SCREEN_HEIGHT, 1.0)
        reward = 0.0

        if self.ball.speed_x > 0:
            reward = (1.0 - normalized) * 0.3
            if aligned and action == 2:
                reward += 0.5
            elif aligned and action != 2:
                reward -= 0.15
        else:
            reward = (1.0 - normalized) * 0.1

        if aligned and action != self.last_action:
            reward -= 0.15

        return reward

class TrainPongParallel:
    def __init__(self):
        self.ai_brain = AIBrain()
        self.model_file = "pong_ai_model.pt"
        self.checkpoint_file = "training_checkpoint.txt"

        self.start_episode = self.load_checkpoint()
        self.max_episodes = self.start_episode + 100000
        self.envs = [Env() for _ in range(NUM_ENVS)]
        self.stats = {'env_wins': 0, 'env_wins_episodes': 0}

    def load_checkpoint(self):
        try:
            with open(self.checkpoint_file, 'r') as f:
                start_episode = int(f.read().strip())
                print(f"[CHECKPOINT] {start_episode} epizoddan davom etiladi...")
                if self.ai_brain.load(self.model_file):
                    return start_episode
        except:
            pass
        print("[CHECKPOINT] Yangi o'qitish boshlanmoqda...")
        return 0

    def save_checkpoint(self, episode):
        with open(self.checkpoint_file, 'w') as f:
            f.write(str(episode))

    def train(self):
        print("PARALLEL AI o'qitish boshlandi!")
        print(f"Parallel muhitlar: {NUM_ENVS}")
        print(f"Epizodlar: {self.start_episode} dan {self.max_episodes} gacha")
        print(f"Batch: {BATCH_SIZE} | Replay/step: {replay_steps}")

        episode = self.start_episode
        total_steps = 0
        episode_rewards = np.zeros(NUM_ENVS)
        episode_count = 0
        start_time = time.time()

        states = np.stack([e.get_state() for e in self.envs])
        done_flags = [False] * NUM_ENVS

        while episode < self.max_episodes:
            states_tensor = torch.FloatTensor(states).to(device)
            actions = self.ai_brain.get_actions_batch(states_tensor)

            next_states = np.zeros_like(states)
            rewards = np.zeros(NUM_ENVS)
            dones = np.zeros(NUM_ENVS, dtype=bool)

            for i in range(NUM_ENVS):
                if not done_flags[i]:
                    r, ns, d = self.envs[i].do_action(int(actions[i]))
                    next_states[i] = ns
                    rewards[i] = r
                    dones[i] = d
                    episode_rewards[i] += r
                    total_steps += 1

            self.ai_brain.remember_batch(states, actions, rewards, next_states, dones)

            # Replay (bir necha marta)
            for _ in range(replay_steps):
                self.ai_brain.replay(BATCH_SIZE)

            states = next_states

            # Tugatilgan muhitlarni qayta boshlash
            for i in range(NUM_ENVS):
                if done_flags[i] or dones[i]:
                    episode += 1
                    episode_count += 1
                    self.ai_brain.decay_epsilon()
                    if self.envs[i].ai_wins > self.envs[i].opp_wins:
                        self.stats['env_wins'] += 1
                    ep_rew = episode_rewards[i]
                    episode_rewards[i] = 0
                    self.envs[i].reset()
                    done_flags[i] = False
                    states[i] = self.envs[i].get_state()

                    if episode % 100 == 0:
                        elapsed = time.time() - start_time
                        speed = total_steps / elapsed if elapsed > 0 else 0
                        print(f"Epizod: {episode}/{self.max_episodes}, "
                              f"O'rt.mukofot: {ep_rew:.2f}, "
                              f"Yutuq: {self.stats['env_wins']}/{episode_count}, "
                              f"Epsilon: {self.ai_brain.epsilon:.3f}, "
                              f"Tezlik: {speed:.0f} qadam/s")

                    if episode % 1000 == 0:
                        self.ai_brain.save(self.model_file)
                        self.save_checkpoint(episode)
                        print(f"Model va checkpoint saqlandi: {episode}")

            done_flags = dones

        print("O'qitish tugadi!")
        self.ai_brain.save(self.model_file)
        self.save_checkpoint(self.max_episodes)
        print("Yangi model saqlandi!")

if __name__ == "__main__":
    trainer = TrainPongParallel()
    trainer.train()