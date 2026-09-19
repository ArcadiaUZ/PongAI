# Pong AI

Reinforcement Learning (Double DQN) asosida o'rganadigan Pong AI. PyTorch GPU/CPU, parallel o'qitish, real-vaqt neyron tarmoq vizualizatsiyasi.

## Xususiyatlar

- **Parallel o'qitish** - 32 ta o'yin muhiti bir vaqtda ishlaydi (GPU samaradorlikni oshirish)
- **Double DQN** - Target network bilan barqaror o'rganish
- **512 Hidden Layer** - 3 qatlamli neyron tarmoq arxitekturasi (512 → 256 → 128)
- **10 ta Input Feature** - To'p pozitsiyasi, tezligi, raqib holati, masofa va boshqalar
- **AI vs AI** - Ikkita AI o'zi bilan o'ynaydi, ikkala neyron tarmoq vizualizatsiyasi ko'rinadi
- **Real-vaqt vizualizatsiya** - Q qiymatlar, harakatlar, neyron tarmoq faoliyati
- **Checkpoint tizimi** - O'qitishni istalgan vaqtda davom ettirish
- **GPU/CPU** - Avtomatik CUDA aniqlash

## O'rnatish

```bash
pip install -r requirements.txt
```

GPU uchun (CUDA):
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Ishga tushirish

### O'qitish (parallel, tez)
```
start_training.bat
```
32 ta muhit parallel, ~500,000 epizod, GPU batch 1024.

### AI bilan o'ynash (odam vs AI)
```
play_with_ai.bat
```

### AI vs AI (yangi AI vs yangi AI)
```
ai_bilan_ai_pong.bat
```
100 raund, ikkala tomonda neyron tarmoq vizualizatsiyasi.

### Neyron tarmoq vizualizatsiyasi
```
visualize_ai.bat
```

## Fayllar

| Fayl | Tavsif |
|------|--------|
| `ai_brain.py` | Neyron tarmoq (PyTorch), DQN, xotira |
| `train_ai.py` | Parallel o'qitish (32 muhit, GPU batch) |
| `play_pong.py` | Odam vs AI + neyron tarmoq vizualizatsiyasi |
| `ai_vs_ai.py` | AI vs AI, ikkala tarmoq vizualizatsiyasi |
| `pong_game.py` | Oddiy Pong o'yini (AI yo'q) |
| `visualize_ai.py` | Alohida neyron tarmoq vizualizatsiyasi |
| `pong_ai_model.pt` | O'qitilgan AI modeli (PyTorch) |
| `enemy_pong_ai/` | Eski AI modeli (AI vs AI uchun) |
| `training_checkpoint.txt` | Oxirgi epizod raqami |
| `start_training.bat` | O'qitishni ishga tushirish |
| `play_with_ai.bat` | Odam vs AI o'yini |
| `ai_bilan_ai_pong.bat` | AI vs AI o'yini |
| `visualize_ai.bat` | Neyron tarmoq vizualizatsiyasi |

## AI Parametrlari

| Parametr | Qiymat | Tavsif |
|----------|--------|--------|
| Input | 10 | To'p X/Y, Tezlik X/Y, AI Y, Raqib Y, Masofa, Vert Dif, Kelyapti, Last Act |
| Hidden | 512 → 256 → 128 | 3 qatlamli neyron tarmoq (LayerNorm + Tanh + Dropout) |
| Output | 3 | Yuqoriga, Pastga, Turish |
| Gamma | 0.99 | Discount factor |
| Epsilon | 0.001 - 0.5 | Eksploratsiya (kamayib boradi, decay=0.99998) |
| Batch | 1024 | O'rganish batch hajmi |
| Replay | 16x/step | Har qadamda 16 marta replay |
| Memory | 1,000,000 | Xotira hajmi |
| Muhitlar | 32 | Parallel o'yinlar soni |
| Optimizer | Adam | LR=0.0003 |
| Loss | SmoothL1 | Huber loss |

## O'qitish tizimi

```
State: [ball.x/800, ball.y/600, speed_x/10, speed_y/10, ai.y/600,
        opponent.y/600, (ball.x-ai.x)/800, (ball.y/600)-paddle_center,
        ball_coming, last_action]
       ↓
512 → 256 → 128 (LayerNorm + Tanh + Dropout 0.1)
       ↓
Epsilon-greedy → Action (Up/Down/Stay)
       ↓
Reward: +10 goal, +7 return hit, -5 miss, +progress, +alignment
       ↓
Double DQN replay (batch 1024, 16x per step)
```

## Qanday o'rganadi?

1. **State** - To'p va raketka pozitsiyasi/tezligi + raqib holati (10 ta kirish)
2. **Q-Learning** - Har bir holat-harakat uchun Q qiymati
3. **Double DQN** - Ikki tarmoq: biri qaror qabul qiladi, ikkinchisi baholaydi
4. **Mukofot** - To'pni ushlash, tekis o'ynash, qaltiramaslik rag'batlantiriladi
5. **Target Network** - Har 500 qadamda yangilanadi

## Natijalar

O'qitish davomida kuzatiladigan ko'rsatkichlar:
- Mukofot oshishi kerak (0 → 100+ → 1000+)
- Epsilon kamayishi (0.5 → 0.001)
- Yutuq foizi oshishi

## Demo

**Odam vs AI:**
[demo1.mp4](content/demo1.mp4)

**AI vs AI:**
[demo2.mp4](content/demo2.mp4)

## Litsenziya

PolyForm Noncommercial 1.0.0 — [LICENSE](LICENSE).

Notijorat maqsadda foydalanish, o'zgartirish va ulashish mumkin. Sotish taqiqlanadi.
