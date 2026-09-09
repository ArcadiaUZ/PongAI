# Pong AI

Reinforcement Learning (Double DQN) asosida o'rganadigan Pong AI. PyTorch GPU/CPU, parallel o'qitish, real-vaqt neyron tarmoq vizualizatsiyasi.

## Xususiyatlar

- **Parallel o'qitish** - 8 ta o'yin muhiti bir vaqtda ishlaydi (GPU samaradorlikni oshirish)
- **Double DQN** - Target network bilan barqaror o'rganish
- **128 Hidden Layer** - Katta neyron tarmoq arxitekturasi
- **AI vs AI** - Ikkita AI o'zi bilan o'ynaydi, ikkala neyron tarmoq vizualizatsiyasi ko'rinadi
- **Real-vaqt vizualizatsiya** - Q qiymatlar, harakatlar, neyron tarmoq faoliyati
- **Checkpoint tizimi** - O'qitishni istalgan vaqtda davom ettirish

## O'rnatish

```bash
pip install pygame numpy torch
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
8 ta muhit parallel, ~850 qadam/s, 100k epizod.

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
| `train_ai.py` | Parallel o'qitish (8 muhit, GPU batch) |
| `play_pong.py` | Odam vs AI + neyron tarmoq vizualizatsiyasi |
| `ai_vs_ai.py` | AI vs AI, ikkala tarmoq vizualizatsiyasi |
| `pong_game.py` | Oddiy Pong o'yini (AI yo'q) |
| `visualize_ai.py` | Alohida neyron tarmoq vizualizatsiyasi |
| `pong_ai_model.pt` | O'qitilgan AI modeli (PyTorch) |
| `training_checkpoint.txt` | Oxirgi epizod raqami |
| `start_training.bat` | O'qitishni ishga tushirish |
| `play_with_ai.bat` | Odam vs AI o'yini |
| `ai_bilan_ai_pong.bat` | AI vs AI o'yini |

## AI Parametrlari

| Parametr | Qiymat | Tavsif |
|----------|--------|--------|
| Input | 5 | To'p X, Y, Tezlik X, Y, AI Y |
| Hidden | 128 | Neyron tarmoq kengligi |
| Output | 3 | Yuqoriga, Pastga, Turish |
| Gamma | 0.95 | Discount factor |
| Epsilon | 0.01 - 0.1 | Eksploratsiya (kamayib boradi) |
| Batch | 256 | O'rganish batch hajmi |
| Replay | 4x/step | Har qadamda 4 marta replay |
| Memory | 200,000 | Xotira hajmi |
| Raqib | 98% | O'qitishdagi raqib aniqligi |
| Muhitlar | 8 | Parallel o'yinlar soni |

## O'qitish tizimi

```
State: [ball.x/800, ball.y/600, speed_x/10, speed_y/10, paddle.y/600]
       ↓
128 Hidden (Tanh) → 3 Output (Q-values)
       ↓
Epsilon-greedy → Action (Up/Down/Stay)
       ↓
Reward: +5 catch, -5 miss, proximity +0.3, alignment +0.5, jitter -0.15
       ↓
Double DQN replay (batch 256, 4x per step)
```

## Qanday o'rganadi?

1. **State** - To'p va raketka pozitsiyasi/tezligi (5 ta kirish)
2. **Q-Learning** - Har bir holat-harakat uchun Q qiymati
3. **Double DQN** - Ikki tarmoq: biri qaror qabul qiladi, ikkinchisi baholaydi
4. **Mukofot** - To'pni ushlash, tekis o'ynash, qaltiramaslik rag'batlantiriladi

## Natijalar

O'qitish davomida kuzatiladigan ko'rsatkichlar:
- Mukofot oshishi kerak (0 → 100+ → 1000+)
- Epsilon kamayishi (0.1 → 0.01)
- Yutuq foizi oshishi

## Demo

**Odam vs AI:**
[demo1.mp4](content/demo1.mp4)

**AI vs AI:**
[demo2.mp4](content/demo2.mp4)

## Litsenziya

MIT
