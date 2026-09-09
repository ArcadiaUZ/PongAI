# Pong AI 🎮

AI bilan Pong o'yini. Neyron tarmoq (PyTorch) va Reinforcement Learning (Double DQN) asosida. **GPU** yoki **CPU** da ishlaydi.

## 🚀 Xususiyatlar

- **GPU/CPU** - NVIDIA GPU yoki CPU da ishlash
- **Double DQN** - Barqaror AI o'rganish
- **Checkpoint tizimi** - O'qitishni davom ettirish
- **Real-vaqt vizualizatsiyasi** - AI neyron tarmogini ko'rish
- **PyTorch** formatda model saqlash

## 📦 O'rnatish

### 1. Python paketlari:
```bash
pip install pygame numpy torch
```

### 2. GPU uchun (CUDA):
```bash
# PyTorch GPU versiyasini o'rnatish (agar GPU bor bolsa)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 🎮 Ishga tushirish

### CPU da o'qitish:
```bash
python train_ai.py
```
- 10,000 epizod (~15-20 daqiqa CPU da)
- Avtomatik checkpoint saqlanadi

### GPU da o'qitish (tezroq):
```bash
python train_ai.py
```
- 10,000 epizod GPU da (~3-5 daqiqa)
- RTX 3060+ dan foydalanadi

### AI bilan o'ynash:
```bash
python play_pong.py
```

## 📁 Fayllar

| Fayl | Tavsif |
|------|--------|
| `pong_game.py` | Oddiy Pong o'yini |
| `ai_brain.py` | Neyron tarmoq (PyTorch GPU/CPU) |
| `train_ai.py` | AI ni o'qitish |
| `play_pong.py` | AI bilan o'yin + vizualizatsiya |
| `visualize_ai.py` | Alohida AI vizualizatsiya oynasi |
| `pong_ai_model.pt` | O'qitilgan AI modeli |
| `training_checkpoint.txt` | Oxirgi epizod raqami |

## 🎯 Boshqaruv

- **W** / **↑** - Yuqoriga harakat
- **S** / **↓** - Pastga harakat
- **SPACE** - Qayta boshlash
- **ESC** - Chiqish

## 🔧 AI Parametrlari

| Parametr | Qiymat | Tavsif |
|----------|--------|--------|
| Input | 5 | To'p X, Y, Tezlik X, Y, AI Y |
| Hidden | 32 | Neyronlar soni |
| Output | 3 | Yuqoriga, Pastga, Turish |
| Gamma | 0.95 | Discount factor |
| Epsilon | 0.01-0.1 | Eksploratsiya |
| Batch | 128 | O'rganish batch |

## 🔄 Checkpoint tiziti

- `training_checkpoint.txt` - saqlangan epizod raqami
- `pong_ai_model.pt` - AI modeli
- Har 1000 epizvda saqlanadi
- Yangi o'qitgacha avtomatik davom etadi

## 🧠 AI qanday ishlaydi?

1. **Holat** - To'p va AI raketkasining pozitsiyasi
2. **Q-Learning** - Har bir harakat uchun Q qiymatini hisoblaydi
3. **Double DQN** - Target network orqali barqaror o'rganish
4. **Mukofot** - To'pni ushlash (+1), o'tkazish (-1)

## 📊 Natijalar

O'qitish davomida kuzating:
- **Mukofotlar** oshishi kerak (+100+ yaxshi)
- **Epsilon** 0.1 dan 0.01 gacha kamayishi
- **AI** 70%+ istalaganida yutishi kerak

## 📄 Litsenziya

Ushbu loyiha MIT litsenziyasi ostida tarqatiladi.