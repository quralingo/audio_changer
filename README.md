# Quran Voice Changer (RVC Pipeline)

This repository contains the code, scripts, and training workflow used to build **multiple Quran recitation voices** (male / female / adult / child) using **Retrieval-based Voice Conversion (RVC)**.

The goal of this project is to **preserve the exact Quranic pronunciation and timing** while converting the *speaker identity* into different voices, enabling richer and more inclusive learning experiences in **Quralingo**.

> ⚠️ This project is Quran-sensitive by design. Accuracy, pronunciation integrity, and ethical use are core principles.

---

## What This Repository Does

- Trains **RVC voice models** for:
    - Male child
    - Female child
    - Female adult
- Uses an existing **male old Quran word-by-word dataset** as the base content
- Converts **77,000+ Quran word audio files** into multiple voices
- Produces consistent, reusable voice models (`.pth`) for production use
- Provides a CLI-friendly pipeline for batch processing

---

## Core Idea

Instead of manipulating pitch, tempo, or acoustic parameters manually (which leads to robotic results), we use **voice conversion**:

- ✅ Linguistic content stays the same
- ✅ Tajwīd and articulation are preserved
- ✅ Speaker identity changes naturally

This approach is far more robust for Quranic content than classic DSP-based voice effects.

---

## Repository Structure
```
.
├── data/
│   ├── README.md
│   ├── old_male_words/        # Quran word-by-word dataset (source voice)
│   ├── male_kid_raw/          # Mp3 files for the data that can be used to create model
│   ├── female_kid_raw/
│   └── female_adult_raw/
│
├── data_processed/
│   ├── male_kid/
│   ├── female_kid/
│   └── female_adult/
│
├── models/
│   ├── old_male.pth
│   ├── male_kid.pth
│   ├── female_kid.pth
│   └── female_adult.pth
│
├── scripts/
│   ├── preprocess_audio.py
│   ├── train_rvc.py
│   └── convert_voice.py
│
├── outputs/
│   ├── male_kid/
│   ├── male_adult/
│   ├── female_kid/
│   └── female_adult/
│
├── requirements.txt
└── README.md
```
---

## Datasets Used

### 1. Base Quran Dataset (Male Old Voice)
- **Source**: Hugging Face
- **Dataset**: Quran word-by-word recitation
- **Size**: ~77,000 audio files
- **Link**:  
  https://huggingface.co/datasets/Buraaq/quran-md-words

This dataset is used as the **content reference** for all voice conversions.
You can download it directly via snapshot_download:

```python
import os
from huggingface_hub import snapshot_download

dataset_path = "<path-to-save-dataset>"

snapshot_download(
    repo_id="Buraaq/quran-audio-text-dataset",
    repo_type="dataset",
    local_dir=dataset_path
)
```

---

### 2. Target Voice Sources (Training Data)

These sources are used **only to learn speaker identity**, not Quran content:

- **Male Child**  
  https://youtube.com/@abdullahshaab1

- **Female Child**  
  https://youtube.com/@mennatallahramadan1

- **Female Adult**  
  https://youtube.com/@maimohamed.official1

All audio is:
- downloaded
- cleaned
- segmented
- converted to mono WAV
- normalized

---

## Setup

### Requirements

- Python 3.9+
- CUDA-enabled GPU (recommended)
- FFmpeg
- PyTorch (CUDA build)

Install dependencies:

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

```bash
pip install -r requirements.txt
```
---

### Audio Preprocessing

Convert and normalize audio:
```bash
python scripts/preprocess_audio.py \
  --input data/male_kid_raw \
  --output data_processed/male_kid
```
This step:
- converts to WAV (48kHz, mono)
- removes silence
- splits long clips
- normalizes loudness

Repeat for each target voice.

---

### Training RVC Models

Train one model per target voice:
```bash
python scripts/train_rvc.py \
  --data data_processed/male_kid \
  --output models/male_kid
```
Recommended training settings:
- Epochs: 200–300
- Batch size: 4–8 (GPU dependent)
- Feature extraction: enabled

Repeat for:
- female_kid
- female_adult

---

### Voice Conversion (Inference)

Convert any Quran audio file (or batch):
```bash
python scripts/convert_voice.py \
  --input input.wav \
  --model models/female_kid.pth \
  --output output_female_kid.wav
```
For batch Quran word conversion:
```bash
python scripts/convert_voice.py \
  --input_dir data/old_male_words \
  --model models/male_kid.pth \
  --output_dir outputs/male_kid
```

---

### Quality Control (Important)

Before production use:
- Compare durations (input vs output)
- Listen for dropped consonants
- Spot-check Tajwīd
- Reject any sample with artifacts

Automated checks help, but human review is mandatory for Quranic audio.

---

### Ethical & Religious Considerations

- No impersonation of real individuals without permission
- No alteration of Quranic wording
- No synthetic additions or removals
- Voices are used only for educational purposes

This repository does not replace traditional recitation — it complements learning.

--- 
 
### Relation to Quralingo

This work supports Quralingo’s mission to:
- provide multiple pronunciation voices
- support kids and adults
- adapt learning to age and preference
- keep Quran learning accessible and engaging

---

### Contributing

Contributions are welcome in:
- audio preprocessing improvements
- quality checks
- training efficiency
- documentation

Please respect the Quran-sensitive nature of this project.

---

### License

This repository is released under an open-source license for educational and research purposes.
Dataset licenses must be respected individually.

---

### Acknowledgment

Thanks to all those who supported with ideas and contribution so far and who will still contribute to the repository.