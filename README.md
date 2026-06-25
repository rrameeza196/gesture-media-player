# 🎮 Touchless Gesture-Controlled Media Player

An HCI project that lets you control a media player using **hand gestures** captured from a webcam — no mouse, no keyboard, no touch. Built with **MediaPipe** hand-landmark detection and classical ML classifiers (k-NN, SVM, Random Forest).

> **Course:** SE305T / MD445T – Human Computer Interaction (Spring '26)
> **Institution:** Information Technology University (ITU), Lahore
> **Assignment:** Code Project Submission 3 (CP3)
> **Author:** [@rrameeza196](https://github.com/rrameeza196)

---

## 📺 Demo Video

https://github.com/rrameeza196/REPO_NAME/raw/main/media/recording.mp4

> ⚠️ Replace `REPO_NAME` above with your actual repository name once created — GitHub will then render this as a playable video directly on this page. Until pushed, you can also just open [`media/recording.mp4`](media/recording.mp4) directly.

A short screen recording of the system detecting live hand gestures through the webcam and triggering the corresponding media commands (play/pause, volume, track skip, mute) in real time.

---

## 📖 Project Overview

This project replaces physical input (mouse, keyboard, remote) with **touchless hand gestures** to control a media player — a classic HCI problem of designing a natural, low-friction interaction model. A webcam feed is passed through **MediaPipe's Hand Landmarker**, which detects 21 keypoints on the hand. These keypoints are normalized into a 63-dimensional feature vector and classified into one of 7 gestures using a trained machine learning model. Each recognized gesture is mapped to a keyboard shortcut (via `pyautogui`) that controls any media player with focus (Spotify, VLC, YouTube, etc.).

---

## ✋ Gesture → Command Mapping

| ID | Gesture | Command | Key Triggered |
|----|---------|---------|----------------|
| 0 | 🖐 Open Palm | Play / Pause | `space` |
| 1 | ✊ Fist | Stop | `s` |
| 2 | 👍 Thumbs Up | Volume Up | `volumeup` |
| 3 | 👎 Thumbs Down | Volume Down | `volumedown` |
| 4 | 👈 Index Left | Previous Track | `left` |
| 5 | 👉 Index Right | Next Track | `right` |
| 6 | ✌️ Victory Sign | Mute / Unmute | `volumemute` |

---

## 🖼️ Figures & Detailed Explanations

### Fig. 1 — System Architecture

![System Architecture](results/fig1_system_architecture.png)

This diagram shows the end-to-end pipeline of the system, from raw input to final action. The webcam captures a live video frame, which MediaPipe processes to detect the hand region. From the detected hand, 21 landmark points are extracted. These points are normalized (translation- and scale-invariant) into a 63-dimensional feature vector. This vector is passed into the trained classifier (k-NN/SVM/Random Forest), which outputs a gesture label. That label is mapped to a specific media command, which is finally shown on the UI overlay and executed as a keyboard action on the system.

### Fig. 2 — MediaPipe Hand Landmarks & Feature Extraction

![Hand Landmarks](results/fig2_hand_landmarks.png)

The left panel visualizes MediaPipe's 21-point hand skeleton — each finger (thumb, index, middle, ring, pinky) is color-coded, and every point has an (x, y, z) coordinate, giving a 63-dimensional raw vector per hand. The right panel breaks down the 7-step feature extraction pipeline: capturing the frame, detecting the hand, extracting the 21 landmarks, anchoring all coordinates relative to the wrist (landmark 0) for translation invariance, normalizing by the maximum coordinate value for scale invariance (so the same gesture is recognized regardless of hand size or distance from the camera), flattening into the final 63-dim feature vector, and finally classifying it.

### Fig. 3 — Dataset Class Distribution

![Class Distribution](results/fig3_class_distribution.png)

The dataset used for training is perfectly balanced: 7 gesture classes with exactly 3,000 samples each (21,000 samples total), shown here as both a bar chart and a pie chart confirming an even 14.3% split per class. A balanced dataset like this prevents the model from being biased toward any one gesture and makes accuracy a reliable evaluation metric.

### Fig. 4 — Confusion Matrix & Per-Class Metrics (k-NN)

![Confusion Matrix](results/fig4_confusion_metrics.png)

The confusion matrix (left) shows how often the k-NN classifier's predictions matched the true gesture labels on the test set. Most gestures (Open Palm, Thumbs Down, Index Left, Index Right, Victory Sign) were classified with zero errors. The main confusion happens between **Fist** and **Thumbs Up** (19 and 36 misclassifications respectively) — gestures that involve closely similar finger-curl patterns. The right panel breaks this down into per-class Precision/Recall/F1, confirming that even the weakest classes (Fist, Thumbs Up) still score above 94%.

### Fig. 5 — Classifier Performance Comparison

![Model Comparison](results/fig5_model_comparison.png)

This compares all three trained models side-by-side across three dimensions: test accuracy, macro F1-score, and inference latency. **SVM (RBF)** achieves the best accuracy and F1 (99.26%), narrowly beating Random Forest (99.02%) and k-NN (98.69%). However, the latency chart reveals a key trade-off: SVM and Random Forest predict in under 0.2ms per sample, while k-NN takes over 2ms — over 10x slower — because k-NN must compare against the entire training set at inference time rather than using a pre-learned decision boundary.

### Fig. 6 — ROC Curves (One-vs-Rest)

![ROC Curves](results/fig6_roc_curves.png)

Since this is a 7-class problem, ROC curves are computed using a One-vs-Rest strategy — each gesture is treated as "positive" against all others combined. Nearly every class hugs the top-left corner with an AUC of 1.000, meaning the classifier almost perfectly separates that gesture from the rest. The macro-average AUC across all 7 classes is **0.996**, confirming excellent overall separability, with Thumbs Up (0.992) and Fist (0.997) being the relatively hardest (though still excellent) classes to distinguish — consistent with the confusion matrix above.

### Fig. 7 — Real-Time System Output Screens

![Output Screens](results/fig7_output_screens.png)

A mockup of what the live application UI looks like for all 7 gestures during real-time inference. Each panel shows the detected hand skeleton overlay, the recognized gesture name, the resulting media command, and a live confidence score bar (ranging ~92–97% in these examples). This is the actual interface a user sees while interacting with the system via webcam.

### Fig. 8 — Usability Evaluation (System Usability Scale)

![Usability Evaluation](results/fig8_usability.png)

Five users tested the system and rated it across five criteria (left panel): Ease of Learning, Ease of Use, Accuracy Perception, Response Time, and Overall Satisfaction — all scoring at or above the "Good" threshold of 4.0/5, with Response Time rated highest (4.4/5). The right panel shows each user's individual System Usability Scale (SUS) score, a standard 0–100 HCI usability metric. The mean SUS score is **82.4**, which falls in the "Good" usability grade (≥80.3), with 4 out of 5 users scoring above that bar.

---

## 🧠 Model Performance Summary

| Model | Accuracy | Precision | Recall | F1 | Latency (ms) |
|---|---|---|---|---|---|
| k-NN (k=5) | 98.69% | 98.70% | 98.69% | 98.69% | 2.466 |
| **SVM (RBF)** ⭐ Best | **99.26%** | **99.27%** | **99.26%** | **99.26%** | 0.185 |
| Random Forest | 99.02% | 99.03% | 99.02% | 99.02% | 0.031 |

5-fold cross-validation confirms stable performance (SVM: 99.11% ± 0.15% across folds). The trained SVM model (best performer) is saved as `gesture_model.pkl` and used in the live demo.

---

## 📁 Repository Structure

```
.
├── Cp.py                  # Full pipeline: data gen → training → evaluation → figures/tables (run as notebook cells)
├── webcam_demo.py         # Standalone live webcam demo (loads gesture_model.pkl)
├── gesture_model.pkl      # Trained SVM model (best performer)
├── hand_landmarker.task   # MediaPipe hand landmark detector model
├── results/               # All generated figures (architecture, landmarks, metrics, ROC, usability)
├── media/                 # Demo video recording
└── docs/                  # Full written project report (PDF)
```

---

## 🚀 Getting Started

### Requirements
- Python 3.9–3.11
- A webcam

### Install dependencies
```bash
pip install mediapipe scikit-learn matplotlib seaborn pandas numpy Pillow pyautogui opencv-python
```

### Run the full pipeline (data generation, training, figures, tables)
Open `Cp.py` in Jupyter/Colab and run cell by cell, **or**:
```bash
python Cp.py
```

### Run the live webcam demo only
Make sure `gesture_model.pkl` and `hand_landmarker.task` are in the same folder, then:
```bash
python webcam_demo.py
```
Press **`q`** to quit the webcam window.

---

## 📄 Full Report

See [`docs/Report.pdf`](docs/Report.pdf) for the complete write-up: literature review, methodology, evaluation, and usability study.

---

## 👤 Author

**Rameeza** — [@rrameeza196](https://github.com/rrameeza196)
