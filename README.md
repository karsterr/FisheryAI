# 🐟 FisheryAI: Intelligence System for Global Fisheries

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Lite-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

**FisheryAI** is a lightweight Deep Learning system designed to analyze time-series data related to global fisheries and aquaculture. Unlike traditional dashboarding tools, it uses an **LSTM-based Data-to-Text** architecture to generate context-aware status reports in natural language (Turkish).

It is optimized for **Edge AI** (offline inference) using TensorFlow Lite, making it capable of running on low-resource devices like Raspberry Pi or standard laptops without internet connectivity.

---

## 🚀 Key Features

* **Offline Intelligence:** No cloud dependency. Runs locally on CPU (~9ms inference time).
* **Context Awareness:** Understands the difference between a "numerical drop" and a "critical stock depletion".
* **Data-to-Text:** Converts raw CSV sequences into meaningful human-readable reports.
* **Multi-Domain:** Supports analysis for Capture, Aquaculture, Consumption, and Stock Sustainability.
* **Lightweight:** Powered by 8-bit quantized TFLite models.

---

## 🏗️ System Architecture

The project moves away from rule-based algorithms (if-else) and utilizes a **Sequence-to-Sequence (Seq2Seq)** neural network approach.

1.  **Input Layer:** Accepts 5-year normalized time-series data + Topic Embeddings.
2.  **Encoder:** An **LSTM (Long Short-Term Memory)** layer processes the temporal trends and compresses them into a latent vector.
3.  **Decoder:** Decodes the latent vector into natural language tokens (Turkish words).
4.  **Optimization:** The trained Keras model is converted to `.tflite` for edge deployment.

---

## 📂 Project Structure

```text
FisheryAI/
├── fishery_model.tflite    # Quantized Deep Learning Model
├── model_meta.json         # Tokenizer vocabulary and configuration
├── pc_inference.py         # Main CLI application for local inference
├── requirements.txt        # Python dependencies
└── README.md               # Documentation

```

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone [https://github.com/karsterr/FisheryAI](https://github.com/karsterr/FisheryAI)
cd FisheryAI

```

### 2. Set Up Virtual Environment (Recommended)

```bash
# Create environment (Python 3.10 or 3.11)
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate

# Activate (Mac/Linux)
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install tensorflow numpy
# OR for lightweight setup
pip install tflite-runtime numpy

```

---

## 💻 Usage

Run the main inference script:

```bash
python pc_inference.py

```

The system will launch an interactive CLI menu. Select a topic and enter the last 5 years of data values (separated by commas).

---

## 📊 Demo & Results

The system has been tested with real-world scenarios. Below is an example of the **Stock Sustainability** analysis detecting a critical trend.

**Scenario:** Critical decline in fish stocks (70% -> 40%).

```text
------------------------------------------------------------
 LÜTFEN ANALİZ TÜRÜNÜ SEÇİNİZ:
------------------------------------------------------------
  [1] AVCILIK ÜRETIMI           : Deniz ve iç sulardan avcılık yoluyla elde edilen miktar.
  [2] YETIŞTIRICILIK            : Çiftliklerde üretilen balık miktarı.
  [3] TÜKETIM                   : Kişi başı yıllık tüketim.
  [4] STOK SÜRDÜRÜLEBILIRLIĞI   : Biyolojik olarak güvenli seviyede olan stok.
------------------------------------------------------------
👉 Seçiminiz (1-4) veya 'q': 4

✅ SEÇİLEN MOD: STOK SÜRDÜRÜLEBILIRLIĞI
ℹ️  BİRİM      : YÜZDE (%)

📊 Son 5 yılın verilerini giriniz (Virgülle ayırın): 70, 65, 55, 48, 40

Analiz Ediliyor: ████████████████████ TAMAMLANDI

============================================================
📄 OTOMATİK RAPOR:
   "STOK SÜRDÜRÜLEBILIRLIĞI VERILERI DÜŞÜŞ SEYRETTI"

⚙️  Inference Süresi: 9.01 ms
============================================================

```

> **Note:** The model currently generates reports in **Turkish**. Multi-language support is planned for futur.

---

## 🔮 Future Roadmap

* [ ] Integration with **n8n** for automated email alerts.
* [ ] Dashboard visualization using **Streamlit**.
* [ ] Support for English and Spanish report generation.
* [ ] Deploying on Raspberry Pi 5 with Coral Accelerator.
---
