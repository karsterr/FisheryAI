# FisheryAI: Intelligence System for Global Fisheries

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Lite-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

**FisheryAI** is a lightweight Deep Learning system designed to analyze time-series data related to global fisheries and aquaculture. Unlike traditional dashboarding tools, it uses an **LSTM-based Data-to-Text** architecture to generate context-aware status reports in natural language (Turkish).

It is optimized for **Edge AI** (offline inference) using TensorFlow Lite, making it capable of running on low-resource devices like Raspberry Pi or standard laptops without internet connectivity.

---

## Key Features

* **Offline Intelligence:** No cloud dependency. Runs locally on CPU (~0.3ms inference time).
* **Context Awareness:** Understands the difference between a "numerical drop" and a "critical stock depletion".
* **Data-to-Text:** Converts raw time-series sequences into meaningful human-readable reports.
* **Multi-Domain:** Supports analysis for Capture, Aquaculture, Consumption, and Stock Sustainability.
* **Lightweight:** Powered by 8-bit quantized TFLite model (~108 KB).
* **Batch Mode:** Non-interactive CLI mode for scripting and automation.
* **Report Export:** Save analysis results to file with timestamps.
* **Session History:** Review all analyses performed during a session with sparkline visualizations.
* **Docker Ready:** One command to build and test with no local dependency setup.

---

## System Architecture

The project moves away from rule-based algorithms (if-else) and utilizes a **Sequence-to-Sequence (Seq2Seq)** neural network approach.

1. **Input Layer:** Accepts 5-year normalized time-series data + Topic Embeddings.
2. **Encoder:** An **LSTM (Long Short-Term Memory)** layer processes the temporal trends and compresses them into a latent vector.
3. **Decoder:** Decodes the latent vector into natural language tokens (Turkish words).
4. **Optimization:** The trained Keras model is converted to `.tflite` for edge deployment.

---

## Project Structure

```text
FisheryAI/
├── fishery_model.tflite       # Quantized Deep Learning Model (~108 KB)
├── fishery_model.tflite.png   # Model architecture visualization
├── model_meta.json            # Tokenizer vocabulary and configuration
├── model_development.ipynb    # Jupyter notebook (training pipeline, Colab)
├── pc_inference.py            # Main CLI application for local inference
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker containerization
├── .dockerignore              # Docker build exclusions
├── CONTRIBUTING.md            # Contribution guidelines
├── LICENSE                    # MIT License
└── README.md                  # This file
```

---

## Installation

### Option A: Docker (Recommended)

No Python setup required. Just Docker.

```bash
git clone https://github.com/karsterr/FisheryAI.git
cd FisheryAI
docker build -t fisheryai .
```

### Option B: Local Python

Requires Python 3.10 or 3.11.

```bash
git clone https://github.com/karsterr/FisheryAI.git
cd FisheryAI

# Create virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
# .\venv\Scripts\Activate      # Windows

# Install dependencies
pip install -r requirements.txt
# OR lightweight (TFLite runtime only):
pip install tflite-runtime numpy
```

---

## Usage

### Interactive Mode

```bash
# Local
python pc_inference.py

# Docker
docker run --rm -it fisheryai
```

Select a topic (1-4), enter 5 comma-separated data points, confirm, and get a report. Press `g` for session history, `q` to quit.

### Batch Mode (Non-Interactive)

```bash
# Local
python pc_inference.py --batch 4 70,65,55,48,40

# Docker
docker run --rm fisheryai --batch 4 70,65,55,48,40
```

Topic numbers:

| # | Topic | Unit |
|---|-------|------|
| 1 | Avcilik Uretimi (Capture Fisheries) | Metric Tons |
| 2 | Yetistiricilik (Aquaculture) | Metric Tons |
| 3 | Tuketim (Consumption) | kg/person/year |
| 4 | Stok Surdurulebilirligi (Stock Sustainability) | Percentage (%) |

### CLI Options

```text
--batch TOPIC_NUM DATA   Non-interactive analysis
--output, -o FILE        Append reports to a file with timestamps
--no-animation           Disable animations (accessibility / piped output)
--help, -h               Show help message
```

### Examples

```bash
# Save report to file
python pc_inference.py --batch 1 25000,28000,31000,35000,42000 -o reports.txt

# Disable animations for piping
python pc_inference.py --no-animation

# Docker with file export
docker run --rm -v $(pwd)/reports:/data fisheryai --batch 4 70,65,55,48,40 -o /data/reports.txt
```

---

## Demo & Results

Below is the actual output from a session testing all 4 analysis domains.

### Single Analysis

**Scenario:** Critical decline in fish stocks (70% -> 40%).

```text
  Secilen Mod : STOK SÜRDÜRÜLEBILIRLIĞI
  Birim       : YUZDE (%)
  Ipucu       : Orn: 80, 75, 60 (Dusmesi tehlikelidir)

  Son 5 yilin verilerini giriniz (virgul ile): 70, 65, 55, 48, 40

  Veri  : [70.0, 65.0, 55.0, 48.0, 40.0]
  Trend : █▆▄▂   [ASAGI]
  Analiz baslatilsin mi? (E/h): e

============================================================
  OTOMATIK RAPOR:
  "STOK SÜRDÜRÜLEBILIRLIĞI VERILERI DÜŞÜŞ SEYRETTI"

  Inference Suresi: 0.52 ms
============================================================
```

### Session History

After running multiple analyses, press `g` to see the full session history:

```text
============================================================
  OTURUM GECMISI (4 analiz)
============================================================
  1. [STOK SÜRDÜRÜLEBILIRLIĞI] (00:27:04)
     Veri  : [70.0, 65.0, 55.0, 48.0, 40.0]  █▆▄▂   [ASAGI]
     Rapor : "STOK SÜRDÜRÜLEBILIRLIĞI VERILERI DÜŞÜŞ SEYRETTI"
     Sure  : 0.52ms
  --------------------------------------------------------
  2. [AVCILIK ÜRETIMI] (00:27:29)
     Veri  : [25000.0, 28000.0, 31000.0, 35000.0, 42000.0]   ▁▂▄█  [YUKARI]
     Rapor : "AVCILIK ÜRETIMI VERILERI ARTIŞ SEYRETTI"
     Sure  : 0.36ms
  --------------------------------------------------------
  3. [YETIŞTIRICILIK] (00:27:39)
     Veri  : [1000.0, 1500.0, 2500.0, 4000.0, 6500.0]    ▂▄█  [YUKARI]
     Rapor : "YETIŞTIRICILIK VERILERI ARTIŞ SEYRETTI"
     Sure  : 0.32ms
  --------------------------------------------------------
  4. [TÜKETIM] (00:27:51)
     Veri  : [12.5, 13.0, 14.2, 15.8, 18.5]    ▂▄█  [YUKARI]
     Rapor : "TÜKETIM VERILERI ARTIŞ SEYRETTI"
     Sure  : 0.41ms
============================================================
```

> **Note:** The model currently generates reports in **Turkish**. Multi-language support is planned for future releases.

---

## Future Roadmap

* [ ] Integration with **n8n** for automated email alerts.
* [ ] Dashboard visualization using **Streamlit**.
* [ ] Support for English and Spanish report generation.
* [ ] Deploying on Raspberry Pi 5 with Coral Accelerator.
* [ ] REST API wrapper for integration with external systems.

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
