import os
import json
import time
import sys
import argparse
import math
import numpy as np

# TF logging suppression (before import)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow.lite as tflite
except ImportError:
    import tflite_runtime.interpreter as tflite

# --- Resolve paths relative to this script, not cwd ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- SUNUM ICIN DETAYLI BILGI SOZLUGU ---
TOPIC_INFO = {
    'avcılık üretimi': {
        'tanim': 'Deniz ve ic sulardan avcilik yoluyla elde edilen toplam balik miktari.',
        'birim': 'METRIK TON (1000 kg)',
        'ornek': 'Orn: 15000, 18000 (Buyuk olcekli uretim)'
    },
    'yetiştiricilik': {
        'tanim': 'Ciftliklerde (Kultur Balikçiligi) uretilen balik miktari.',
        'birim': 'METRIK TON (1000 kg)',
        'ornek': 'Orn: 5000, 7500 (Hizla buyuyen sektor)'
    },
    'tüketim': {
        'tanim': 'Bir kisinin yillik ortalama tukettigi balik/su urunleri miktari.',
        'birim': 'KG / KISI / YIL',
        'ornek': 'Orn: 15.5, 16.2 (Turkiye ort: ~6-8 kg)'
    },
    'stok sürdürülebilirliği': {
        'tanim': 'Balik stoklarinin biyolojik olarak guvenli seviyede olan kismi.',
        'birim': 'YUZDE (%)',
        'ornek': 'Orn: 80, 75, 60 (Dusmesi tehlikelidir)'
    }
}

# --- Width constant for consistent formatting ---
WIDTH = 60


def clear_screen():
    """Clear the terminal using ANSI escape codes (safe, no shell spawn)."""
    sys.stdout.write('\033[2J\033[H')
    sys.stdout.flush()


def typing_effect(text, speed=0.02, animate=True):
    """Print text with optional typewriter animation."""
    if not animate:
        print(text)
        return
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(speed)
    print()


def sparkline(values):
    """Generate a small ASCII sparkline chart for a list of numbers."""
    if not values:
        return ""
    blocks = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    lo, hi = min(values), max(values)
    if hi == lo:
        return blocks[4] * len(values)
    spread = hi - lo
    return "".join(
        blocks[min(8, max(0, int((v - lo) / spread * 8)))] for v in values
    )


def format_trend_arrow(values):
    """Return a trend indicator based on start vs end value."""
    if len(values) < 2:
        return ""
    diff = values[-1] - values[0]
    pct = (diff / (abs(values[0]) + 1e-9)) * 100
    if pct > 10:
        return "  [YUKARI]"
    elif pct < -10:
        return "  [ASAGI]"
    else:
        return "  [YATAY]"


class FisheryAI:
    """
    TFLite-based fishery data-to-text inference engine.

    Loads a quantized LSTM model and generates natural-language trend
    reports (in Turkish) from 5-year time-series windows.
    """

    def __init__(self, model_path=None, meta_path=None):
        # Resolve default paths relative to script location
        if model_path is None:
            model_path = os.path.join(_SCRIPT_DIR, 'fishery_model.tflite')
        if meta_path is None:
            meta_path = os.path.join(_SCRIPT_DIR, 'model_meta.json')

        # 1. File existence checks with specific error messages
        missing = []
        if not os.path.exists(model_path):
            missing.append(f"Model: {model_path}")
        if not os.path.exists(meta_path):
            missing.append(f"Meta:  {meta_path}")
        if missing:
            raise FileNotFoundError(
                "Gerekli dosyalar bulunamadi:\n  " + "\n  ".join(missing)
            )

        # 2. Meta data loading
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.meta = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Meta dosyasi gecersiz JSON iceriyor: {e}")

        # 3. Tokenizer reconstruction (supports multiple Keras formats)
        try:
            tokenizer_str = self.meta['tokenizer_json']
            tokenizer_data = (
                json.loads(tokenizer_str)
                if isinstance(tokenizer_str, str)
                else tokenizer_str
            )

            if 'config' in tokenizer_data and 'word_index' in tokenizer_data['config']:
                raw_word_index = tokenizer_data['config']['word_index']
            elif 'word_index' in tokenizer_data:
                raw_word_index = tokenizer_data['word_index']
            else:
                raw_word_index = tokenizer_data

            if isinstance(raw_word_index, str):
                self.word_index = json.loads(raw_word_index)
            else:
                self.word_index = raw_word_index

        except KeyError as e:
            raise ValueError(f"Meta dosyasinda beklenen anahtar bulunamadi: {e}")
        except Exception as e:
            raise ValueError(f"Sozluk olusturma hatasi: {e}")

        self.index_word = {v: k for k, v in self.word_index.items()}

        # 4. Topic map validation
        if 'topic_map' not in self.meta:
            raise ValueError("Meta dosyasinda 'topic_map' anahtari bulunamadi.")
        self.topic_map = self.meta['topic_map']

        # 5. Model loading
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def analyze(self, data_window, topic_name):
        """
        Analyze a 5-value time series window for the given topic.

        Args:
            data_window: list/tuple of exactly 5 numeric values
            topic_name: one of the keys in self.topic_map

        Returns:
            tuple: (report_string, inference_time_ms)

        Raises:
            ValueError: if data_window length != 5, contains NaN/Inf,
                        or topic_name is invalid
            RuntimeError: if model invocation fails
        """
        # --- Input validation ---
        if not isinstance(data_window, (list, tuple)) or len(data_window) != 5:
            count = (
                len(data_window)
                if isinstance(data_window, (list, tuple))
                else type(data_window).__name__
            )
            raise ValueError(
                f"Tam olarak 5 veri noktasi gerekli, {count} verildi."
            )

        for i, v in enumerate(data_window):
            if not isinstance(v, (int, float)):
                raise TypeError(f"Veri noktasi {i+1} sayisal degil: {v!r}")
            if math.isnan(v) or math.isinf(v):
                raise ValueError(f"Veri noktasi {i+1} gecersiz (NaN/Inf): {v}")

        if topic_name not in self.topic_map:
            valid = ", ".join(self.topic_map.keys())
            raise ValueError(
                f"Gecersiz konu: '{topic_name}'. Gecerli konular: {valid}"
            )

        start_time = time.time()

        # --- Preprocessing (identical logic to original) ---
        data = np.array(data_window, dtype=np.float32)
        _min, _max = data.min(), data.max()
        scaled = (
            np.zeros_like(data) if _max == _min
            else (data - _min) / (_max - _min)
        )

        input_ts = scaled.reshape(1, 5, 1).astype(np.float32)
        topic_id = self.topic_map[topic_name]
        input_type = np.array([[topic_id]], dtype=np.float32)

        # --- Inference (fully inside try/except) ---
        try:
            idx_ts = self.input_details[0]['index']
            idx_type = self.input_details[1]['index']
            for detail in self.input_details:
                if 'ts_input' in detail['name']:
                    idx_ts = detail['index']
                if 'type_input' in detail['name']:
                    idx_type = detail['index']

            self.interpreter.set_tensor(idx_ts, input_ts)
            self.interpreter.set_tensor(idx_type, input_type)
            self.interpreter.invoke()
        except Exception as e:
            raise RuntimeError(f"Model calistirma hatasi: {e}")

        output_data = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )

        # --- Decoding ---
        result_words = []
        for step in output_data[0]:
            token_id = np.argmax(step)
            if token_id == 0:
                continue
            word = self.index_word.get(token_id, '')
            if word == 'end':
                break
            if word not in ('start', ''):
                result_words.append(word)

        elapsed_ms = (time.time() - start_time) * 1000

        if not result_words:
            return "(Model bu girdi icin rapor uretemedi)", elapsed_ms

        return " ".join(result_words), elapsed_ms


# =========================================================================
#  CLI Application
# =========================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FisheryAI - Global Fishery Intelligence System (v1.1)",
        epilog="Ornek: python pc_inference.py --batch 4 70,65,55,48,40",
    )
    parser.add_argument(
        '--batch', nargs=2, metavar=('TOPIC_NUM', 'DATA'),
        help="Non-interactive mod: TOPIC_NUM (1-4) ve DATA (virgul ile ayrilmis 5 deger)",
    )
    parser.add_argument(
        '--output', '-o', metavar='FILE',
        help="Raporu bir dosyaya kaydet",
    )
    parser.add_argument(
        '--no-animation', action='store_true',
        help="Animasyonlari devre disi birak (erisilebilirlik / pipe icin)",
    )
    return parser.parse_args()


def save_report(filepath, topic, data, report, ms):
    """Append a timestamped report line to a file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(
            f"[{timestamp}] Konu: {topic.upper()} | "
            f"Veri: {data} | Rapor: {report} | Sure: {ms:.2f}ms\n"
        )


def show_history(history):
    """Display analysis history from the current session."""
    if not history:
        print(f"\n{'='*WIDTH}")
        print("  Henuz analiz yapilmadi.")
        print(f"{'='*WIDTH}")
        return

    print(f"\n{'='*WIDTH}")
    print(f"  OTURUM GECMISI ({len(history)} analiz)")
    print(f"{'='*WIDTH}")
    for idx, entry in enumerate(history, 1):
        trend = sparkline(entry['data'])
        arrow = format_trend_arrow(entry['data'])
        print(f"  {idx}. [{entry['topic'].upper()}] ({entry['time']})")
        print(f"     Veri  : {entry['data']}  {trend}{arrow}")
        print(f"     Rapor : \"{entry['report'].upper()}\"")
        print(f"     Sure  : {entry['ms']:.2f}ms")
        if idx < len(history):
            print(f"  {'-'*(WIDTH-4)}")
    print(f"{'='*WIDTH}")


def run_batch(args, ai):
    """Run a single analysis in batch (non-interactive) mode."""
    topics = list(ai.topic_map.keys())
    topic_num, data_str = args.batch

    if not topic_num.isdigit() or int(topic_num) < 1 or int(topic_num) > len(topics):
        print(
            f"HATA: Gecersiz konu numarasi '{topic_num}'. "
            f"1-{len(topics)} araliginda olmali.",
            file=sys.stderr,
        )
        sys.exit(1)

    selected_topic = topics[int(topic_num) - 1]

    try:
        data = [float(x.strip()) for x in data_str.split(',')]
    except ValueError:
        print("HATA: Veriler sayisal olmali (virgul ile ayrilmis).", file=sys.stderr)
        sys.exit(1)

    if len(data) != 5:
        print(f"HATA: 5 veri noktasi gerekli, {len(data)} verildi.", file=sys.stderr)
        sys.exit(1)

    try:
        report, ms = ai.analyze(data, selected_topic)
    except (ValueError, TypeError, RuntimeError) as e:
        print(f"HATA: {e}", file=sys.stderr)
        sys.exit(1)

    output_line = f"{selected_topic.upper()}: \"{report.upper()}\" ({ms:.2f}ms)"
    print(output_line)

    if args.output:
        save_report(args.output, selected_topic, data, report, ms)
        print(f"Rapor kaydedildi: {args.output}", file=sys.stderr)


def run_interactive(args, ai):
    """Run the interactive CLI menu loop."""
    animate = not args.no_animation
    topics = list(ai.topic_map.keys())
    history = []

    clear_screen()
    print(f"""
{'='*WIDTH}
  GLOBAL FISHERY INTELLIGENCE SYSTEM (v1.1)
  Deep Learning based Data-to-Text Reporter
{'='*WIDTH}
""")

    while True:
        # --- Menu ---
        print(f"\n{'-'*WIDTH}")
        print("  LUTFEN ANALIZ TURUNU SECINIZ:")
        print(f"{'-'*WIDTH}")
        for i, t in enumerate(topics, 1):
            desc = TOPIC_INFO.get(t, {}).get('tanim', 'Genel Veri')
            print(f"  [{i}] {t.upper().ljust(25)} : {desc}")
        print(f"{'-'*WIDTH}")
        print(f"  [g] Gecmis   [q] Cikis")
        print(f"{'-'*WIDTH}")

        try:
            choice = input(
                f"\n  Seciminiz (1-{len(topics)}, g, q): "
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Cikis yapiliyor...")
            break

        # --- Quit ---
        if choice == 'q':
            print("\n  Sistem kapatiliyor... Tesekkurler.")
            break

        # --- History ---
        if choice == 'g':
            show_history(history)
            try:
                input("\n  Devam etmek icin Enter'a basin...")
            except (KeyboardInterrupt, EOFError):
                print("\n\n  Cikis yapiliyor...")
                break
            clear_screen()
            continue

        # --- Topic selection ---
        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(topics):
            print(f"\n  [!] Gecersiz secim. 1-{len(topics)} araliginda bir sayi girin.")
            continue

        selected_topic = topics[int(choice) - 1]
        info = TOPIC_INFO.get(selected_topic, {})

        print(f"\n  Secilen Mod : {selected_topic.upper()}")
        print(f"  Birim       : {info.get('birim', 'Birim Belirsiz')}")
        print(f"  Ipucu       : {info.get('ornek', '')}")

        # --- Data entry ---
        try:
            raw_in = input("\n  Son 5 yilin verilerini giriniz (virgul ile): ")
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Cikis yapiliyor...")
            break

        try:
            data = [float(x.strip()) for x in raw_in.split(',')]
        except ValueError:
            print("  [!] HATA: Sadece sayisal deger giriniz.")
            continue

        if len(data) != 5:
            print(f"\n  [!] HATA: 5 adet veri noktasi gerekli ({len(data)} girildi).")
            continue

        # NaN / Inf guard
        bad = [v for v in data if math.isnan(v) or math.isinf(v)]
        if bad:
            print(f"\n  [!] HATA: Gecersiz deger tespit edildi: {bad}")
            continue

        # Show data with sparkline and ask for confirmation
        trend = sparkline(data)
        arrow = format_trend_arrow(data)
        print(f"\n  Veri  : {data}")
        print(f"  Trend : {trend}{arrow}")

        try:
            confirm = input("  Analiz baslatilsin mi? (E/h): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Cikis yapiliyor...")
            break

        if confirm == 'h':
            print("  Iptal edildi.")
            continue

        # --- Analysis ---
        print()
        spinner_chars = "\u28CB\u28D9\u28F9\u28F8\u28FC\u28F4\u28E6\u28E7\u28C7\u28CF"

        try:
            if animate:
                sys.stdout.write("  Analiz ediliyor... ")
                sys.stdout.flush()

            report, ms = ai.analyze(data, selected_topic)

            if animate:
                # Brief spinner proportional to inference time (min 0.3s visibility)
                spin_duration = max(0.3, ms / 1000)
                end_spin = time.time() + spin_duration
                i = 0
                while time.time() < end_spin:
                    sys.stdout.write(
                        f"\r  Analiz ediliyor... "
                        f"{spinner_chars[i % len(spinner_chars)]}"
                    )
                    sys.stdout.flush()
                    time.sleep(0.08)
                    i += 1
                sys.stdout.write("\r  Analiz tamamlandi.       \n")
                sys.stdout.flush()

        except (ValueError, TypeError, RuntimeError) as e:
            print(f"  [!] Analiz hatasi: {e}")
            continue

        # --- Display report ---
        print(f"\n{'='*WIDTH}")
        print(f"  OTOMATIK RAPOR:")
        typing_effect(f'  "{report.upper()}"', speed=0.04, animate=animate)
        print(f"\n  Inference Suresi: {ms:.2f} ms")
        print(f"{'='*WIDTH}")

        # Save to session history
        history.append({
            'topic': selected_topic,
            'data': data,
            'report': report,
            'ms': ms,
            'time': time.strftime("%H:%M:%S"),
        })

        # Save to file if --output specified
        if args.output:
            save_report(args.output, selected_topic, data, report, ms)
            print(f"  (Rapor kaydedildi: {args.output})")

        try:
            input("\n  Devam etmek icin Enter'a basin...")
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Cikis yapiliyor...")
            break
        clear_screen()

    # Session summary on exit
    if history:
        print(f"\n  Bu oturumda {len(history)} analiz yapildi.")


# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    args = parse_args()

    try:
        print("  Model yukleniyor...", end=" ", flush=True)
        ai = FisheryAI()
        print("Hazir.")
    except FileNotFoundError as e:
        print(f"\n  [!] DOSYA HATASI: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n  [!] YAPILANDIRMA HATASI: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n  [!] BEKLENMEYEN HATA: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.batch:
            run_batch(args, ai)
        else:
            run_interactive(args, ai)
    except KeyboardInterrupt:
        print("\n\n  Ctrl+C ile cikis yapildi.")
        sys.exit(0)
    except EOFError:
        print("\n\n  Girdi sonu - cikis yapiliyor.")
        sys.exit(0)
    except Exception as e:
        print(f"\n  [!] Kritik Hata: {e}", file=sys.stderr)
        sys.exit(1)
