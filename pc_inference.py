import os
import json
import time
import sys
import numpy as np

# Konsol temizliği ve uyarı gizleme
os.system('cls' if os.name == 'nt' else 'clear')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow.lite as tflite
except ImportError:
    import tflite_runtime.interpreter as tflite

# --- SUNUM İÇİN DETAYLI BİLGİ SÖZLÜĞÜ ---
TOPIC_INFO = {
    'avcılık üretimi': {
        'tanim': 'Deniz ve iç sulardan avcılık yoluyla elde edilen toplam balık miktarı.',
        'birim': 'METRİK TON (1000 kg)',
        'ornek': 'Örn: 15000, 18000 (Büyük ölçekli üretim)'
    },
    'yetiştiricilik': {
        'tanim': 'Çiftliklerde (Kültür Balıkçılığı) üretilen balık miktarı.',
        'birim': 'METRİK TON (1000 kg)',
        'ornek': 'Örn: 5000, 7500 (Hızla büyüyen sektör)'
    },
    'tüketim': {
        'tanim': 'Bir kişinin yıllık ortalama tükettiği balık/su ürünleri miktarı.',
        'birim': 'KG / KİŞİ / YIL',
        'ornek': 'Örn: 15.5, 16.2 (Türkiye ort: ~6-8 kg)'
    },
    'stok sürdürülebilirliği': {
        'tanim': 'Balık stoklarının biyolojik olarak güvenli seviyede olan kısmı.',
        'birim': 'YÜZDE (%)',
        'ornek': 'Örn: 80, 75, 60 (Düşmesi tehlikedir)'
    }
}

class FisheryAI:
    def __init__(self, model_path='fishery_model.tflite', meta_path='model_meta.json'):
        # 1. Dosya Kontrol
        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            print(f"❌ HATA: Dosyalar eksik! ({model_path})")
            exit(1)

        # 2. Meta Veri Yükleme
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)

        # 3. Tokenizer Onarımı
        try:
            tokenizer_str = self.meta['tokenizer_json']
            tokenizer_data = json.loads(tokenizer_str) if isinstance(tokenizer_str, str) else tokenizer_str

            # Farklı Keras sürümleri için path kontrolü
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

        except Exception as e:
            print(f"❌ Sözlük Hatası: {e}")
            exit(1)

        self.index_word = {v: k for k, v in self.word_index.items()}
        self.topic_map = self.meta['topic_map']

        # 4. Model Yükleme
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def analyze(self, data_window, topic_name):
        start_time = time.time()

        # Preprocessing
        data = np.array(data_window, dtype=np.float32)
        _min, _max = data.min(), data.max()
        scaled = np.zeros_like(data) if _max == _min else (data - _min) / (_max - _min)

        input_ts = scaled.reshape(1, 5, 1).astype(np.float32)
        topic_id = self.topic_map.get(topic_name, 0)
        input_type = np.array([[topic_id]], dtype=np.float32)

        # Inference
        try:
            idx_ts = self.input_details[0]['index']
            idx_type = self.input_details[1]['index']
            # İsim kontrolü
            for i in self.input_details:
                if 'ts_input' in i['name']: idx_ts = i['index']
                if 'type_input' in i['name']: idx_type = i['index']

            self.interpreter.set_tensor(idx_ts, input_ts)
            self.interpreter.set_tensor(idx_type, input_type)
        except:
            return "MODEL GİRDİ HATASI", 0

        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Decoding
        result_words = []
        for step in output_data[0]:
            token_id = np.argmax(step)
            if token_id == 0: continue
            word = self.index_word.get(token_id, '')
            if word == 'end': break
            if word not in ['start', '']: result_words.append(word)

        return " ".join(result_words), (time.time() - start_time) * 1000

def typing_effect(text, speed=0.02):
    """Metni daktilo efektiyle yazar"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(speed)
    print()

def loading_bar():
    """Yapay zeka düşünüyormuş gibi animasyon"""
    print("Analiz Ediliyor: ", end="")
    for _ in range(20):
        sys.stdout.write("█")
        sys.stdout.flush()
        time.sleep(0.05)
    print(" TAMAMLANDI\n")

# --- ANA PROGRAM ---
if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    print(r"""
=======================================================
GLOBAL FISHERY INTELLIGENCE SYSTEM (v1.0)
Deep Learning based Data-to-Text Reporter
=======================================================
    """)

    try:
        ai = FisheryAI()

        topics = list(ai.topic_map.keys())

        while True:
            print("\n" + "-"*60)
            print(" LÜTFEN ANALİZ TÜRÜNÜ SEÇİNİZ:")
            print("-"*60)
            for i, t in enumerate(topics, 1):
                # Konu açıklamasını sözlükten çek
                desc = TOPIC_INFO.get(t, {}).get('tanim', 'Genel Veri')
                print(f"  [{i}] {t.upper().ljust(25)} : {desc}")
            print("-"*60)

            choice = input(f"👉 Seçiminiz (1-{len(topics)}) veya 'q': ").strip()
            if choice.lower() == 'q':
                print("\nSistem kapatılıyor... Teşekkürler.")
                break

            if not choice.isdigit() or int(choice) < 1 or int(choice) > len(topics):
                print("⚠️  Geçersiz seçim!")
                continue

            selected_topic = topics[int(choice) - 1]
            info = TOPIC_INFO.get(selected_topic, {})

            print(f"\n✅ SEÇİLEN MOD: {selected_topic.upper()}")
            print(f"ℹ️  BİRİM      : {info.get('birim', 'Birim Belirsiz')}")
            print(f"💡 İPUCU      : {info.get('ornek', '')}")

            raw_in = input("\n📊 Son 5 yılın verilerini giriniz (Virgülle ayırın): ")
            try:
                data = [float(x.strip()) for x in raw_in.split(',')]
                if len(data) != 5:
                    print(f"❌ HATA: 5 adet veri noktası girilmelidir. ({len(data)} girildi)")
                    continue

                # --- SUNUM EFEKTİ ---
                print("")
                loading_bar()

                report, ms = ai.analyze(data, selected_topic)

                print("="*60)
                print(f"📄 OTOMATİK RAPOR:")
                typing_effect(f"   \"{report.upper()}\"", speed=0.04)
                print(f"\n⚙️  Inference Süresi: {ms:.2f} ms")
                print("="*60)

                input("\nDevam etmek için Enter'a basın...")
                os.system('cls' if os.name == 'nt' else 'clear') # Ekranı temizle

            except ValueError:
                print("❌ HATA: Sadece sayısal değer giriniz.")

    except Exception as e:
        print(f"\n💥 Kritik Hata: {e}")
        input()