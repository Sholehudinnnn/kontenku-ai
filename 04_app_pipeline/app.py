import re
import math
from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "raw" / "social_media_viral_content_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "viral_classifier.joblib"
PIPELINE_PATH = BASE_DIR / "KontenKu_AI_Pipeline_Bertahap.ipynb"

st.set_page_config(page_title="KontenKu AI | Viral Studio", page_icon="✨", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--ink:#172033;--muted:#718096;--purple:#6657f5;--line:#e5e9f5}
.stApp{background:linear-gradient(180deg,#f6f7ff 0%,#fbfcff 45%,#f5f7ff 100%);color:var(--ink);font-family:'DM Sans',sans-serif}
.block-container{max-width:1320px;padding-top:1.2rem;padding-bottom:3rem}
h1,h2,h3,h4{font-family:'Space Grotesk',sans-serif!important;color:var(--ink)!important;letter-spacing:-.5px}
p,label,li{color:#43506a}
[data-testid="stWidgetLabel"] p,[data-testid="stWidgetLabel"] label{color:#202b45!important;font-weight:700!important}
div[data-baseweb="select"]>div,div[data-baseweb="input"]>div,div[data-baseweb="textarea"]>div{background:#fff!important;border:1px solid #dfe5f2!important;border-radius:13px!important;min-height:44px}
div[data-baseweb="select"] *,div[data-baseweb="input"] *,div[data-baseweb="textarea"] *{color:#172033!important;-webkit-text-fill-color:#172033!important}
input,textarea{color:#172033!important;-webkit-text-fill-color:#172033!important;background:#fff!important}
input::placeholder,textarea::placeholder{color:#8c96aa!important;-webkit-text-fill-color:#8c96aa!important;opacity:1!important}
[role="listbox"],[role="option"]{background:#fff!important;color:#172033!important}
div.stButton>button,div.stFormSubmitButton>button,div.stDownloadButton>button{background:linear-gradient(110deg,#5145e5,#8b5cf6);color:#fff!important;border:0;border-radius:14px;min-height:49px;font-weight:700;box-shadow:0 8px 20px #6657f52a;transition:transform .15s ease}
div.stButton>button:hover,div.stFormSubmitButton>button:hover,div.stDownloadButton>button:hover{transform:translateY(-1px);border:0}
div.stButton>button *,div.stFormSubmitButton>button *,div.stDownloadButton>button *{color:#fff!important}
.hero{position:relative;overflow:hidden;padding:34px 36px;border-radius:28px;background:radial-gradient(circle at 85% 15%,#a78bfa66,transparent 28%),linear-gradient(115deg,#151b3e,#3935a9 55%,#7254ed);margin-bottom:24px;box-shadow:0 22px 48px #3935a922}
.hero *{color:#fff!important}.eyebrow{font-size:11px;letter-spacing:2.2px;font-weight:700;opacity:.82}.hero h1{font-size:clamp(34px,4vw,52px);line-height:1.05;margin:.6rem 0}.hero p{opacity:.9;max-width:720px;font-size:16px}
.hero-pill{display:inline-block;padding:7px 12px;border:1px solid #ffffff44;background:#ffffff12;border-radius:999px;font-size:11px;letter-spacing:.5px;font-weight:700;margin-top:8px}
.section-kicker{font-size:11px;letter-spacing:1.6px;text-transform:uppercase;color:#7166d9;font-weight:800;margin-bottom:5px}
.card{background:#fff;border:1px solid #e4e8f3;border-radius:20px;padding:22px;box-shadow:0 9px 28px #27345c08;margin:8px 0}
.result-heading{font-size:27px;font-weight:700;line-height:1.2;color:#172033!important}
.prediction{border-radius:22px;padding:24px;background:linear-gradient(135deg,#171d42,#35339a);color:#fff;box-shadow:0 16px 36px #292b8224;margin:8px 0 18px}
.prediction *{color:#fff!important}.prediction-label{font-size:11px;letter-spacing:2px;font-weight:700;opacity:.8}.prediction-value{font-family:'Space Grotesk';font-size:clamp(27px,3vw,39px);font-weight:700;line-height:1.2;margin:10px 0}.prediction-note{font-size:13px;opacity:.82}
.small-chip{display:inline-block;background:#f0efff;color:#5145cd;border-radius:999px;padding:6px 10px;font-size:12px;font-weight:700;margin:4px 4px 4px 0}
div[data-testid="stAlert"]{border-radius:14px}
.footer{text-align:center;color:#9aa4b8;font-size:12px;padding-top:25px}
hr{border-color:#e5e9f5!important}
</style>
""", unsafe_allow_html=True)

def hashtags_for(topic, platform):
    words = re.findall(r"[a-zA-Z0-9]+", topic.lower())
    tags = []
    for word in words:
        tag = "#" + word
        if len(word) > 2 and tag not in tags:
            tags.append(tag)
    platform_tags = {"Instagram":["#InstagramReels", "#ReelsIndonesia"], "TikTok":["#TikTokIndonesia", "#FYP"], "YouTube Shorts":["#Shorts", "#YouTubeShorts"], "Facebook":["#FacebookReels", "#KontenIndonesia"], "X":["#Diskusi", "#Trending"]}
    for tag in platform_tags.get(platform, []):
        if tag not in tags: tags.append(tag)
    for tag in ["#KontenKuAI", "#TipsHarian"]:
        if tag not in tags: tags.append(tag)
    return " ".join(tags[:8])

def generate_content(platform, topic, audience, fmt, style, goal, duration):
    t = topic.strip(); aud = audience.strip() or "audiens umum"
    title = f"{t.title()}: 3 Tips Praktis yang Bisa Kamu Coba"
    hooks = {
      "Santai dan menarik": f"Stop scroll dulu! Kalau kamu tertarik dengan {t.lower()}, coba simak tips ini!",
      "Edukatif": f"Masih bingung tentang {t.lower()}? Ini panduan singkat yang bisa langsung kamu praktikkan.",
      "Lucu": f"Siapa yang masih suka bingung soal {t.lower()}? Tenang, kamu nggak sendirian 😅",
      "Profesional": f"Berikut strategi praktis untuk memahami {t.lower()} secara bertahap.",
      "Inspiratif": f"Kamu tidak harus langsung jago. Mulai tingkatkan kemampuan {t.lower()} lewat langkah kecil ini!"}
    hook = hooks[style]
    if fmt == "Video pendek":
        body = "\n\n".join([
          f"**HOOK · 0–3 detik**\n{hook}\n_Arahan visual: close-up, teks besar dan jelas._",
          f"**TUNJUKKAN MASALAH · 3–8 detik**\nPerlihatkan satu tantangan yang sering terjadi saat {t.lower()}.\n_Arahan visual: contoh situasi yang mudah dikenali._",
          f"**TIPS PERTAMA · 8–15 detik**\nBerikan satu langkah sederhana tentang {t.lower()} dan contoh penerapannya.\n_Arahan visual: demonstrasi singkat._",
          f"**TIPS KEDUA · 15–22 detik**\nTambahkan tips praktis kedua, lalu jelaskan manfaatnya dengan kalimat pendek.\n_Arahan visual: gunakan teks layar yang ringkas._",
          f"**PENUTUP · 22–{duration} detik**\nRangkum manfaatnya dan ajak penonton membagikan pengalaman.\n_Arahan visual: tampilkan CTA di layar._"])
    elif fmt == "Carousel":
        body = "\n\n".join([f"**Slide 1 — Sampul:** {title}", f"**Slide 2 — Masalah:** Tantangan yang sering dialami {aud} terkait {t.lower()}.", "**Slide 3 — Tips pertama:** Jelaskan langkah paling mudah untuk dicoba.", "**Slide 4 — Tips kedua:** Berikan contoh sederhana.", "**Slide 5 — Kesalahan umum:** Sebutkan satu hal yang sebaiknya dihindari.", "**Slide 6 — Ringkasan:** Rangkum dan ajak pembaca menyimpan postingan."])
    elif fmt == "Gambar / poster":
        body = f"**Headline:** {title}\n\n**Isi visual:** 3 poin utama tentang {t.lower()}.\n\n**Arahan desain:** Judul besar, satu ilustrasi utama, maksimal tiga poin, kontras warna yang jelas, dan ruang kosong cukup.\n\n**CTA:** Simpan tips ini untuk nanti."
    else:
        body = f"**Pembuka:** {hook}\n\n**Poin 1:** Kenalkan hal dasar tentang {t.lower()}.\n\n**Poin 2:** Berikan contoh yang mudah dipahami oleh {aud}.\n\n**Poin 3:** Sarankan langkah berikutnya yang realistis.\n\n**Penutup:** Ajukan pertanyaan agar pembaca ikut berdiskusi."
    caption = f"{hook}\n\nKita bahas {t.lower()} dengan cara yang praktis dan mudah dicoba. Pilih satu tips, praktikkan, lalu sesuaikan dengan kebutuhanmu.\n\nMenurut kamu, bagian mana yang paling menantang?"
    ctas = {"Engagement":"💬 Ceritakan pengalamanmu di komentar!", "Edukasi":"📌 Simpan postingan ini agar bisa dipelajari lagi.", "Promosi":"🛍️ Tanyakan detail produk atau layanan melalui kanal resmi.", "Brand awareness":"✨ Ikuti akun ini untuk mendapatkan ide berikutnya."}
    return {"title":title,"hook":hook,"body":body,"caption":caption,"hashtags":hashtags_for(t,platform),"cta":ctas[goal],"platform":platform,"topic":t,"audience":aud,"format":fmt,"goal":goal,"duration":duration}

def predict_fyp(r):
    """Use the trained local classifier when compatible; otherwise use a clearly labelled demo rule."""
    if MODEL_PATH.exists():
        try:
            import joblib
            model = joblib.load(MODEL_PATH)
            features = pd.DataFrame([{
                "platform": r["platform"],
                "content_type": r["format"],
                "topic": r["topic"],
                "language": r["language"],
                "region": r["region"],
                "post_month": int(r["month"]),
                "post_dayofweek": int(r["dayofweek"]),
                "hashtag_count": len(r["hashtags"].split())
            }])
            prediction = model.predict(features)[0]
            return ("Diprediksi viral menurut model", "Model Machine Learning terlatih") if str(prediction).lower() in ("1", "true", "viral", "yes") else ("Diprediksi tidak viral menurut model", "Model Machine Learning terlatih")
        except Exception:
            pass
    signals = 0
    if len(r["hook"]) >= 45: signals += 1
    if r["format"] == "Video pendek": signals += 1
    if r["platform"] in ("TikTok", "Instagram", "YouTube Shorts", "Facebook"): signals += 1
    if r["duration"] <= 30: signals += 1
    label = "Sinyal demo relatif kuat" if signals >= 3 else "Sinyal demo perlu diperkuat"
    return label, "Aturan demo (model belum bisa dipakai)"


st.markdown("""<div class="hero"><div class="eyebrow">✦ CREATOR INTELLIGENCE STUDIO</div><h1>KontenKu AI</h1><p>Rancang konten sosial media, susun naskah, lalu cek indikasi viral dari model proyek.</p><span class="hero-pill">● LOCAL DEMO · TANPA BIAYA API</span></div>""", unsafe_allow_html=True)
with st.form("content_form"):
    st.markdown('<div class="section-kicker">01 / Creative brief</div>', unsafe_allow_html=True)
    st.markdown("### Ceritakan konten yang ingin kamu buat")
    topic = st.text_input("Topik konten", placeholder="Contoh: resep anak kos, tips belajar, skincare pemula")
    audience = st.text_input("Target audiens · opsional", placeholder="Boleh dikosongkan, misalnya mahasiswa atau pemula")
    c1, c2, c3 = st.columns(3)
    with c1:
        platform = st.selectbox("Platform", ["Instagram", "TikTok", "YouTube Shorts", "Facebook", "X"])
        fmt = st.selectbox("Format konten", ["Video pendek", "Carousel", "Gambar / poster", "Teks"])
    with c2:
        style = st.selectbox("Gaya bahasa", ["Santai dan menarik", "Edukatif", "Lucu", "Profesional", "Inspiratif"])
        goal = st.selectbox("Tujuan konten", ["Engagement", "Edukasi", "Promosi", "Brand awareness"])
    with c3:
        duration = st.select_slider("Durasi video", options=[15, 20, 30, 45, 60], value=30, format_func=lambda x: f"{x} detik")
        language = st.selectbox("Bahasa konten", ["Indonesian", "English", "Malay", "Other"], format_func=lambda x: {"Indonesian":"Indonesia","English":"Inggris","Malay":"Melayu","Other":"Lainnya"}[x])
    with st.expander("⚙️ Pengaturan model (fitur yang digunakan saat training)"):
        st.caption("Model proyek dilatih menggunakan platform, tipe konten, topik, bahasa, region, bulan, hari dalam minggu, dan jumlah hashtag.")
        f1, f2 = st.columns(2)
        with f1:
            region = st.selectbox("Region", ["Indonesia", "Southeast Asia", "North America", "Europe", "Other"])
            month = st.selectbox("Bulan posting", list(range(1,13)), index=9, format_func=lambda x: ["Januari","Februari","Maret","April","Mei","Juni","Juli","Agustus","September","Oktober","November","Desember"][x-1])
        with f2:
            dayofweek = st.selectbox("Hari posting", list(range(7)), index=4, format_func=lambda x: ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"][x])
            hashtag_count = st.slider("Target jumlah hashtag", 1, 12, 6)
    submitted = st.form_submit_button("✨ BUAT KONTEN & CEK FYP", use_container_width=True)
if submitted:
    if not topic.strip():
        st.warning("Masukkan topik konten terlebih dahulu.")
    else:
        result = generate_content(platform, topic, audience, fmt, style, goal, duration)
        result.update({"language": language, "region": region, "month": month, "dayofweek": dayofweek})
        # Align the feature used for prediction with the user's selected hashtag count.
        tags = result["hashtags"].split()
        result["hashtags"] = " ".join(tags[:hashtag_count])
        st.session_state["konten_result"] = result

r = st.session_state.get("konten_result")
if r:
    prediction, prediction_source = predict_fyp(r)
    st.divider()
    st.markdown('<div class="section-kicker">02 / Model analysis</div><h2>Analisis & prediksi performa konten</h2>', unsafe_allow_html=True)
    st.caption("Prediksi ditentukan oleh model Machine Learning dari dataset proyek jika tersedia. Label viral pada dataset bukan berarti jaminan masuk FYP platform.")
    st.markdown(f'<div class="prediction"><div class="prediction-label">HASIL KLASIFIKASI MODEL</div><div class="prediction-value">{("✦ " if prediction in ("Berpotensi FYP", "Diprediksi viral menurut model", "Sinyal demo relatif kuat") else "↗ ")}{prediction}</div><div class="prediction-note">Sumber prediksi: {prediction_source}</div></div>', unsafe_allow_html=True)
    with st.expander("🔎 Apa arti hasil ini?"):
        if "Machine Learning" in prediction_source:
            st.write("Model mengklasifikasikan kombinasi fitur konten berdasarkan pola yang dipelajari saat training. Dataset proyek memiliki keterbatasan performa, sehingga hasil perlu dipakai sebagai indikasi awal, bukan keputusan pasti.")
        else:
            st.write("Model terlatih belum berhasil digunakan, sehingga hasil ini berasal dari aturan demo sederhana dan bukan prediksi dari training.")
        st.write("Fitur yang digunakan dalam model: platform, tipe konten, topik, bahasa, region, bulan posting, hari dalam minggu, dan jumlah hashtag.")
    st.markdown("---")
    lcol, rcol = st.columns([1.2, 1])
    with lcol:
        st.markdown('<div class="section-kicker">03 / Content blueprint</div>', unsafe_allow_html=True)
        st.markdown("### 🎬 Rancangan konten")
        st.markdown(f'<div class="card"><div class="result-heading">{r["title"]}</div><p>📱 {r["platform"]} · 🎯 {r["goal"]} · ⏱️ {r["duration"]} detik</p></div>', unsafe_allow_html=True)
        st.markdown("**Hook pembuka**")
        st.info(r["hook"])
        st.markdown("**Naskah / urutan konten**")
        st.markdown(r["body"])
    with rcol:
        st.markdown('<div class="section-kicker">04 / Publish kit</div>', unsafe_allow_html=True)
        st.markdown("### ✍️ Siap diposting")
        caption = st.text_area("Edit caption sebelum diunduh", value=r["caption"], height=180, key="caption_edit")
        st.markdown("**Hashtag**")
        st.code(r["hashtags"], language=None)
        st.markdown("**Call to action**")
        st.success(r["cta"])
    report = f"""KONTENKU AI — HASIL DEMO

Prediksi demo: {prediction}
Catatan: perkiraan berbasis aturan sederhana, bukan jaminan FYP.
Judul: {r['title']}
Platform: {r['platform']}
Format: {r['format']}
Topik: {r['topic']}
Target audiens: {r['audience'] or 'Tidak dibatasi'}
Tujuan: {r['goal']}
Durasi: {r['duration']} detik

HOOK
{r['hook']}

NASKAH
{r['body']}

CAPTION
{caption}

HASHTAG
{r['hashtags']}

CTA
{r['cta']}
"""
    st.download_button("⬇️ Unduh hasil konten (.txt)", report, "kontenku_ai_hasil.txt", "text/plain", use_container_width=True)
else:
    st.markdown('<div class="card"><h3>Mulai dalam 1 langkah</h3><p>Isi topik, pilih pengaturan, lalu tekan <b>Buat Analisis Konten</b>. Target audiens boleh dikosongkan.</p></div>', unsafe_allow_html=True)
with st.expander("📁 Detail dataset & model Machine Learning", expanded=False):
    st.markdown("### 1. Ringkasan dataset")
    st.write("Dataset digunakan untuk eksperimen klasifikasi label viral (`is_viral`), bukan untuk menjamin konten akan masuk FYP. Sumber dataset dan notebook eksperimen ditautkan pada bagian **Sumber data & bukti pipeline training**.")
    if DATASET_PATH.exists():
        try:
            df = pd.read_csv(DATASET_PATH)
            total_rows, total_cols = df.shape
            missing_total = int(df.isna().sum().sum())
            a, b, c, d = st.columns(4)
            a.metric("Jumlah baris", f"{total_rows:,}")
            b.metric("Jumlah kolom", f"{total_cols}")
            c.metric("Sel kosong", f"{missing_total:,}")
            d.metric("Duplikat baris", f"{int(df.duplicated().sum()):,}")
            st.markdown("**Nama kolom yang terbaca dari file**")
            st.code(", ".join(map(str, df.columns)), language="text")
            with st.expander("Lihat contoh data (maksimal 10 baris)"):
                st.dataframe(df.head(10), use_container_width=True)
            if "is_viral" in df.columns:
                st.markdown("**Distribusi target `is_viral`**")
                label_counts = df["is_viral"].astype(str).value_counts(dropna=False)
                st.bar_chart(label_counts)
                label_table = label_counts.rename_axis("Label").reset_index(name="Jumlah")
                label_table["Persentase"] = (label_table["Jumlah"] / max(len(df), 1) * 100).round(2)
                st.dataframe(label_table, hide_index=True, use_container_width=True)
            st.caption("Catatan kualitas data: jumlah baris, kolom, nilai kosong, duplikat, dan distribusi label di atas dihitung langsung dari CSV yang tersedia di folder proyek saat aplikasi dijalankan.")
        except Exception as e:
            st.warning(f"Dataset ditemukan tetapi tidak berhasil dibaca: {e}")
    else:
        st.warning("Dataset belum ditemukan. Lokasi yang diharapkan: data/raw/social_media_viral_content_dataset.csv")

    st.markdown("### 2. Fitur yang digunakan")
    feature_info = pd.DataFrame([
        {"Fitur": "platform", "Jenis": "Kategorikal", "Keterangan": "Platform media sosial yang dipilih"},
        {"Fitur": "content_type", "Jenis": "Kategorikal", "Keterangan": "Format konten, misalnya video atau gambar jika tersedia di dataset"},
        {"Fitur": "topic", "Jenis": "Kategorikal", "Keterangan": "Topik utama konten"},
        {"Fitur": "language", "Jenis": "Kategorikal", "Keterangan": "Bahasa konten"},
        {"Fitur": "region", "Jenis": "Kategorikal", "Keterangan": "Wilayah audiens atau publikasi sesuai dataset"},
        {"Fitur": "post_month", "Jenis": "Numerik/kalendar", "Keterangan": "Bulan publikasi yang diturunkan dari waktu posting"},
        {"Fitur": "post_dayofweek", "Jenis": "Numerik/kalendar", "Keterangan": "Hari dalam minggu yang diturunkan dari waktu posting"},
        {"Fitur": "hashtag_count", "Jenis": "Numerik", "Keterangan": "Jumlah hashtag pada konten"},
    ])
    st.dataframe(feature_info, hide_index=True, use_container_width=True)
    st.info("Pada file sumber proyek, waktu posting tercatat pada jam 00:00 untuk semua baris. Karena itu, jam posting tidak dipakai sebagai fitur terpisah; fitur waktu yang digunakan adalah bulan dan hari dalam minggu.")

    st.markdown("### 3. Algoritma dan alur training")
    st.write("Algoritma eksperimen: **Random Forest Classifier**. Secara umum, pipeline memuat data, memeriksa struktur dan distribusi label, membentuk fitur tanggal dan jumlah hashtag, melakukan preprocessing untuk fitur kategorikal/numerik, membagi data latih dan uji, melatih model, lalu mengevaluasi hasil pada data uji.")
    st.markdown("**Tahapan yang dapat ditelusuri di notebook:**")
    st.markdown("1. Import library dan membaca dataset.\n2. Pemeriksaan bentuk data, tipe kolom, nilai kosong, serta distribusi label.\n3. Feature engineering untuk bulan, hari dalam minggu, dan jumlah hashtag.\n4. Preprocessing fitur kategorikal dan numerik.\n5. Pembagian data train/test dan pelatihan Random Forest.\n6. Evaluasi model dan penyimpanan artefak model bila langkah penyimpanan berhasil.")

    st.markdown("### 4. Status artefak model pada komputer ini")
    if MODEL_PATH.exists():
        st.success(f"File model ditemukan: `{MODEL_PATH.relative_to(BASE_DIR)}`")
        st.caption("Keberadaan file menunjukkan artefak tersedia; bukan dengan sendirinya bukti performa model. Metrik evaluasi perlu dibaca dari output notebook untuk memastikan model yang dievaluasi sama dengan artefak ini.")
    else:
        st.warning("File model lokal belum ditemukan di `models/viral_classifier.joblib`. Jika demikian, hasil generator dapat menggunakan aturan demo dan bukan model terlatih. Periksa label sumber prediksi pada bagian hasil.")

    st.markdown("### 5. Cara membaca hasil dan keterbatasan")
    st.markdown("- `is_viral` adalah label pada dataset, bukan pengukuran langsung bahwa konten pasti masuk halaman FYP.\n- Model mempelajari pola dari data yang tersedia; pola tersebut belum tentu mewakili algoritma platform saat ini atau semua jenis audiens.\n- Jika label tidak seimbang, akurasi saja bisa menyesatkan. Bandingkan juga balanced accuracy, precision, recall, F1-score, confusion matrix, dan baseline sederhana.\n- Hasil klasifikasi adalah indikasi eksperimen, bukan probabilitas FYP yang terkalibrasi dan bukan jaminan jumlah tayangan.")
    st.caption("Untuk laporan akademik, cantumkan metrik aktual dari hasil eksekusi notebook yang dibagikan. Jangan mengklaim angka evaluasi yang tidak muncul pada output eksperimen.")

with st.expander("🔗 Sumber data & bukti pipeline training", expanded=False):
    st.markdown("Bagian ini menampilkan sumber yang bisa diperiksa, supaya proses proyek transparan dan tidak hanya berupa klaim di aplikasi.")
    st.markdown("**Dataset yang digunakan**")
    st.markdown("- [Dataset di Kaggle — Social Media Viral Content and Engagement Metrics](https://www.kaggle.com/datasets/aliiihussain/social-media-viral-content-and-engagement-metrics)")
    st.markdown("- [Salinan dataset di Hugging Face](https://huggingface.co/datasets/jason1966/aliiihussain_social-media-viral-content-and-engagement-metrics)")
    st.markdown("**Pipeline training model**")
    st.markdown("Notebook berisi tahapan impor library, mengambil/membaca dataset, EDA, feature engineering, preprocessing, training Random Forest, evaluasi, dan penyimpanan model.")
    if PIPELINE_PATH.exists():
        with open(PIPELINE_PATH, "rb") as f:
            st.download_button("⬇️ Unduh notebook pipeline (.ipynb)", data=f.read(), file_name="KontenKu_AI_Pipeline_Bertahap.ipynb", mime="application/x-ipynb+json", use_container_width=True)
        st.link_button("🚀 Buka notebook pipeline KontenKu AI di Google Colab", "https://colab.research.google.com/drive/1v7WTuGfa9vadYgOwMC9Ql6iL58brl0pW?usp=sharing", use_container_width=True)
        st.caption("Tombol di atas membuka notebook pipeline Google Colab yang sudah dibagikan. Tombol unduh notebook tetap tersedia sebagai salinan lokal.")
    else:
        st.warning("File notebook belum ada di folder aplikasi. Letakkan KontenKu_AI_Pipeline_Bertahap.ipynb di folder yang sama dengan file aplikasi agar tombol unduh muncul.")
        st.link_button("🚀 Buka notebook pipeline KontenKu AI di Google Colab", "https://colab.research.google.com/drive/1v7WTuGfa9vadYgOwMC9Ql6iL58brl0pW?usp=sharing", use_container_width=True)
    st.caption("Catatan: tautan sumber membuktikan asal dataset; notebook menunjukkan alur eksperimen. Hasil model tetap perlu dinilai dari metrik evaluasi dan bukan jaminan FYP.")

st.markdown('<div class="footer">KONTENKU AI · SUMBER TERBUKA · PREDIKSI BUKAN JAMINAN FYP</div>',unsafe_allow_html=True)
