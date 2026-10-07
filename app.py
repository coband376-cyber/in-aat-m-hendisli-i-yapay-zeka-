import streamlit as st
import pickle
import json
import os
import zipfile
import pandas as pd

# Sayfa Yapılandırması
st.set_page_config(page_title="İnşaat Süre Tahmin Aracı", layout="centered")
st.title("🏗️ İnşaat Süre Tahmin Aracı")
st.write("Gerekli parametreleri girerek projenizin tahmini inşaat süresini hesaplayabilirsiniz.")

# Hata Önleyici Otomatik Çıkartma ve Yol Düzeltme
if os.path.exists('models.zip') and not os.path.exists('models'):
    try:
        with zipfile.ZipFile('models.zip', 'r') as zip_ref:
            zip_ref.extractall('models')
    except Exception:
        pass

# Modelleri ve Meta Veriyi Yükle
@st.cache_resource
def load_resources():
    paths = {
        'basit': 'models/sure_modeli_basit.pkl',
        'gelismis': 'models/sure_modeli_gelismis.pkl',
        'meta': 'models/meta.json'
    }

    # Alternatif: Eğer klasör yapılmadan direkt ana dizine yüklendiyse
    if not os.path.exists(paths['basit']):
        paths = {
            'basit': 'sure_modeli_basit.pkl',
            'gelismis': 'sure_modeli_gelismis.pkl',
            'meta': 'meta.json'
        }

    with open(paths['basit'], 'rb') as f:
        model_basit = pickle.load(f)
    with open(paths['gelismis'], 'rb') as f:
        model_gelismis = pickle.load(f)
    with open(paths['meta'], 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return model_basit, model_gelismis, meta

try:
    model_basit, model_gelismis, meta = load_resources()
except Exception as e:
    st.error("🚨 Model Dosyaları Eksik veya Yanlış Yüklenmiş!")
    st.info("GitHub deponuzda şunlardan birinin olduğundan emin olun:\n\n1. ya **models** adında bir klasör ve içinde modeller,\n2. ya da modellerin doğrudan ana dizinde bulunması.")
    st.exception(e)
    st.stop()

# Model Seçimi
model_tipi = st.selectbox("Tahmin Modeli Seçin:", ["Basit Model", "Gelişmiş Model"])

# Giriş Parametreleri
st.subheader("Proje Özellikleri")

alan_m2 = st.number_input("İnşaat Alanı (m²):", min_value=10, max_value=100000, value=1500, step=50)
kat_sayisi = st.number_input("Kat Sayısı:", min_value=1, max_value=100, value=5, step=1)

if model_tipi == "Gelişmiş Model":
    insaat_yili = st.number_input("İnşaat Yılı:", min_value=2015, max_value=2026, value=2023, step=1)
    zemin_sinifi = st.selectbox("Zemin Sınıfı:", ["A", "B", "C", "D"])

# Ekstrapolasyon (Aralık Dışı Girdi) Kontrolü ve Uyarısı
if alan_m2 < meta['alan_m2_min'] or alan_m2 > meta['alan_m2_max'] or \
   kat_sayisi < meta['kat_sayisi_min'] or kat_sayisi > meta['kat_sayisi_max']:
    st.warning("⚠️ UYARI: Girdiğiniz değerler eğitim verisi aralığının dışındadır! Model tahmini ekstrapolasyon nedeniyle daha az güvenilir olabilir.")

# Tahmin Butonu ve Hesaplama
if st.button("Süreyi Tahmin Et"):
    if model_tipi == "Basit Model":
        girdi = pd.DataFrame([[alan_m2, kat_sayisi]], columns=['alan_m2', 'kat_sayisi'])
        tahmin = model_basit.predict(girdi)[0]
    else:
        z_B = 1 if zemin_sinifi == "B" else 0
        z_C = 1 if zemin_sinifi == "C" else 0
        z_D = 1 if zemin_sinifi == "D" else 0

        girdi = pd.DataFrame(
            [[alan_m2, kat_sayisi, insaat_yili, z_B, z_C, z_D]],
            columns=['alan_m2', 'kat_sayisi', 'insaat_yili', 'zemin_sinifi_B', 'zemin_sinifi_C', 'zemin_sinifi_D']
        )
        tahmin = model_gelismis.predict(girdi)[0]

    # Tahmin negatif veya mantık dışı küçük olamaz
    tahmin_duzeltilmis = max(0.1, tahmin)
    st.success(f"### ⏱️ Tahmini İnşaat Süresi: **{tahmin_duzeltilmis:.1f} ay**")

# Kenar Çubuğu Performans Gösterimleri
st.sidebar.markdown("### 📊 Model Performans Değerleri")
if model_tipi == "Basit Model":
    st.sidebar.write(f"**Test R²:** {meta['r2_basit']:.3f}")
    st.sidebar.write(f"**Test MAE:** {meta['mae_basit']:.1f} ay")
else:
    st.sidebar.write(f"**Test R²:** {meta['r2_gelismis']:.3f}")
    st.sidebar.write(f"**Test MAE:** {meta['mae_gelismis']:.1f} ay")
