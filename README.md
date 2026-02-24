# 🔍 Exploratory Data Analysis (EDA) — Pothole Segmentation
### ARA 7.0 ITS Data Science Competition

---

## 📋 Deskripsi

Notebook ini berisi **Exploratory Data Analysis (EDA)** yang dilakukan terhadap dataset kompetisi **Data Science ARA 7.0 ITS** untuk tugas **semantic segmentation lubang jalan (pothole)**. Tujuan utama EDA ini adalah memahami karakteristik dataset secara mendalam sebelum membangun model segmentasi, sehingga setiap keputusan arsitektur dan augmentasi yang diambil memiliki dasar analitis yang kuat. Dalam kompetisi ini, peserta membangun model segmentasi lubang jalan dari citra permukaan jalan.

---

## 🗂️ Struktur Dataset

Dataset bersumber dari Kaggle Competition `data-science-ara-7-0` dan memiliki struktur sebagai berikut:

```
dataset/
├── train/
│   ├── images/          # Citra input RGB (format .jpg)
│   └── masks/           # Mask biner segmentasi pothole (format .jpg)
├── test/
│   └── images/          # Citra uji tanpa label (format .jpg)
└── sample_submission.csv
```

- **Training set:** ~350+ gambar berlabel
- **Test set:** ~325 gambar tanpa label
- **Task:** Binary semantic segmentation (pothole vs. background)

---

## 📦 Library yang Digunakan

| Library | Kegunaan |
|---|---|
| `numpy` | Operasi numerik dan array |
| `pandas` | Manipulasi data tabular |
| `matplotlib` | Visualisasi data dan gambar |
| `seaborn` | Visualisasi statistik |
| `PIL / Pillow` | Pembacaan dan manipulasi citra |
| `cv2 (OpenCV)` | Pemrosesan citra tingkat lanjut |
| `os` | Navigasi file system |

---

## 🔎 Alur Analisis EDA

### 1. 📁 Eksplorasi Struktur File
- Listing seluruh file dalam direktori input Kaggle
- Verifikasi ketersediaan data train, test, dan file submission
- Pengecekan jumlah total citra dan mask

### 2. 📐 Analisis Dimensi & Resolusi Gambar
- Membaca ukuran (width × height) setiap citra dan mask pada training set
- Menemukan **54 variasi ukuran gambar yang berbeda** dalam dataset
- Distribusi resolusi divisualisasikan menggunakan scatter plot dan histogram
- **Temuan:** Ketidakkonsistenan resolusi memerlukan strategi resize/tiling khusus

### 3. 🎨 Analisis Properti Piksel Citra
Setiap gambar training dianalisis untuk mengekstrak properti berikut:
- **Mean Brightness** — rata-rata kecerahan piksel
- **Std Brightness** — standar deviasi kecerahan (indikator variasi pencahayaan)
- **Mean Contrast** — kontras rata-rata gambar
- **Skewness** — kemiringan distribusi intensitas piksel
- **Kurtosis** — ketajaman puncak distribusi intensitas

Distribusi masing-masing properti divisualisasikan dengan histogram dan KDE plot.

### 4. 🕳️ Analisis Mask Segmentasi
Setiap mask dianalisis untuk mengekstrak:
- **Pixel Ratio** — persentase piksel pothole terhadap total piksel
- **Num Potholes** — jumlah lubang terpisah per gambar (menggunakan connected components)
- **Mask Coverage** — seberapa besar area yang tertutup lubang

**Temuan Kritis:**
- **64% gambar** mengandung lebih dari 1 lubang jalan per citra
- Rata-rata rasio piksel pothole sangat kecil dibandingkan latar belakang, mengindikasikan **class imbalance yang signifikan**

### 5. 📊 Analisis Distribusi Kelas (Class Imbalance)
- Perhitungan rasio foreground (pothole) vs. background (aspal) di seluruh dataset
- Visualisasi distribusi menggunakan bar chart dan pie chart
- **Temuan:** Dataset mengalami **class imbalance berat** — piksel background mendominasi secara signifikan

### 6. 🌈 Analisis Distribusi Warna & Kecerahan
- Analisis histogram RGB pada sampel gambar
- Visualisasi distribusi kecerahan keseluruhan dataset
- Korelasi antara kondisi pencahayaan (siang vs. senja/fajar) dengan kualitas citra
- **Temuan:** ~350 gambar berlatar siang hari, ~120 gambar berlatar senja/fajar

### 7. ⚠️ Identifikasi Tantangan Dataset
Berdasarkan analisis di atas, sistem otomatis mengidentifikasi tantangan dengan threshold tertentu:

| # | Tantangan | Tingkat Keparahan | Deskripsi | Rekomendasi |
|---|---|---|---|---|
| 1 | **Variable Lighting** | MEDIUM | Brightness CV: 0.38 (>0.3) | CLAHE, brightness/contrast augmentation |
| 2 | **Multiple Potholes per Image** | MEDIUM | 64% gambar memiliki >1 pothole | Model harus menangani multi-instance |
| 3 | **Inconsistent Image Dimensions** | LOW | 54 variasi ukuran gambar | Resize ke 512×512 atau tiling |

---

## 💡 Insight & Rekomendasi Augmentasi

Berdasarkan hasil EDA, berikut adalah rekomendasi augmentasi dan preprocessing yang disarankan:

### 🔆 Terkait Pencahayaan & Warna
1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** — meratakan distribusi intensitas piksel untuk meningkatkan kecerahan pada area gelap
2. **Gamma Correction Augmentation** — mensimulasikan kondisi pencahayaan ekstrem agar model robust di berbagai cuaca
3. **Color Jittering & Saturation Randomization** — menangani citra buram akibat cahaya terang berlebih
4. **Shadow Augmentation** — melatih model membedakan bayangan gelap dengan kedalaman lubang jalan
5. **Specular Highlight / Lens Flare Augmentation** — mengatasi pantulan cahaya pada permukaan aspal

### 🖼️ Terkait Resolusi & Skala
6. **Tiling / Multi-Scale Resizing** — memotong gambar beresolusi tinggi menjadi patch-patch kecil tanpa mengubah ukuran piksel, menjaga detail lubang kecil tetap tajam
7. **Multi-Scale Feature Extraction** — diperlukan karena pola persebaran resolusi tidak konsisten
8. **Random Cropping** — melatih model agar tidak salah mengartikan tekstur aspal alami sebagai lubang

### ⚖️ Terkait Class Imbalance
9. **Dice Loss / Focal Loss / Weighted BCE** — menangani ketidakseimbangan kelas yang berat antara piksel pothole dan background
10. **Class-Balanced Sampling** — memastikan model tidak bias dan mampu membedakan tekstur kasar alami dari pothole

### 🔬 Terkait Kualitas Citra
11. **Filter Median / Total Variation Denoising** — menghaluskan noise dari kompleksitas tekstur aspal
12. **Edge Sharpening Filter** — mempertajam tepi agar model lebih jelas membedakan batas lubang dengan latar belakang
13. **Local Contrast Enhancement / Gamma Correction** — memperjelas kedalaman lubang jalan
14. **Low-Light Synthetic Augmentation** — menangani variasi kondisi waktu pengambilan gambar (siang vs. senja/fajar)

### 📈 Terkait Evaluasi Model
15. **Stratified IoU (Stratified Error Analysis)** — menilai performa model berdasarkan: ukuran objek, kondisi pencahayaan, dan rasio aspek citra asli — bukan hanya metrik global
16. **Mask-to-Image Correlation Filter** — filter post-processing untuk mengurangi False Positive akibat bayangan; area dengan std dev piksel rendah (homogen) diklasifikasikan sebagai bayangan, area dengan std dev tinggi (kasar) diklasifikasikan sebagai pothole
17. **Global Value Normalization** — menstandarisasi variasi pencahayaan pada seluruh dataset, mengingat rentang intensitas yang signifikan (60–190)

---

## 📈 Ringkasan Statistik Dataset

| Metrik | Nilai |
|---|---|
| Total variasi ukuran gambar | 54 ukuran berbeda |
| Persentase gambar dengan >1 pothole | ~64% |
| Coefficient of Variation brightness | 0.38 (variasi tinggi) |
| Kondisi pencahayaan | ~350 siang, ~120 senja/fajar |
| Tingkat class imbalance | Berat (background >> pothole) |

---

## 🚀 Cara Menjalankan

1. **Clone repository ini** atau upload notebook ke Kaggle
2. Pastikan dataset `data-science-ara-7-0` tersedia di `/kaggle/input/`
3. Jalankan semua sel secara berurutan (Run All)
4. Seluruh output visualisasi dan insight akan ter-generate otomatis

**Environment yang direkomendasikan:**
- Python 3.12+
- Kaggle Kernel (CPU sudah cukup untuk EDA)
- RAM minimal 16 GB untuk memuat seluruh dataset

---

## 📁 File dalam Direktori Ini

```
EDA/
├── EDA FINAL ARA ITS.ipynb    # Notebook EDA utama
└── README.md                   # Dokumentasi ini
```

---

## 📝 Catatan

> EDA ini merupakan fondasi analitis sebelum tahap modeling. Seluruh keputusan arsitektur model, strategi loss function, dan pipeline augmentasi dalam tahap selanjutnya didasarkan pada temuan di sini.
