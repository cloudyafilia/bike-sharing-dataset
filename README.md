# Bike Sharing Dashboard

Dashboard ini dibuat menggunakan Streamlit untuk mengeksplorasi dan memvisualisasikan data peminjaman sepeda pada rentang tahun 2011-2012 yang dianalisis berdasarkan best and worst performing time periods, weather impact, dan customer demographic.

## Persyaratan
Sebelum menjalankan dashboard, pastikan Anda telah menginstal Python dan memiliki package yang diperlukan. Berikut adalah daftar package yang digunakan:

```bash
matplotlib==3.7.0
pandas==1.5.3
seaborn==0.13.0
streamlit==1.27.2
```

## Instalasi
1. **Setup Environment**
   Jika ingin menjalankan di dalam lingkungan virtual, buat dan aktifkan environment baru:
   
   ```bash
   python -m venv env
   source env/bin/activate  # Untuk macOS/Linux
   env\Scripts\activate     # Untuk Windows
   ```

2. **Instal Dependensi**
   
   ```bash
   pip install -r requirements.txt
   ```

   Jika file `requirements.txt` belum ada, Anda bisa menginstal package satu per satu dengan:

   ```bash
   pip install matplotlib pandas seaborn streamlit
   ```

## Run streamlit app
Setelah semua dependensi terinstal, jalankan perintah berikut untuk menjalankan dashboard:

```bash
streamlit run dashboard.py
```

Pastikan Anda berada di direktori yang berisi file `dashboard.py` sebelum menjalankan perintah ini.

## Struktur Direktori
```
📂 bike-rental-dashboard
├── 📄 app.py  # Skrip utama yang digunakan untuk membuat dashboard hasil analisis data di Streamlit
├── 📂 data    # Folder untuk menyimpan dataset dalam format .csv
└──📄 notebook.ipnyb  # File yang digunakan untuk melakukan analisis data
```



