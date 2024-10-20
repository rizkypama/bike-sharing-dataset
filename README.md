# Analisis Penggunaan Bike Sharing ✨

Aplikasi ini menggunakan data penggunaan bike sharing untuk melakukan analisis dan visualisasi terkait tren penggunaan bike sharing, faktor-faktor cuaca yang mempengaruhi penggunaan, dan klastering penggunaan berdasarkan suhu dan jumlah penggunaan. Berikut adalah dokumentasi lengkap untuk aplikasi ini:

1. Analisis Tren Penggunaan Bike Sharing
Aplikasi ini memberikan visualisasi tentang tren penggunaan bike sharing per hari selama rentang waktu yang dipilih oleh pengguna. Grafik linier menunjukkan total penggunaan bike sharing per hari.

2. Analisis Faktor Cuaca
Aplikasi ini memungkinkan pengguna untuk melihat distribusi penggunaan bike sharing berdasarkan faktor cuaca seperti musim, kondisi cuaca, hari biasa/libur, dll. Visualisasi yang disediakan meliputi diagram batang untuk musim, kondisi cuaca, hari biasa/libur, dan lainnya.

3. Heatmap Korelasi
Heatmap korelasi memperlihatkan hubungan antara suhu, kelembaban, kecepatan angin, kondisi cuaca, dan jumlah penggunaan bike sharing. Hal ini membantu pengguna untuk memahami seberapa kuat hubungan antara faktor-faktor cuaca dengan penggunaan bike sharing.

4. Klastering Penggunaan Bike Sharing Berdasarkan Temperatur dan Jumlah Penggunaan
Pengguna dapat menentukan jumlah klaster yang diinginkan dan aplikasi akan melakukan klastering penggunaan bike sharing berdasarkan suhu dan jumlah penggunaan. Visualisasi scatterplot menunjukkan klastering penggunaan bike sharing dengan warna yang berbeda untuk setiap klaster.

5. Informasi Tambahan
Aplikasi ini juga menyediakan informasi tambahan seperti total penggunaan bike sharing, rata-rata penggunaan per hari, distribusi temperatur, distribusi kondisi cuaca, dan sebagainya.

Teknologi yang Digunakan
Aplikasi ini dibangun menggunakan Streamlit, sebuah framework Python untuk membuat aplikasi web interaktif dengan cepat. Analisis data dan visualisasi dilakukan menggunakan pandas, numpy, matplotlib, seaborn, dan scikit-learn.

## Setup environment
```
conda create --name yourname-ds python=3.9
conda activate yourname-ds
pip install -r requirements.txt
```

## Run steamlit app 🚀
```
cd dashboard
streamlit run dashboard.py
```
"# bike-sharing-dataset" 
