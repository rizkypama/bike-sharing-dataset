# Import library yang diperlukan
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.cluster import KMeans

# Mengatur style plot seaborn
sns.set(style='dark')

# Fungsi untuk menghasilkan titik-titik pusat klaster secara acak
def random_centers(dim,k):
    centers = []
    for i in range(k):
        center = []
        for d in range(dim):
            rand = random.randint(0,100)
            center.append(rand)
        centers.append(center)
    return centers

# Fungsi untuk mengelompokkan titik data ke klaster terdekat
def point_clustering(data, centers, dims, first_cluster=False):
    for point in data:
        nearest_center = 0
        nearest_center_dist = None
        for i in range(0, len(centers)):
            euclidean_dist = 0
            for d in range(0, dims):
                dist = abs(point[d] - centers[i][d])
                euclidean_dist += dist
            euclidean_dist = np.sqrt(euclidean_dist)
            if nearest_center_dist == None:
                nearest_center_dist = euclidean_dist
                nearest_center = i
            elif nearest_center_dist > euclidean_dist:
                nearest_center_dist = euclidean_dist
                nearest_center = i
        if first_cluster:
            point.append(nearest_center)
        else:
            point[-1] = nearest_center
    return data

# Fungsi untuk menghitung pusat klaster baru
def mean_center(data, centers, dims):
    new_centers = []
    for i in range(len(centers)):
        new_center = []
        n_of_points = 0
        total_of_points = []
        for point in data:
            if point[-1] == i:
                n_of_points += 1
                for dim in range(0,dims):
                    if dim < len(total_of_points):
                        total_of_points[dim] += point[dim]
                    else:
                        total_of_points.append(point[dim])
        if len(total_of_points) != 0:
            for dim in range(0,dims):
                new_center.append(total_of_points[dim]/n_of_points)
            new_centers.append(new_center)
        else: 
            new_centers.append(centers[i])
            
    return new_centers

# Fungsi untuk melatih model klastering K-Means
def train_k_means_clustering(data, k=2, epochs=5):
    dims = len(data[0])
    centers = random_centers(dims,k)
    clustered_data = point_clustering(data, centers, dims, first_cluster=True)
    for i in range(epochs):
        centers = mean_center(clustered_data, centers, dims)
        clustered_data = point_clustering(data, centers, dims, first_cluster=False)
    return centers

# Fungsi untuk memprediksi klaster titik data baru
def predict_k_means_clustering(point, centers):
    dims = len(point)
    center_dims = len(centers[0])
    if dims != center_dims:
        raise ValueError('Point yang diberikan untuk prediksi memiliki' + dims + 'dimensi sedangkan centers mempunyai' + center_dims + 'dimensi')
    nearest_center = None
    nearest_dist = None
    for i in range(len(centers)):
        euclidean_dist = 0
        for dim in range(1, dims):
            dist = point[dim] - centers[i][dim]
            euclidean_dist += dist**2
        euclidean_dist = np.sqrt(euclidean_dist)
        if nearest_dist == None:
            nearest_dist = euclidean_dist
            nearest_center = i
        elif nearest_dist > euclidean_dist:
            nearest_dist = euclidean_dist
            nearest_center = i
    return nearest_center

# Memuat dataset
df = pd.read_csv('dashboard/day.csv')
datetime_columns = ["dteday"]
df.sort_values(by="dteday", inplace=True)
df.reset_index(inplace=True)
for column in datetime_columns:
    df[column] = pd.to_datetime(df[column])

# Menentukan rentang waktu pada sidebar
min_date = df["dteday"].min()
max_date = df["dteday"].max()
with st.sidebar:
    st.title("Analisis Data Bike Sharing")
    st.image("dashboard/logo.png")
    st.subheader('Rentang Waktu')
    start_date = st.date_input(label="Tanggal Mulai", value=min_date, max_value=max_date)
    end_date = st.date_input(label="Tanggal Selesai", value=max_date, max_value=max_date)

# Mengambil data sesuai rentang waktu yang ditentukan
main_df = df[(df["dteday"] >= str(start_date)) & (df["dteday"] <= str(end_date))]

# Menampilkan metrik penggunaan bike sharing
st.subheader('Tren Penggunaan Bike Sharing per Hari')
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Penggunaan Bike Sharing", value=main_df['cnt'].sum())
with col2:
    st.metric("Rata-rata Penggunaan Bike Sharing per Hari", value=main_df['cnt'].mean().round(2))
    
# Menampilkan grafik tren penggunaan bike sharing per hari
daily_data = main_df.groupby('dteday').sum()['cnt']
st.line_chart(daily_data)

# Menampilkan distribusi temperatur dan kondisi cuaca
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.hist(data=main_df, x='temp', bins=20)
    plt.xlabel('Temperature (Celsius)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Temperature')
    st.pyplot(fig)

with col2:
    weather_counts = main_df['weathersit'].value_counts()
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.pie(weather_counts, labels=['Clear', 'Mist', 'Light Rain/Snow'], autopct='%1.1f%%', startangle=140, colors=['lightgreen', 'skyblue', 'salmon', 'gray'])
    plt.title('Distribution of Weather Conditions')
    st.pyplot(fig)

# Menampilkan distribusi penggunaan bike sharing berdasarkan faktor-faktor tertentu
st.subheader('Distribusi Penggunaan Bike Sharing')
tab1, tab2, tab3, tab4 = st.tabs(["Musim", "Cuaca", "Hari Biasa", "Hari Libur"])
with tab1:
    plot = sns.catplot(x='season', data=main_df, kind='count', aspect=1.5)
    st.write("1 - Musim Semi")
    st.write("2 - Musim Panas")
    st.write("3 - Musim Gugur")
    st.write("4 - Musim Dingin")
    st.pyplot(plot)
    
with tab2:
    plot = sns.catplot(x='weathersit', data=main_df, kind='count', aspect=1.5)
    st.write("1 - Cerah, Sedikit Berawan")
    st.write("2 - Berkabut, Berawan")
    st.write("3 - Hujan atau Salju Ringan")
    st.write("4 - Badai, Salju")
    st.pyplot(plot)
    
with tab3:
    plot = sns.catplot(x='workingday', data=main_df, kind='count', aspect=1.5)
    st.write("0 - Weekend / Libur")
    st.write("1 - Hari Biasa")
    st.pyplot(plot)

with tab4:
    plot = sns.catplot(x='holiday', data=main_df, kind='count', aspect=1.5)
    st.write("0 - Bukan Hari Libur")
    st.write("1 - Hari Libur")
    st.pyplot(plot)

# Menampilkan heatmap korelasi
st.subheader('Heatmap Korelasi')
data_heatmap = main_df[['temp', 'hum', 'windspeed', 'weathersit', 'cnt']]
correlation_matrix = data_heatmap.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap Korelasi:\n Temperatur, Kelembaban, Kecepatan Angin, Cuaca dan Penggunaan Bike Sharing')
st.pyplot(plt)

# Klastering penggunaan bike sharing berdasarkan temperatur dan jumlah penggunaan
st.subheader('Klastering Penggunaan Bike Sharing Berdasarkan Temperatur dan Jumlah Penggunaan')

# Slider untuk memilih jumlah klaster
cluster = st.slider(
        label='Pilih jumlah klaster',
        min_value=0, max_value=10, value=(3)
    )

# Mengambil fitur untuk klastering
X = main_df[['temp', 'cnt']]
plotx = []
ploty = []
for i in range(len(X)):
    plotx.append(X.iloc[i, 0])
    ploty.append(X.iloc[i, 1])

# Melatih model klastering K-Means
X_train = [[temp, cnt] for temp, cnt in zip(plotx, ploty)]
centers = train_k_means_clustering(X_train, k=cluster, epochs=5)

# Melakukan prediksi klaster untuk setiap titik data
X_pred = main_df[['temp', 'cnt']]
main_df['Cluster'] = [predict_k_means_clustering(point, centers) for point in X_pred.values]

# Visualisasi klastering
plt.figure(figsize=(10, 6))
sns.scatterplot(data=main_df, x='temp', y='cnt', hue='Cluster', palette='flare')
plt.title('Klastering Sederhana: Temperatur dan Jumlah Penggunaan Bike Sharing')
plt.xlabel('Temperatur (Celsius)')
plt.ylabel('Frekuensi')
plt.grid(True)

# Menampilkan plot
st.pyplot(plt)

st.caption('© 2024 - Analisis Data Bike Sharing by Rizkiyan Tri Ade Pama with ❤️ and ☕')