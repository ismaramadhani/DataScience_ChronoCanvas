import re
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_data
def load_raw_data():
	base = Path(__file__).parent
	f1 = base / "data" / "dataset_ai_indonesia_500.csv"
	f2 = base / "data" / "dataset_ai_pengetahuan_umum.csv"

	df1 = pd.read_csv(f1, sep=';', usecols=['nama_kunci', 'mata_pelajaran', 'deskripsi', 'link_gambar'])
	df2 = pd.read_csv(f2, usecols=['nama_kunci', 'mata_pelajaran', 'deskripsi', 'link_gambar'])

	df = pd.concat([df1, df2], axis=0, ignore_index=True)
	return df


def bersihkan_teks_deskripsi(teks):
	if not isinstance(teks, str):
		return ""
	teks = teks.strip('"\'')
	teks = re.sub(r'\s+', ' ', teks)
	return teks.strip()


def upgrade_ke_gambar_hd(url):
	if not isinstance(url, str) or pd.isna(url):
		return url
	if "/commons/thumb/" in url:
		url_asli = url.rsplit('/', 1)[0]
		url_asli = url_asli.replace("/commons/thumb/", "/commons/")
		return url_asli
	return url


def clean_and_feature_engineer(df):
	df = df.dropna()
	df['deskripsi'] = df['deskripsi'].apply(bersihkan_teks_deskripsi)
	df = df.drop_duplicates(subset=['deskripsi'])
	df['link_gambar'] = df['link_gambar'].apply(upgrade_ke_gambar_hd)

	df['mata_pelajaran'] = df['mata_pelajaran'].astype(str).str.strip().str.title()
	kamus_kategori = {
		'Budaya Dan Sejarah': 'Sejarah & Pengetahuan Indonesia',
		'Spanyol & Belanda': 'Sejarah & Pengetahuan Indonesia',
		'Desa Dan Kelurahan': 'Pengetahuan Desa & Kampung',
		'Desa': 'Pengetahuan Desa & Kampung',
		'Kampung': 'Pengetahuan Desa & Kampung',
		'Perkebunan & Perikanan': 'Perkebunan & Perikanan',
		'Pengetahuan Umum & Sains': 'Pengetahuan Umum & Sains'
	}
	df['mata_pelajaran'] = df['mata_pelajaran'].replace(kamus_kategori)

	df = df.groupby('nama_kunci').agg(
		mata_pelajaran=('mata_pelajaran', 'first'),
		deskripsi=('deskripsi', lambda x: ' '.join(x.astype(str))),
		link_gambar=('link_gambar', 'first')
	).reset_index()

	df['panjang_deskripsi'] = df['deskripsi'].str.len()
	df['jumlah_kata'] = df['deskripsi'].apply(lambda x: len(str(x).split()))

	return df


def plot_count_mata_pelajaran(df):
	fig, ax = plt.subplots(figsize=(10, 6))
	order = df['mata_pelajaran'].value_counts().index
	sns.countplot(y='mata_pelajaran', data=df, order=order, palette='viridis', ax=ax)
	ax.set_xlabel('Jumlah Data')
	ax.set_ylabel('Mata Pelajaran')
	ax.set_title('Distribusi Mata Pelajaran')
	plt.tight_layout()
	return fig


def plot_box_panjang_deskripsi(df):
	fig, ax = plt.subplots(figsize=(12, 7))
	sns.boxplot(x='panjang_deskripsi', y='mata_pelajaran', data=df, palette='pastel', ax=ax)
	ax.set_title('Panjang Deskripsi Berdasarkan Mata Pelajaran')
	ax.set_xlabel('Panjang Karakter Deskripsi')
	ax.set_ylabel('Mata Pelajaran')
	plt.tight_layout()
	return fig


def plot_hist_image_usage(df):
	counts = df['link_gambar'].value_counts()
	fig, ax = plt.subplots(figsize=(10, 6))
	sns.histplot(counts, bins=20, kde=False, color='coral', ax=ax)
	ax.set_title('Distribusi Frekuensi Penggunaan Link Gambar')
	ax.set_xlabel('Jumlah Nama Kunci yang Menggunakan Link Gambar')
	ax.set_ylabel('Frekuensi Link Gambar')
	ax.set_yscale('log')
	plt.tight_layout()
	return fig, counts


def plot_top_duplicates_distribution(df, counts, top_n=5):
	top_links = counts[counts > 1].head(top_n).index
	df_top = df[df['link_gambar'].isin(top_links)]
	distribusi = df_top.groupby(['link_gambar', 'mata_pelajaran']).size().unstack(fill_value=0)
	fig = distribusi.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='viridis').get_figure()
	plt.title(f'Distribusi Top {top_n} Link Gambar yang Paling Duplikat berdasarkan Mata Pelajaran')
	plt.xlabel('Link Gambar')
	plt.ylabel('Jumlah Nama Kunci')
	plt.tight_layout()
	return fig, distribusi


def main():
	st.title('Dashboard: Insight & Kesimpulan — ChronoCanvas Dataset')
	st.markdown('Interactive dashboard yang merangkum insight dan kesimpulan dari notebook Anda.')

	df_raw = load_raw_data()
	df = clean_and_feature_engineer(df_raw.copy())

	# Sidebar
	st.sidebar.header('Filter & Pilihan')
	show_raw = st.sidebar.checkbox('Tampilkan data mentah', value=False)
	mata_options = st.sidebar.multiselect('Pilih Mata Pelajaran (opsional)', options=sorted(df['mata_pelajaran'].unique()), default=None)
	top_n = st.sidebar.slider('Top duplicated images (N)', min_value=1, max_value=20, value=5)

	if show_raw:
		st.subheader('Data Mentah (Sample)')
		st.dataframe(df_raw.head(200))

	if mata_options:
		df = df[df['mata_pelajaran'].isin(mata_options)]

	st.subheader('Ringkasan Data')
	col1, col2, col3 = st.columns(3)
	with col1:
		st.metric('Total Baris', f"{len(df):,}")
	with col2:
		st.metric('Unique link_gambar', f"{df['link_gambar'].nunique():,}")
	with col3:
		counts = df['link_gambar'].value_counts()
		duplicated_count = (counts > 1).sum()
		st.metric('Link Gambar Duplikat (>1)', f"{duplicated_count:,}")

	st.subheader('Visualisasi')
	fig1 = plot_count_mata_pelajaran(df)
	st.pyplot(fig1)

	fig2 = plot_box_panjang_deskripsi(df)
	st.pyplot(fig2)

	fig3, counts = plot_hist_image_usage(df)
	st.pyplot(fig3)

	if not counts[counts > 1].empty:
		st.subheader(f'Top {top_n} Link Gambar Duplikat')
		st.write(counts[counts > 1].head(top_n))
		fig4, distribusi = plot_top_duplicates_distribution(df, counts, top_n=top_n)
		st.pyplot(fig4)

	st.subheader('Insight (Ringkasan)')
	st.markdown('''
- Distribusi `mata_pelajaran` tidak merata; beberapa kategori mendominasi dataset.
- Terdapat variasi panjang deskripsi antar kategori; beberapa kategori menunjukkan outlier panjang deskripsi.
- Sebagian besar `link_gambar` hanya digunakan sekali, namun ada beberapa URL yang dipakai berulang (duplikasi).
- Duplikasi gambar cenderung terkonsentrasi pada beberapa URL tertentu dan seringkali hadir lintas kategori.
- Pembersihan termasuk upgrade URL thumbnail, penggabungan deskripsi per `nama_kunci`, dan standarisasi kategori telah dilakukan.
''')

	st.subheader('Kesimpulan & Rekomendasi')
	st.markdown('''
- Validasi gambar saat pengumpulan data disarankan untuk menghindari duplikasi visual yang menurunkan kualitas pelatihan model.
- Perlu penyeimbangan atau penyesuaian bobot saat menggunakan data untuk pelatihan karena ketidakseimbangan kategori `mata_pelajaran`.
- Periksa entri dengan deskripsi sangat panjang (outlier) sebelum menggunakan rata-rata sebagai metrik ringkasan.
''')

	st.caption('Notebook sumber: Salinan_dari_ChronoCanvas_fix.ipynb')


if __name__ == '__main__':
	main()
	

