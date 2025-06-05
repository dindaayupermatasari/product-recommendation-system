import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
from bs4 import BeautifulSoup  # ← Tambahkan ini

# Load data dan model
df = pd.read_csv("data/final_clean_data.csv")

with open("model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)


def clean_text(text):
    """Membersihkan teks dari tanda baca dan huruf kapital"""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def bersihkan_html(teks):
    """Menghapus tag HTML dari deskripsi"""
    return BeautifulSoup(teks, "html.parser").get_text()


def rekomendasi_dari_query(
    query, df, tfidf_matrix, vectorizer, threshold=0.25, top_n=30
):
    """Mengembalikan DataFrame rekomendasi berdasarkan query"""

    # Bersihkan input query
    query_clean = clean_text(query)

    # Vektorisasi dan hitung cosine similarity
    query_vec = vectorizer.transform([query_clean])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Ambil top_n index kemiripan tertinggi
    top_indices = cosine_sim.argsort()[::-1][:top_n]
    top_scores = cosine_sim[top_indices]

    # Buat DataFrame hasil
    hasil = df.iloc[top_indices].copy()
    hasil["skor_kemiripan"] = top_scores

    # Filter berdasarkan threshold kemiripan
    hasil = hasil[hasil["skor_kemiripan"] >= threshold]

    if hasil.empty:
        return pd.DataFrame(
            columns=[
                "nama_produk",
                "harga",
                "kota",
                "rating",
                "diskon",
                "deskripsi",
                "gambar",
            ]
        )

    # Bersihkan deskripsi dari tag HTML
    hasil["deskripsi"] = hasil["deskripsi"].apply(bersihkan_html)

    return hasil[
        ["nama_produk", "harga", "kota", "rating", "diskon", "deskripsi", "gambar"]
    ]


def rekomendasi_dari_produk(nama_produk, df, tfidf_matrix, top_n=10):
    """Mengembalikan produk-produk yang mirip dengan nama produk yang dipilih"""

    if nama_produk not in df["nama_produk"].values:
        return pd.DataFrame(columns=df.columns)

    idx = df[df["nama_produk"] == nama_produk].index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Ambil top_n produk mirip, selain dirinya sendiri
    similar_indices = cosine_sim.argsort()[::-1][1 : top_n + 1]
    similar_scores = cosine_sim[similar_indices]

    hasil = df.iloc[similar_indices].copy()
    hasil["skor_kemiripan"] = similar_scores
    hasil["deskripsi"] = hasil["deskripsi"].apply(bersihkan_html)

    return hasil[
        ["nama_produk", "harga", "kota", "rating", "diskon", "deskripsi", "gambar"]
    ]


def get_data_and_model():
    return df, vectorizer, tfidf_matrix
