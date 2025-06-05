import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from recommendation.recommend import rekomendasi_dari_produk
from recommendation.recommend import (
    rekomendasi_dari_query,
    get_data_and_model,
    bersihkan_html,
)

df, vectorizer, tfidf_matrix = get_data_and_model()

st.set_page_config(page_title="Rekomendasi Produk Halal", layout="wide")
st.title("🛒 Temukan Produk Halal Terbaik Anda!")

st.sidebar.header("Opsi Filter")

if "category_filter" not in st.session_state:
    st.session_state.category_filter = df["label"].unique()[0]
if "location_filter" not in st.session_state:
    st.session_state.location_filter = df["kota"].unique()[0]
if "min_price" not in st.session_state:
    st.session_state.min_price = int(df["harga"].min())
if "max_price" not in st.session_state:
    st.session_state.max_price = int(df["harga"].max())
if "rating_filter" not in st.session_state:
    st.session_state.rating_filter = 1.0

if st.sidebar.button("🔄 Reset Filter"):
    st.session_state.category_filter = df["label"].unique()[0]
    st.session_state.location_filter = df["kota"].unique()[0]
    st.session_state.min_price = int(df["harga"].min())
    st.session_state.max_price = int(df["harga"].max())
    st.session_state.rating_filter = 1.0
    st.experimental_rerun()

st.sidebar.subheader("Kategori & Lokasi")
category_filter = st.sidebar.selectbox(
    "Pilih Kategori",
    df["label"].unique(),
    index=list(df["label"].unique()).index(st.session_state.category_filter),
    key="category_filter",
)
location_filter = st.sidebar.selectbox(
    "Pilih Lokasi",
    df["kota"].unique(),
    index=list(df["kota"].unique()).index(st.session_state.location_filter),
    key="location_filter",
)

st.sidebar.subheader("Rentang Harga (Rp)")
min_price = st.sidebar.number_input(
    "Minimum Harga",
    min_value=int(df["harga"].min()),
    max_value=int(df["harga"].max()),
    value=st.session_state.min_price,
    key="min_price",
)
max_price = st.sidebar.number_input(
    "Maksimum Harga",
    min_value=int(df["harga"].min()),
    max_value=int(df["harga"].max()),
    value=st.session_state.max_price,
    key="max_price",
)

st.sidebar.subheader("Minimum Rating Produk")


def rating_to_stars(rating):
    full_stars = int(rating)
    empty_stars = 5 - full_stars
    stars = "⭐" * full_stars + "☆" * empty_stars
    return f"{stars} ({rating:.1f})"


rating_options = [1.0, 2.0, 3.0, 4.0, 5.0]
rating_filter = st.sidebar.radio(
    "Pilih Rating Minimal",
    rating_options,
    format_func=lambda x: rating_to_stars(x),
    index=rating_options.index(st.session_state.rating_filter),
    key="rating_filter",
)


def generate_star_rating(rating):
    full_stars = int(rating)
    half_star = rating - full_stars >= 0.25 and rating - full_stars < 0.75
    empty_stars = 5 - full_stars - (1 if half_star else 0)
    stars = "⭐" * full_stars
    if half_star:
        stars += "✬"
    stars += "☆" * empty_stars
    return stars


def display_product_card(col, row):
    with col:
        harga_asli = row["harga"]
        diskon = row["diskon"]
        harga_setelah_diskon = int(harga_asli * (1 - diskon / 100))
        cleaned_description = bersihkan_html(row["deskripsi"])

        st.markdown(
            f"""
            <div style="border: 1px solid #e0e0e0; border-radius: 10px; padding: 10px; margin: 6px; background-color: #fafafa;
                        text-align: center; box-shadow: 1px 1px 5px rgba(0,0,0,0.1); width: 100%; height: auto;
                        display: flex; flex-direction: column; justify-content: flex-start; max-height: 350px; overflow: hidden;">
                <div style="text-align: center;">
                    <img src="{row['gambar']}" width="130" height="130" style="object-fit: cover; border-radius: 8px;">
                </div>
                <div style="margin-top: 6px;">
                    <div style="font-weight: bold; font-size: 15px; height: 55px; overflow: hidden; text-overflow: ellipsis; color: #333;">
                        {row['nama_produk']}
                    </div>
                </div>
                <div style="margin-top: 4px;">
                    <div style="color: #888; font-size: 11px;">
                        <s>Rp{harga_asli:,.0f}</s>
                    </div>
                    <div style="color: #388e3c; font-size: 11px; font-weight: bold;">
                        Diskon {diskon}%
                    </div>
                    <div style="color: #d32f2f; font-weight: bold; font-size: 15px;">
                        Rp{harga_setelah_diskon:,.0f}
                    </div>
                    <div style="font-size: 13px; color: #f39c12; margin-top: 5px;">
                        {generate_star_rating(row['rating'])} <span style="color: #555; font-size: 11px;">({row['rating']:.1f})</span>
                    </div>
                    <div style="font-size: 12px; color: #555; margin-bottom: 4px; margin-top: 3px;"><span style="font-size: 15px;">📍</span> {row['kota']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("📄 Selengkapnya"):
            st.markdown(f"**{row['nama_produk']}**")
            st.write(cleaned_description)
        st.write("---")


st.markdown("---")
st.header("🔍 Cari Produk Impian Anda")
query = st.text_input(
    "Ketik kata kunci produk (contoh: 'susu segar', 'keripik renyah')",
    key="product_query",
)

if st.button("Cari Produk", type="primary"):
    with st.spinner("Mencari rekomendasi produk..."):
        filtered_df = df[
            (df["label"] == category_filter)
            & (df["kota"] == location_filter)
            & (df["harga"] >= min_price)
            & (df["harga"] <= max_price)
            & (df["rating"] >= rating_filter)
        ]

        if query:
            recommendations = rekomendasi_dari_query(
                query, df, tfidf_matrix, vectorizer
            )
            recommendations = recommendations[
                (recommendations["kota"] == location_filter)
                & (recommendations["harga"] >= min_price)
                & (recommendations["harga"] <= max_price)
                & (recommendations["rating"] >= rating_filter)
            ]
        else:
            recommendations = filtered_df

        if recommendations.empty:
            st.warning(
                "Mohon maaf, tidak ada produk yang cocok dengan kriteria Anda. Coba sesuaikan filter atau kata kunci pencarian Anda."
            )
        else:
            st.subheader("Rekomendasi Produk Halal Pilihan Untuk Anda:")
            cols = st.columns(3)
            for index, row in recommendations.iterrows():
                display_product_card(cols[index % 3], row)

st.markdown("---")

st.header("✨ Temukan Produk Serupa dari Pilihan Anda")
selected_product = st.selectbox(
    "Pilih produk yang ingin Anda cari produk serupa dengannya:",
    df["nama_produk"].unique(),
    key="similar_product_select",
)

if st.button("Cari Produk Serupa", type="primary"):
    with st.spinner(f"Mencari produk serupa dengan **{selected_product}**..."):
        rekomendasi_produk_serupa = rekomendasi_dari_produk(
            selected_product, df, tfidf_matrix
        )

        if rekomendasi_produk_serupa.empty:
            st.info(
                "Tidak ada produk serupa yang ditemukan untuk pilihan Anda. Coba pilih produk lain."
            )
        else:
            st.subheader(f"Produk Serupa dengan: **{selected_product}**")
            cols = st.columns(3)
            for index, row in rekomendasi_produk_serupa.iterrows():
                display_product_card(cols[index % 3], row)
