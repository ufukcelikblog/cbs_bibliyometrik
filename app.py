import math
import re
import unicodedata
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.lda_model
import pyLDAvis.utils
import scipy.cluster.hierarchy as sch
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import jensenshannon, squareform
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    TfidfVectorizer,
)
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

SEED = 42

# Alan-genel sözcükler: tüm derlemde geçtikleri için konuları ayrıştırmaz
DOMAIN_STOP = {
    "data", "gis", "using", "based", "study", "analysis", "research",
    "information", "paper", "results", "method", "approach", "used",
    "use", "proposed", "model", "system", "systems", "geographic",
}
STOP_WORDS = list(ENGLISH_STOP_WORDS.union(DOMAIN_STOP))


# ============================================================================
# YARDIMCI FONKSİYONLAR
# ============================================================================
@st.cache_resource
def download_nltk_data():
    for resource in ["punkt", "punkt_tab", "stopwords"]:
        try:
            nltk.download(resource, quiet=True)
        except (LookupError, OSError, RuntimeError, ValueError) as e:
            st.error(f"NLTK kaynağı indirilemedi ({resource}): {e}")
    return True


download_nltk_data()


@st.cache_resource
def get_nltk_stopwords():
    return set(stopwords.words("english"))


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    stop = get_nltk_stopwords()
    tokens = [t for t in word_tokenize(text) if t not in stop]
    return " ".join(tokens)


def normalize_text(text):
    if pd.isna(text):
        return text
    text = unicodedata.normalize("NFKD", str(text))
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201C\u201D]", '"', text)
    text = re.sub(r"[\r\n\t\xa0]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_data(show_spinner="Metinler ön işleniyor...")
def preprocess_series(s: pd.Series) -> pd.Series:
    return s.apply(normalize_text).apply(preprocess_text)


@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    if file.name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return None


def parse_year_or_date(series):
    numeric = pd.to_numeric(series, errors="coerce")
    year_mask = numeric.between(1000, 2200)
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if year_mask.any():
        result.loc[year_mask] = pd.to_datetime(
            numeric.loc[year_mask].astype("Int64").astype(str),
            format="%Y",
            errors="coerce",
        )
    remaining = ~year_mask
    if remaining.any():
        result.loc[remaining] = pd.to_datetime(series.loc[remaining], errors="coerce")
    return result


def top_words(components, feature_names, n=10):
    return [[feature_names[i] for i in comp.argsort()[: -n - 1 : -1]] for comp in components]


def topics_table(components, feature_names, labels, n=10):
    return pd.DataFrame(
        {"Konu": labels, "Anahtar kelimeler": [", ".join(w) for w in top_words(components, feature_names, n)]}
    )


def plot_topic_wordclouds(components, feature_names, labels, n_words=20, ncols=3):
    n = len(components)
    nrows = math.ceil(n / ncols)
    fig = plt.figure(figsize=(6.5 * ncols, 3.4 * nrows))
    for idx, topic in enumerate(components):
        idx_top = topic.argsort()[: -n_words - 1 : -1]
        freq = {feature_names[i]: float(topic[i]) for i in idx_top}
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        wc = WordCloud(width=800, height=400, background_color="white", max_words=50).generate_from_frequencies(freq)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(labels[idx])
        ax.axis("off")
    fig.tight_layout()
    return fig


def yearly_topic_means(theta, labels, years, min_docs):
    tdf = pd.DataFrame(theta, columns=labels)
    tdf["year"] = np.asarray(years)
    counts = tdf["year"].value_counts()
    valid = counts[counts >= min_docs].index
    yearly = tdf[tdf["year"].isin(valid)].groupby("year").mean().sort_index()
    dropped = sorted(set(counts.index) - set(valid))
    return yearly, dropped


def show_time_analysis(theta, labels, years, min_docs):
    yearly, dropped = yearly_topic_means(theta, labels, years, min_docs)
    if yearly.empty:
        st.warning(f"⚠️ Yıl başına ≥ {min_docs} doküman koşulunu sağlayan yıl yok. Eşiği düşürün.")
        return
    st.subheader("📆 Yıllara Göre Ortalama Konu Dağılımı")
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette("husl", n_colors=yearly.shape[1])
    yearly.plot(ax=ax, linewidth=2, color=palette)
    ax.set_title("Ortalama Konu Dağılımı Zaman İçinde")
    ax.set_xlabel("Yıl")
    ax.set_ylabel("Ortalama Konu Ağırlığı")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("📊 Yığılmış Çubuk Grafik")
    ax = yearly.plot(kind="bar", stacked=True, figsize=(14, 7), colormap="tab20")
    ax.set_title(f"Yıllara Göre Ortalama Konu Oranı (yıl başına ≥ {min_docs} doküman)")
    ax.set_xlabel("Yıl")
    ax.set_ylabel("Konu oranı")
    ax.legend(title="Konu", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    st.pyplot(ax.figure)
    plt.close(ax.figure)
    if dropped:
        st.caption(f"Çıkarılan yıllar (doküman sayısı < {min_docs}): {dropped}")


def patch_pyldavis_encoder():
    """pyLDAvis JSON kodlayıcısının karmaşık sayıları (MDS çıktısı) işlemesini sağlar (bir kez uygulanır)."""
    enc = pyLDAvis.utils.NumPyEncoder
    if getattr(enc, "_complex_patched", False):
        return
    original = enc.default

    def custom_default(self, obj):
        if isinstance(obj, (complex, np.complexfloating)):
            return float(obj.real)
        return original(self, obj)

    enc.default = custom_default
    enc._complex_patched = True


# ============================================================================
# ÖNBELLEKLİ MODEL FONKSİYONLARI
# ============================================================================
@st.cache_resource(show_spinner=False)
def fit_lda(texts, n_topics, min_df, max_df, max_iter):
    vec = CountVectorizer(stop_words=STOP_WORDS, min_df=min_df, max_df=max_df)
    X = vec.fit_transform(texts)
    model = LatentDirichletAllocation(
        n_components=n_topics, random_state=SEED, learning_method="batch", max_iter=max_iter
    )
    out = model.fit_transform(X)
    return vec, X, model, out


@st.cache_resource(show_spinner=False)
def fit_nmf(texts, n_topics, min_df, max_df):
    vec = TfidfVectorizer(stop_words=STOP_WORDS, min_df=min_df, max_df=max_df)
    tfidf = vec.fit_transform(texts)
    model = NMF(n_components=n_topics, random_state=SEED, init="nndsvd", max_iter=500)
    out = model.fit_transform(tfidf)
    return vec, model, out


@st.cache_resource(show_spinner=False)
def fit_ctm(texts, k, min_df, rm_top, n_iter):
    import tomotopy as tp

    stop = set(STOP_WORDS)
    ctm = tp.CTModel(tw=tp.TermWeight.ONE, min_df=min_df, rm_top=rm_top, k=k, seed=SEED)
    for t in texts:
        toks = [w for w in t.split() if w not in stop]
        if toks:
            ctm.add_doc(toks)
    ctm.burn_in = 100
    for _ in range(0, n_iter, 10):
        ctm.train(10)
    corr = np.array([ctm.get_correlations(i) for i in range(ctm.k)])
    topics = [[w for w, _ in ctm.get_topic_words(i, top_n=10)] for i in range(ctm.k)]
    return corr, topics, float(ctm.ll_per_word)


@st.cache_resource(show_spinner=False)
def fit_bertopic(texts, n_topics):
    from bertopic import BERTopic
    from umap import UMAP

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED)
    model = BERTopic(nr_topics=n_topics, umap_model=umap_model)
    topics, _ = model.fit_transform(list(texts))
    return model, topics


def compute_k_scores(texts, ks, min_df, max_df, max_iter, progress):
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel

    vec = CountVectorizer(stop_words=STOP_WORDS, min_df=min_df, max_df=max_df)
    X = vec.fit_transform(texts)
    names = vec.get_feature_names_out()
    tokenized = [t.split() for t in texts]
    dictionary = Dictionary(tokenized)
    dictionary.filter_extremes(no_below=min_df, no_above=max_df)

    rows = []
    for i, k in enumerate(ks):
        m = LatentDirichletAllocation(n_components=k, random_state=SEED, learning_method="batch", max_iter=max_iter)
        m.fit(X)
        topics = top_words(m.components_, names, 10)
        topics = [[w for w in t if w in dictionary.token2id] for t in topics]
        topics = [t for t in topics if t]
        cm = CoherenceModel(topics=topics, texts=tokenized, dictionary=dictionary, coherence="c_v", processes=1)
        rows.append({"K": k, "coherence_cv": cm.get_coherence(), "perplexity": m.perplexity(X)})
        progress.progress((i + 1) / len(ks), text=f"K={k} tamamlandı")
    return pd.DataFrame(rows)


# ============================================================================
# SAYFA VE YAN PANEL
# ============================================================================
st.set_page_config(page_title="Konu Modelleme Uygulaması", layout="wide")
st.title("🧠 Kapsamlı Konu Modelleme Uygulaması")
st.sidebar.header("Yükleme ve Ayarlar")

if "df" not in st.session_state:
    st.session_state.df = None

uploaded_file = st.sidebar.file_uploader("Dosya yükleme (.csv veya .xlsx)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        st.session_state.df = load_data(uploaded_file)
        st.sidebar.success("Dosya başarıyla yüklendi!")
    except (ValueError, TypeError, OSError, UnicodeError) as e:
        st.error(f"Dosya okunamadı: {e}")
        st.stop()

if st.sidebar.button("Ön tanımlı veriyi yükle", type="primary"):
    try:
        st.session_state.df = pd.read_csv("data.csv")
        st.sidebar.info("Önceden tanımlanmış veri yüklendi.")
    except (ValueError, TypeError, OSError, UnicodeError) as e:
        st.error(f"Ön tanımlı veri okunamadı: {e}")

if st.session_state.df is None:
    st.info("Başlamak için bir dosya yükleyin veya önceden tanımlanmış veri kullanın.")
    st.stop()

df = st.session_state.df.copy()
st.sidebar.success(f"{len(df):,} satır, {len(df.columns)} sütun yüklendi.")

use_existing = False
if "processed_texts" in df.columns:
    use_existing = st.sidebar.checkbox("Hazır 'processed_texts' sütununu kullan", value=True)

text_column = st.sidebar.selectbox(
    "Metin sütununu seçin (ham metin)",
    index=max(len(df.columns) - 2, 0),
    options=df.columns,
    disabled=use_existing,
)
date_column = st.sidebar.selectbox(
    "Tarih/Yıl sütununu seçin (isteğe bağlı)",
    index=1,
    options=[None] + list(df.columns),
)

model_choice = st.sidebar.selectbox(
    "Analiz / Yöntem", ["EDA", "K Seçimi", "LDA", "NMF", "CTM", "BERTopic"]
)
n_topics = st.sidebar.slider("Konu Sayısı (K)", 2, 20, 12)

st.sidebar.subheader("Sözlük Budama")
min_df = st.sidebar.number_input("min_df (en az kaç dokümanda geçmeli)", 1, 100, 5)
max_df = st.sidebar.slider("max_df (dokümanların en çok bu oranında geçmeli)", 0.1, 1.0, 0.5, 0.05)
lda_max_iter = st.sidebar.number_input("LDA max_iter", 10, 200, 50, step=10)
min_docs = st.sidebar.number_input("Zaman analizi: yıl başına min. doküman", 1, 500, 20)

st.subheader("📄 Veri Önizleme")
st.dataframe(df.head(20), width="stretch")

# ============================================================================
# VERİ HAZIRLAMA
# ============================================================================
if date_column:
    df[date_column] = parse_year_or_date(df[date_column])
    df = df.dropna(subset=[date_column]).copy()
    df["year"] = df[date_column].dt.year.astype(int)
    st.subheader("📅 Yıl Dağılımı")
    st.bar_chart(df["year"].value_counts().sort_index())

if use_existing:
    df = df.dropna(subset=["processed_texts"]).copy()
    df["processed_texts"] = df["processed_texts"].astype(str)
else:
    df[text_column] = df[text_column].replace("[NO ABSTRACT AVAILABLE]", pd.NA)
    df = df.dropna(subset=[text_column]).copy()
    df["processed_texts"] = preprocess_series(df[text_column].astype(str))

df = df[df["processed_texts"].str.strip() != ""].reset_index(drop=True)
if df.empty:
    st.error("İşlenebilir metin kalmadı. Metin sütununu kontrol edin.")
    st.stop()

texts = tuple(df["processed_texts"].tolist())
years = df["year"].to_numpy() if date_column else None
labels = [f"K{i + 1}" for i in range(n_topics)]

# ============================================================================
# EDA
# ============================================================================
if model_choice == "EDA":
    st.subheader("📊 Keşifçi Veri Analizi")
    st.write(f"🗋 Doküman: {df.shape[0]:,}, 📚 Sütun: {df.shape[1]}")

    df["text_length"] = df["processed_texts"].str.split().str.len()
    st.markdown("**Metin Uzunluğu İstatistikleri (işlenmiş metin, kelime):**")
    st.write(df["text_length"].describe())

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df["text_length"], bins=30, kde=True, color="teal", ax=ax)
    ax.set_title("Belgelerin Uzunluk Dağılımı")
    ax.set_xlabel("Kelime Sayısı")
    ax.set_ylabel("Frekans")
    st.pyplot(fig)
    plt.close(fig)

    stop_set = set(STOP_WORDS)
    word_freq = Counter(w for t in texts for w in t.split() if w not in stop_set)
    st.markdown("**İlk 20 En Sık Kelime (durak sözcükler hariç):**")
    st.dataframe(pd.DataFrame(word_freq.most_common(20), columns=["Kelime", "Frekans"]))

    st.markdown("**Kapsamlı Kelime Bulutu:**")
    wc = WordCloud(width=1600, height=800, background_color="white", max_words=200, colormap="viridis")
    wc = wc.generate_from_frequencies(word_freq)
    fig_wc, ax_wc = plt.subplots(figsize=(14, 7))
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc)
    plt.close(fig_wc)

    st.markdown("---")
    st.markdown("✅ Ardından **K Seçimi** ile konu sayısını belirleyip bir yöntem seçebilirsiniz.")

# ============================================================================
# K SEÇİMİ
# ============================================================================
elif model_choice == "K Seçimi":
    st.subheader("🎯 Konu Sayısı Seçimi (c_v tutarlılık + perplexity)")
    k_lo, k_hi = st.slider("K aralığı", 2, 30, (4, 18))
    k_step = st.selectbox("Adım", [1, 2, 3], index=1)

    if st.button("Hesapla", type="primary"):
        ks = list(range(k_lo, k_hi + 1, k_step))
        progress = st.progress(0.0, text="Başlıyor...")
        try:
            st.session_state.k_scores = compute_k_scores(texts, ks, min_df, max_df, lda_max_iter, progress)
        except ImportError:
            st.error("Bu analiz için `gensim` gerekli: pip install gensim")
        except ValueError as e:
            st.error(f"Hesaplama başarısız: {e}")
        progress.empty()

    k_scores = st.session_state.get("k_scores")
    if k_scores is not None:
        k_coh = int(k_scores.loc[k_scores["coherence_cv"].idxmax(), "K"])
        k_per = int(k_scores.loc[k_scores["perplexity"].idxmin(), "K"])
        c1, c2 = st.columns(2)
        c1.metric("En yüksek tutarlılık (c_v)", f"K = {k_coh}")
        c2.metric("En düşük perplexity", f"K = {k_per}")

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(k_scores["K"], k_scores["coherence_cv"], marker="o")
        ax[0].set_title("Tutarlılık (c_v) – yüksek daha iyi")
        ax[0].set_xlabel("Konu sayısı (K)")
        ax[1].plot(k_scores["K"], k_scores["perplexity"], marker="o", color="tab:red")
        ax[1].set_title("Perplexity – düşük daha iyi")
        ax[1].set_xlabel("Konu sayısı (K)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(k_scores, width="stretch")
        st.download_button("K skorlarını CSV indir", k_scores.to_csv(index=False).encode("utf-8"), "k_secimi.csv", "text/csv")
        st.info("Seçtiğiniz K değerini yan paneldeki **Konu Sayısı** kaydırıcısından ayarlayın.")

# ============================================================================
# LDA
# ============================================================================
elif model_choice == "LDA":
    st.subheader("🔍 LDA Konu Modelleme")
    try:
        with st.spinner("LDA modeli kuruluyor..."):
            vectorizer, X, lda_model, lda_output = fit_lda(texts, n_topics, min_df, max_df, lda_max_iter)
    except ValueError as e:
        st.error(f"Vektörleme/model kurulumu başarısız: {e} (min_df / max_df değerlerini gevşetmeyi deneyin)")
        st.stop()

    feature_names = vectorizer.get_feature_names_out()
    st.caption(f"DTM boyutu: {X.shape[0]:,} doküman × {X.shape[1]:,} terim")
    topic_word = lda_model.components_ / lda_model.components_.sum(axis=1, keepdims=True)

    st.subheader("🔠 Konu Anahtar Kelimeleri")
    st.dataframe(topics_table(lda_model.components_, feature_names, labels, 10), width="stretch")

    st.subheader("☁️ Konu Kelime Bulutları")
    fig_wc = plot_topic_wordclouds(lda_model.components_, feature_names, [f"Topic {i + 1}" for i in range(n_topics)])
    st.pyplot(fig_wc)
    plt.close(fig_wc)

    st.subheader("📊 Doküman-Konu Dağılımı")
    topic_df = pd.DataFrame(lda_output, columns=labels)
    st.dataframe(topic_df.head(100).style.background_gradient(cmap="Blues"), width="stretch")
    st.download_button("Konu dağılımını CSV indir", topic_df.to_csv(index=False).encode("utf-8"), "topic_distribution.csv", "text/csv")

    # --- pyLDAvis (sort_topics=False: Topic 1 = components_[0]) ---
    st.subheader("🗟️ pyLDAvis İnteraktif Görselleştirme")
    patch_pyldavis_encoder()
    try:
        with st.spinner("pyLDAvis oluşturuluyor..."):
            vis = pyLDAvis.lda_model.prepare(lda_model, X, vectorizer, mds="mmds", sort_topics=False)
            components.html(pyLDAvis.prepared_data_to_html(vis), width=1300, height=800, scrolling=True)

        coords = vis.topic_coordinates.copy()
        coords["x"] = np.real(coords["x"].astype(complex))
        coords["y"] = np.real(coords["y"].astype(complex))
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(coords["x"], coords["y"], s=coords["Freq"] * 100, c=coords["topics"],
                   cmap="tab20", alpha=0.7, edgecolors="k")
        for _, r in coords.iterrows():
            ax.text(r["x"] + 0.008, r["y"] + 0.008, f"K{int(r['topics'])}", fontsize=9)
        ax.set_title("LDA Konularının PCoA (MDS) Tabanlı Görselleştirmesi")
        ax.set_xlabel("Eksen 1")
        ax.set_ylabel("Eksen 2")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001 - pyLDAvis sürüm uyumsuzlukları çok çeşitli hatalar verebilir
        st.error(f"pyLDAvis oluşturulamadı. Hata: {e}")

    # --- Hiyerarşik kümeleme: KONU-sözcük dağılımları, Jensen-Shannon ---
    st.subheader("🧬 Konu Hiyerarşik Kümelemesi (Jensen-Shannon)")
    linkage_method = st.selectbox("Bağlantı Yöntemi", ["average", "complete", "single"], index=0)
    dist = np.zeros((n_topics, n_topics))
    for i in range(n_topics):
        for j in range(i + 1, n_topics):
            dist[i, j] = dist[j, i] = jensenshannon(topic_word[i], topic_word[j])
    Z = sch.linkage(squareform(dist, checks=False), method=linkage_method)
    fig, ax = plt.subplots(figsize=(12, 6))
    sch.dendrogram(Z, labels=labels, leaf_rotation=0, leaf_font_size=11, ax=ax)
    ax.set_title(f"LDA Konularının Hiyerarşik Kümelenmesi (Jensen-Shannon, {linkage_method} linkage)")
    ax.set_ylabel("Mesafe")
    st.pyplot(fig)
    plt.close(fig)

    # --- Korelasyon: CLR dönüşümü ---
    st.subheader("📈 Konu Korelasyonları (CLR dönüşümlü)")
    st.caption("Doküman-konu oranları bileşimsel (toplam = 1) olduğundan negatif korelasyonlar kısmen yapaydır; bu yüzden log-oran (CLR) dönüşümü uygulanır.")
    theta = lda_output + 1e-6
    clr = np.log(theta) - np.log(theta).mean(axis=1, keepdims=True)
    corr = np.corrcoef(clr.T)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("LDA Konu Korelasyonları (CLR dönüşümlü)")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("🔗 Konu Korelasyon Ağ Grafiği")
    thresh = st.slider("Kenar Korelasyon Eşiği (|r|)", 0.0, 1.0, 0.15, 0.01)
    G = nx.Graph()
    G.add_nodes_from(labels)
    for i in range(n_topics):
        for j in range(i + 1, n_topics):
            if abs(corr[i, j]) > thresh:
                G.add_edge(labels[i], labels[j], weight=round(float(corr[i, j]), 2))
    if G.number_of_edges() == 0:
        st.warning("⚠️ Seçilen eşiğin üzerinde konu korelasyonu yok. Eşiği düşürmeyi deneyin.")
    else:
        fig, ax = plt.subplots(figsize=(9, 9))
        pos = nx.spring_layout(G, seed=SEED, k=0.6)
        colors = ["tab:green" if G[u][v]["weight"] > 0 else "tab:red" for u, v in G.edges]
        nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=1400,
                edge_color=colors, font_weight="bold", ax=ax)
        nx.draw_networkx_edge_labels(G, pos, nx.get_edge_attributes(G, "weight"), ax=ax)
        ax.set_title(f"Konu Korelasyon Ağı (|r| > {thresh}; yeşil: pozitif, kırmızı: negatif)")
        st.pyplot(fig)
        plt.close(fig)

    if date_column:
        show_time_analysis(lda_output, labels, years, min_docs)

# ============================================================================
# NMF
# ============================================================================
elif model_choice == "NMF":
    st.subheader("🔍 NMF (TF-IDF) Konu Modelleme")
    try:
        with st.spinner("NMF modeli kuruluyor..."):
            tfidf_vec, nmf, nmf_output = fit_nmf(texts, n_topics, min_df, max_df)
    except ValueError as e:
        st.error(f"Vektörleme/model kurulumu başarısız: {e}")
        st.stop()

    nmf_features = tfidf_vec.get_feature_names_out()
    st.subheader("🔠 Konu Anahtar Kelimeleri")
    st.dataframe(topics_table(nmf.components_, nmf_features, [f"Konu {i + 1}" for i in range(n_topics)], 10), width="stretch")

    st.subheader("☁️ Konu Kelime Bulutları")
    fig_wc = plot_topic_wordclouds(nmf.components_, nmf_features, [f"Konu {i + 1}" for i in range(n_topics)])
    st.pyplot(fig_wc)
    plt.close(fig_wc)

    st.subheader("📊 Konu Dağılımı (Doküman × Konu)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(nmf_output, cmap="YlGnBu", yticklabels=False, xticklabels=labels, ax=ax)
    ax.set_title("NMF (TF-IDF) – Konu Dağılımı")
    ax.set_xlabel("Konular")
    ax.set_ylabel("Belgeler")
    st.pyplot(fig)
    plt.close(fig)

    topic_df = pd.DataFrame(nmf_output, columns=labels)
    st.download_button("Konu dağılımını CSV indir", topic_df.to_csv(index=False).encode("utf-8"), "nmf_topic_distribution.csv", "text/csv")

    if date_column:
        row_sum = nmf_output.sum(axis=1, keepdims=True)
        normalized = np.divide(nmf_output, row_sum, out=np.zeros_like(nmf_output), where=row_sum > 0)
        show_time_analysis(normalized, labels, years, min_docs)

# ============================================================================
# CTM (tomotopy)
# ============================================================================
elif model_choice == "CTM":
    st.subheader("🔗 Korelatif Konu Modeli (CTM – tomotopy)")
    st.caption("Blei & Lafferty (2007) CTM'si; konu korelasyonları modelin kendi kovaryans matrisinden gelir.")
    n_iter = st.number_input("Eğitim iterasyonu", 100, 5000, 1000, step=100)
    if st.button("CTM'i çalıştır", type="primary"):
        st.session_state.ctm_params = (n_topics, int(min_df), int(n_iter))

    params = st.session_state.get("ctm_params")
    if params:
        try:
            with st.spinner("CTM eğitiliyor (birkaç dakika sürebilir)..."):
                ctm_corr, ctm_topics, ll = fit_ctm(texts, params[0], params[1], 20, params[2])
        except ImportError:
            st.error("Bu analiz için `tomotopy` gerekli: pip install tomotopy")
            st.stop()
        st.metric("CTM log-likelihood / kelime", f"{ll:.4f}")
        ctm_labels = [f"C{i + 1}" for i in range(len(ctm_topics))]
        st.dataframe(pd.DataFrame({"Konu": ctm_labels, "Anahtar kelimeler": [", ".join(w) for w in ctm_topics]}), width="stretch")

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(ctm_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    xticklabels=ctm_labels, yticklabels=ctm_labels, ax=ax)
        ax.set_title("Korelatif Konu Modeli – Konu Korelasyonları")
        st.pyplot(fig)
        plt.close(fig)

# ============================================================================
# BERTopic
# ============================================================================
elif model_choice == "BERTopic":
    st.subheader("🔍 BERTopic")
    st.caption("İlk çalıştırmada gömme (embedding) modeli indirilir; büyük veri setlerinde uzun sürebilir.")
    if st.button("BERTopic'i çalıştır", type="primary"):
        st.session_state.bert_params = n_topics

    if st.session_state.get("bert_params"):
        k = st.session_state.bert_params
        try:
            with st.spinner("BERTopic analizi yapılıyor..."):
                bert, bert_topics = fit_bertopic(texts, k)
        except ImportError as e:
            st.error(f"Gerekli paket eksik: {e}")
            st.stop()
        docs = list(texts)

        info = bert.get_topic_info()
        st.dataframe(info[["Topic", "Count", "Name"]], width="stretch")
        st.metric("Aykırı değer (-1) oranı", f"{np.mean(np.array(bert_topics) == -1):.3f}")

        def safe_plot(title, fn):
            st.subheader(title)
            try:
                st.plotly_chart(fn(), width="stretch")
            except (AttributeError, TypeError, ValueError, IndexError) as exc:
                st.warning(f"⚠️ Oluşturulamadı: {exc}")

        safe_plot("📊 En Önemli Sözcükler", lambda: bert.visualize_barchart(top_n_topics=k, n_words=10))

        if date_column:
            def tot_fig():
                tot = bert.topics_over_time(docs, years.tolist())
                counts = pd.Series(years).value_counts()
                valid = counts[counts >= min_docs].index
                tot = tot[tot["Timestamp"].isin(valid)]
                return bert.visualize_topics_over_time(
                    tot, title=f"BERTopic – Yıllara Göre Konu Dağılımı (yıl başına ≥ {min_docs} doküman)"
                )
            safe_plot("📆 Yıllara Göre Konu Dağılımı", tot_fig)

        safe_plot("📌 Konular Arası Mesafe Haritası", lambda: bert.visualize_topics())
        safe_plot(
            "🌿 Hiyerarşik Konular",
            lambda: bert.visualize_hierarchy(hierarchical_topics=bert.hierarchical_topics(docs)),
        )
        safe_plot("🔥 Konu Benzerlik Isı Haritası", lambda: bert.visualize_heatmap(top_n_topics=k))

st.markdown("---")
st.markdown("© Powered by Streamlit and Scikit-learn")
