import base64
import logging
import re
import unicodedata
import warnings

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
from bertopic import BERTopic
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import pdist
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud

warnings.filterwarnings("ignore", category=FutureWarning, message=".*np.matrix.*")

@st.cache_resource
def download_nltk_data():
    resources = ['punkt', 'punkt_tab', 'stopwords']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except (LookupError, OSError, RuntimeError, ValueError) as e:
            st.error(f"NLTK kaynağı indirilemedi ({resource}): {e}")

# Fonksiyonu çağırın (sadece ilk çalıştırmada çalışır, sonrasında cache'den gelir)
download_nltk_data()

# Function to preprocess text
def preprocess_text(text):
  # Convert to lowercase
  text = text.lower()
  # Remove special characters and digits
  text = re.sub(r'[^a-zA-Z\s]', '', text)
  # Tokenize
  tokens = word_tokenize(text)
  # Remove stopwords
  stop_words = set(stopwords.words('english'))
  tokens = [token for token in tokens if token not in stop_words]
  return ' '.join(tokens)

def normalize_text(text):
    if pd.isna(text):
        return text
    text = str(text)
    # 1. Unicode karakterleri NFKD normuna çevir (özel tırnak ve tireleri ayrıştırır)
    text = unicodedata.normalize('NFKD', text)
    # 2. Özel tire çeşitlerini (En-dash '–', Em-dash '—', vb.) standart '-' ile değiştir
    text = re.sub(r'[\u2010-\u2015\u2212]', '-', text)
    # 3. Akıllı tırnakları (“, ”, ‘, ’) standart tırnak işaretlerine çevir
    text = re.sub(r'[\u2018\u2019]', "'", text)
    text = re.sub(r'[\u201C\u201D]', '"', text)
    # 4. Satır sonları (\n, \r), sekme (\t) ve görünmez boşlukları (\xa0) standart boşluğa çevir
    text = re.sub(r'[\r\n\t\xa0]', ' ', text)
    # 5. Birden fazla boşluğu tek boşluğa indir
    text = re.sub(r'\s+', ' ', text)
    # 6. Baştaki/sondaki boşlukları kaldır ve küçük harfe çevir (harf büyüklüğü farkı kalmasın)
    return text

@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        return None

st.set_page_config(page_title="Konu Modelleme Uygulaması", layout="wide")
st.title("🧠 Kapsamlı Konu Modelleme Uygulaması")
st.sidebar.header("Yükleme ve Ayarlar")

# Session state üzerinde df tanımlı değilse None yapalım
if "df" not in st.session_state:
    st.session_state.df = None

# 1) FILE UPLOAD & ÖN TANIMLI VERİ
# ============================================================================
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

# Veri yüklenmediyse akışı durdur
if st.session_state.df is None:
    st.info("Başlamak için bir dosya yükleyin veya önceden tanımlanmış veri kullanın.")
    st.stop()

# 2) VERİ GÖSTERİMİ
# ============================================================================
df = st.session_state.df

st.sidebar.success(f"{len(df):,} satır, {len(df.columns)} sütun yüklendi.")

text_column = st.sidebar.selectbox(
    "Metin sütununu seçin",
    index=len(df.columns) - 2,
    options=df.columns
)

date_column = st.sidebar.selectbox(
    "Tarih sütununu seçin (isteğe bağlı)",
    index=1,
    options=[None] + list(df.columns)
)

n_topics = st.sidebar.slider("Konu Sayısı", 2, 20, 12)
model_choice = st.sidebar.selectbox("Konu Modelleme Yöntemini Seç", ["EDA", "LDA", "NMF", "BERTopic"])

st.subheader("📄 Veri Önizleme")
st.dataframe(df.head(20), use_container_width=True)

if date_column:
    # Year/date handling:
    # - Numeric 4-digit values such as 1998, 2020, 2024 are treated as YEARS.
    # - Strings such as "2020" are also treated as YEARS.
    # - Full dates such as "2020-05-17" are parsed as dates.
    # - Excel serial dates are handled separately when applicable.
    raw_dates = df[date_column].copy()

    def parse_year_or_date(series):
        numeric = pd.to_numeric(series, errors="coerce")
        # Values in the normal year range are actual years, not timestamps.
        year_mask = numeric.between(1000, 2200)
        result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        if year_mask.any():
            result.loc[year_mask] = pd.to_datetime(
                numeric.loc[year_mask].astype("Int64").astype(str),
                format="%Y",
                errors="coerce"
            )
        # Parse remaining non-year values as normal dates.
        remaining = ~year_mask
        if remaining.any():
            result.loc[remaining] = pd.to_datetime(
                series.loc[remaining],
                errors="coerce"
            )
        return result

    df[date_column] = parse_year_or_date(raw_dates)
    df = df.dropna(subset=[date_column]).copy()
    df["year"] = df[date_column].dt.year.astype(int)
    st.subheader("📅 Yıl Dağılımı")
    st.bar_chart(df["year"].value_counts().sort_index())

if text_column:
    df[text_column] = df[text_column].replace("[NO ABSTRACT AVAILABLE]", pd.NA)
    df.dropna(subset=[text_column], inplace=True)  # Keep only entries with abstracts
    df[text_column] = df[text_column].apply(normalize_text)
    df["processed_texts"] = df[text_column].apply(preprocess_text)  # Assuming you have a preprocess_text function defined

texts = df["processed_texts"].fillna("").astype(str).tolist()
vectorizer = CountVectorizer(stop_words="english")
try:
    X = vectorizer.fit_transform(texts)
except ValueError as e:
    st.error(f"Metin vektörleme başarısız oldu: {e}")
    st.stop()
X = vectorizer.fit_transform(texts)

# --- Export Button ---
def generate_download_link(df, filename="topic_distribution.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📅 Konu Dağılımı CSV Olarak İndir</a>'
    return href

# --- Visualization Style Enhancements ---
def styled_line_chart(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette("husl", n_colors=df.shape[1])
    df.plot(ax=ax, linewidth=2, color=palette)
    plt.title("Ortalama Konu Dağılımı Zaman İçinde", fontsize=16)
    plt.xlabel("Yıl")
    plt.ylabel("Ortalama Konu Ağırlığı")
    plt.grid(True)
    st.pyplot(fig)
    
if model_choice == "EDA":
    with st.spinner("Keşifci Veri Analizi Yapılıyor..."):
        st.subheader("📊 Keşifçi Veri Analizi")
        # Basic info
        st.markdown("**Veri Kümesi Boyutları:**")
        st.write(f"🗋 Satırlar: {df.shape[0]}, 📚 Sütunlar: {df.shape[1]}")
        # Text Lengths
        df["text_length"] = df[text_column].apply(lambda x: len(str(x).split()))
        st.markdown("**Metin Uzunluğu İstatistikleri:**")
        st.write(df["text_length"].describe())
        # Histogram of text lengths
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df["text_length"], bins=30, kde=True, color="teal") # type: ignore
        ax.set_title("Belgelerin Uzunluk Dağılımı")
        ax.set_xlabel("Kelime Sayısı")
        ax.set_ylabel("Frekans")
        st.pyplot(fig)
        # Most common words
        from collections import Counter
        all_words = [word for text in texts for word in text.split()]
        word_freq = Counter(all_words)
        common_words = word_freq.most_common(20)
        st.markdown("**İlk 20 En Sık Karakterler:**")
        common_df = pd.DataFrame(common_words, columns=["Word", "Frequency"])
        st.dataframe(common_df)
        # Word cloud
        st.markdown("**Sık Kullanılan Kelimelerin Kelime Bulutu:**")
        wordcloud = WordCloud(width=1000, height=400, background_color='white').generate_from_frequencies(word_freq)
        fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)
        # If date column exists, show number of documents over time
        if date_column:
            st.markdown("**🗓️ Yıllara Göre Belgeler:**")
            year_counts = df['year'].value_counts().sort_index()
            st.bar_chart(year_counts)
        st.markdown("---")
        st.markdown("✅ Veriyi inceledikten sonra, yan panelden bir konu modelleme algoritması seçebilir ve çalıştırabilirsiniz.")

elif model_choice == "LDA":
    with st.spinner("LDA Analizi Yapılıyor..."):
        st.subheader("🔍 LDA Konu Modelleme")
        doc_term_matrix = X
        lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda_output = lda_model.fit_transform(doc_term_matrix)
        topic_df = pd.DataFrame(lda_output, columns=[f"T{i+1}" for i in range(n_topics)])
        st.dataframe(topic_df.style.background_gradient(cmap="Blues"))
        st.markdown(generate_download_link(topic_df), unsafe_allow_html=True)

        # lda_model.fit(doc_term_matrix)
        vocab = vectorizer.get_feature_names_out()
        topic_distributions = lda_model.transform(doc_term_matrix)

        st.subheader("🔠 LDA Kelime Bulutları ve Anahtar Kelimeler")
        for idx, topic in enumerate(lda_model.components_):
            with st.expander(f"Konu {idx}", expanded=False):
                words = [vocab[i] for i in topic.argsort()[:-11:-1]]
                st.write(", ".join(words)) # type: ignore
                word_freq = {vocab[i]: topic[i] for i in topic.argsort()[:-11:-1]}
                wc = WordCloud(width=600, height=300, background_color='white').generate_from_frequencies(word_freq)
                st.image(wc.to_array())

        st.subheader("📊 Dökümanlara Göre Konu Dağılımı")
        st.write(pd.DataFrame(topic_distributions, columns=[f"T{i+1}" for i in range(n_topics)]))

        if date_column:
            if lda_output.shape[0] == len(df):
                df['topic_distr'] = lda_output.tolist()
                yearly_topics = df.groupby('year')['topic_distr'].apply(lambda x: np.mean(x.tolist(), axis=0))
                yearly_df = pd.DataFrame(yearly_topics.tolist(), index=yearly_topics.index, columns=[f"T{i+1}" for i in range(n_topics)])
                st.subheader("📆 Yıllara Göre Ortalama Konu Dağılımı")
                styled_line_chart(yearly_df)
            else:
                st.warning("⚠️ Konu dağılımı satırları ve belge sayısı arasında uyuşmazlık var. Yıllık toplama atlanıyor.")

            if 'yearly_df' in locals():
                st.subheader("📊 Zamana Göre Konu Dağılımı - Yığılmış Çubuk Grafik")
                fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
                yearly_df.plot(kind='bar', stacked=True, ax=ax_bar, colormap='tab20') # type: ignore
                ax_bar.set_ylabel("Ortalama Konu Oranı")
                ax_bar.set_xlabel("Yıl")
                ax_bar.set_title("Yıllara Göre Yığılmış Ortalama Konu Dağılımı")
                st.pyplot(fig_bar)

        st.subheader("🗟️ pyLDAvis İnteraktif Görselleştirme")
        # Store original default encoder
        _original_default = pyLDAvis.utils.NumPyEncoder.default

        # Patch NumPyEncoder to strip imaginary parts
        def custom_default(self, obj):
            if isinstance(obj, (complex, np.complex128, np.complex64)): # type: ignore
                return float(obj.real)
            return _original_default(self, obj)

        pyLDAvis.utils.NumPyEncoder.default = custom_default 

        with st.spinner("pyLDAvis oluşturuluyor..."):
            try:
                vis_data = pyLDAvis.lda_model.prepare(lda_model, doc_term_matrix, vectorizer, mds='pcoa')
                html_string = pyLDAvis.prepared_data_to_html(vis_data)
                components.html(html_string, width=1300, height=800, scrolling=True)
            except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                st.error(f"pyLDAvis oluşturulamadı. Hata: {e}")

        st.subheader("📈 Konu Korelasyon Isı Haritası")
        corr_matrix = np.corrcoef(topic_distributions.T)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, fmt=".2f", 
                    xticklabels=[f"T{i+1}" for i in range(n_topics)], 
                    yticklabels=[f"T{i+1}" for i in range(n_topics)], 
                    ax=ax)
        st.pyplot(fig)

        st.subheader("🔗 Konu Korelasyon Ağ Grafiği")
        threshold = st.slider("Kenar Korelasyon Eşiği", 0.0, 1.0, 0.1, 0.01)
        G = nx.Graph()
        for i in range(n_topics):
            G.add_node(f"T{i+1}")
            for j in range(i + 1, n_topics):
                if abs(corr_matrix[i, j]) >= threshold:
                    G.add_edge(f"T{i+1}", f"T{j+1}", weight=round(corr_matrix[i, j], 3))
        fig_net, ax_net = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        weights = [G[u][v]['weight'] for u, v in G.edges]

        if weights:
            nx.draw(G, pos, with_labels=True, node_size=1000, 
                    node_color="skyblue", edge_color=weights, 
                    edge_cmap=plt.cm.Blues, width=2, ax=ax_net)
            # Draw edge weights
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(weights), vmax=max(weights))) # type: ignore
            sm.set_array([])
            fig_net.colorbar(sm, ax=ax_net, label='Korelasyon Ağırlığı')
            st.pyplot(fig_net)
        else:
            st.warning("⚠️ Seçilen eşiğin üzerinde konu korelasyonu yok. Eşiği düşürmeyi deneyin.")

        st.subheader("🧬 Konu Hiyerarşik Kümelemesi")
        distance_metric = st.selectbox("Mesafe Metriği", ["euclidean", "cosine"], index=1)
        linkage_method = st.selectbox("Bağlantı Yöntemi", ["ward", "complete", "average", "single"], index=2)
        fig_dendro, ax_dendro = plt.subplots(figsize=(10, 5))
        dist_matrix = pdist(topic_distributions.T, metric=distance_metric) # type: ignore
        linkage_matrix = sch.linkage(dist_matrix, method=linkage_method)
        sch.dendrogram(linkage_matrix, labels=[f"T{i+1}" for i in range(n_topics)], ax=ax_dendro)
        st.pyplot(fig_dendro)

elif model_choice == "NMF":
    with st.spinner("NMF Konu Modelleme Yapılıyor..."):
        st.subheader("🔍 NMF Topic Modeling")
        nmf_model = NMF(n_components=n_topics, random_state=42)
        nmf_output = nmf_model.fit_transform(X)
        topic_df = pd.DataFrame(nmf_output, columns=[f"T{i+1}" for i in range(n_topics)])
        st.dataframe(topic_df.style.background_gradient(cmap="Purples"))
        st.markdown(generate_download_link(topic_df), unsafe_allow_html=True)

        tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(texts)
        nmf = NMF(n_components=n_topics, random_state=42)
        nmf.fit(tfidf_matrix)
        vocab = tfidf.get_feature_names_out()
        topic_distributions = nmf.transform(tfidf_matrix)

        for idx, topic in enumerate(nmf.components_):
            with st.expander(f"Konu {idx}", expanded=False):
                words = [vocab[i] for i in topic.argsort()[:-11:-1]]
                st.write(", ".join(words)) # type: ignore
                word_freq = {vocab[i]: topic[i] for i in topic.argsort()[:-11:-1]}
                wc = WordCloud(width=600, height=300, background_color='white').generate_from_frequencies(word_freq)
                st.image(wc.to_array())

        st.subheader("📊 Konu Dağılımı - Belge Bazlı")
        st.write(pd.DataFrame(topic_distributions, columns=[f"T{i+1}" for i in range(n_topics)]))

        if date_column:
            if nmf_output.shape[0] == len(df):
                df['topic_distr'] = nmf_output.tolist()
                yearly_topics = df.groupby('year')['topic_distr'].apply(lambda x: np.mean(x.tolist(), axis=0))
                yearly_df = pd.DataFrame(yearly_topics.tolist(), index=yearly_topics.index, columns=[f"T{i+1}" for i in range(n_topics)])
                st.subheader("📆 Zaman İçinde Ortalama Konu Dağılımı")
                styled_line_chart(yearly_df)
            else:
                st.warning("⚠️ Döküm Uyumsuzluğu: Konu dağılımı satırları ile belge sayısı uyuşmuyor. Yıl bazlı agregasyon atlanıyor.")

elif model_choice == "BERTopic":
    with st.spinner("BERTopic Analizi Yapılıyor..."):
        st.subheader("🔍 BERTopic")
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(texts)
        try:
            topic_model = topic_model.reduce_topics(texts, nr_topics=n_topics)
        except (TypeError, ValueError) as exc:
            logging.exception("BERTopic konu azaltma işlemi başarısız oldu; azaltma yapılmadan devam ediliyor.")  # noqa: LOG015
            st.warning(f"⚠️ BERTopic konu azaltma işlemi uygulanamadı: {exc}")

        assert topic_model is not None
        st.write(topic_model.get_topic_info().head(n_topics))
        fig = topic_model.visualize_barchart(top_n_topics=n_topics)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔠 BERTopic Kelime Bulutları ve Anahtar Kelimeler")
        for topic_id in topic_model.get_topic_freq().head(n_topics)['Topic']: # type: ignore
            if topic_id == -1:
                continue
            with st.expander(f"Konu {topic_id}", expanded=False):
                words_scores = topic_model.get_topic(topic_id)
                keywords = [word for word, _ in words_scores[:10]] # type: ignore
                st.markdown(f"**Zirvedeki Anahtar Kelimeler:** {', '.join(keywords)}")
                word_freq = {word: score for word, score in words_scores[:30]} # type: ignore
                wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
                st.image(wc.to_array(), use_container_width=True)

        # st.subheader("📊 Topic Distribution by Document")
        if date_column:
            if len(texts) == len(df):
                try:
                    topics_over_time = topic_model.topics_over_time(texts, df[date_column]) # type: ignore
                    fig = topic_model.visualize_topics_over_time(topics_over_time)
                    st.plotly_chart(fig, use_container_width=True)
                except (AttributeError, TypeError, ValueError) as exc:
                    logging.exception("BERTopic zaman analizi başarısız oldu.")  # noqa: LOG015
                    st.warning(f"⚠️ BERTopic zaman analizi oluşturulamadı: {exc}")
            else:
                st.warning("⚠️ Döküm Uyumsuzluğu: Konu dağılımı satırları ile belge sayısı uyuşmuyor. Yıl bazlı agregasyon atlanıyor.")

        st.subheader("📌 BERTopic Konular Arası Mesafe Haritası")
        st.plotly_chart(topic_model.visualize_topics(), use_container_width=True)

        st.subheader("🌿 BERTopic Hiyerarşi Görselleştirme")
        hierarchical_topics = topic_model.hierarchical_topics(texts)
        st.plotly_chart(topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics), use_container_width=True)

        st.subheader("🌿 BERTopic Isı Haritası Görseleştirme")
        fig = topic_model.visualize_heatmap()
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("© Powered by Streamlit and Scikit-learn")
