import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import pandas as pd
import numpy as np
import torch
import faiss

# Загрузка данных
@st.cache_data
def load_data():
    return pd.read_csv("/Users/marinakochetova/Downloads/cosmo_sum2.csv")

# Загрузка эмбеддингов для текстов статей
@st.cache_data
def load_embeddings_text(embed_text_path):
    embeddings_text = torch.load(embed_text_path)
    embeddings_text_np = embeddings_text.numpy()  # Преобразование в NumPy массив
    faiss.normalize_L2(embeddings_text_np)
    return embeddings_text_np

# Загрузка эмбеддингов для заголовков статей
@st.cache_data
def load_embeddings_title(embed_title_path):
    embeddings_title = torch.load(embed_title_path)
    embeddings_title_np = embeddings_title.numpy()  # Преобразование в NumPy массив
    faiss.normalize_L2(embeddings_title_np)
    return embeddings_title_np

# Инициализация FAISS индекса для текстов статей
@st.cache_data
def initialize_faiss_index_text(embeddings_text):
    dimension_text = embeddings_text.shape[1]
    index_text = faiss.IndexFlatIP(dimension_text)
    index_text.add(embeddings_text)
    return index_text

# Инициализация FAISS индекса для заголовков статей
@st.cache_data
def initialize_faiss_index_title(embeddings_title):
    dimension_title = embeddings_title.shape[1]
    index_title = faiss.IndexFlatIP(dimension_title)
    index_title.add(embeddings_title)
    return index_title

# Поиск похожих статей
def find_similar_articles(index_text, index_title, embeddings_text, embeddings_title, df, art_ind, num_similar,
                          use_text=True, use_title=True):
    query_embedding_text = np.array([embeddings_text[art_ind]], dtype='float32')
    query_embedding_title = np.array([embeddings_title[art_ind]], dtype='float32')
    faiss.normalize_L2(query_embedding_text)
    faiss.normalize_L2(query_embedding_title)

    k = num_similar * 2  # Увеличиваем количество запрашиваемых ближайших соседей

    filtered_indices = []
    filtered_distances = []

    if use_text and use_title:
        D_text, I_text = index_text.search(query_embedding_text, k)
        D_title, I_title = index_title.search(query_embedding_title, k)

        # Взвешивание эмбеддингов
        combined_distances = 0.4 * D_text + 0.6 * D_title
        combined_indices = I_text  # Мы будем использовать индексы текстов, так как они одинаковы для обоих поисков

        query_id = df.loc[art_ind, 'id']

        # Исключение исходной статьи из результатов
        seen_ids = set()

        # Обработка взвешенных результатов
        for idx, dist in zip(combined_indices[0], combined_distances[0]):
            article_id = df.loc[idx, 'id']
            if article_id != query_id and article_id not in seen_ids:
                filtered_indices.append(idx)
                filtered_distances.append(dist)
                seen_ids.add(article_id)

        # Фильтрация результатов до num_similar статей
        filtered_indices = filtered_indices[:num_similar]
        filtered_distances = filtered_distances[:num_similar]

    elif use_text:
        D_text, I_text = index_text.search(query_embedding_text, k)
        query_id = df.loc[art_ind, 'id']

        # Исключение исходной статьи из результатов
        seen_ids = set()

        # Обработка результатов по текстам
        for idx_text, dist_text in zip(I_text[0], D_text[0]):
            article_id_text = df.loc[idx_text, 'id']
            if article_id_text != query_id and article_id_text not in seen_ids:
                filtered_indices.append(idx_text)
                filtered_distances.append(dist_text)
                seen_ids.add(article_id_text)

        # Фильтрация результатов до num_similar статей
        filtered_indices = filtered_indices[:num_similar]
        filtered_distances = filtered_distances[:num_similar]

    elif use_title:
        D_title, I_title = index_title.search(query_embedding_title, k)
        query_id = df.loc[art_ind, 'id']

        # Исключение исходной статьи из результатов
        seen_ids = set()

        # Обработка результатов по заголовкам
        for idx_title, dist_title in zip(I_title[0], D_title[0]):
            article_id_title = df.loc[idx_title, 'id']
            if article_id_title != query_id and article_id_title not in seen_ids:
                filtered_indices.append(idx_title)
                filtered_distances.append(dist_title)
                seen_ids.add(article_id_title)

        # Фильтрация результатов до num_similar статей
        filtered_indices = filtered_indices[:num_similar]
        filtered_distances = filtered_distances[:num_similar]

    return filtered_indices, filtered_distances

# Основное тело приложения
st.markdown(
    '<div style="text-align: center; font-size: 36px;">Рекомендация статей на базе SentenceTransformer</div>',
    unsafe_allow_html=True)
st.markdown('<div style="text-align: center; font-size: 16px;"></div>', unsafe_allow_html=True)

df = load_data()
embed_text_path = "/Users/marinakochetova/Downloads/embeddings_brt_cosmo_text.pth"
embed_title_path = "/Users/marinakochetova/Downloads/embeddings_brt_cosmo_title.pth"
embeddings_text = load_embeddings_text(embed_text_path)
embeddings_title = load_embeddings_title(embed_title_path)
index_text = initialize_faiss_index_text(embeddings_text)
index_title = initialize_faiss_index_title(embeddings_title)

art_ind = st.slider("Выберите статью для поиска", min_value=0, max_value=len(df) - 1, value=0)
num_similar = st.slider("Задайте количество статей для подбора похожих", min_value=5, max_value=10, value=5)
use_text = st.checkbox("Искать по текстам", value=True)
use_title = st.checkbox("Искать по заголовкам", value=True)

# Получение уникальных категорий из датафрейма
categories = df['category'].unique()
selected_categories = st.multiselect("Выберите категории (опционально)", categories, default=[])

generate = st.button("Перегенерировать")

# Инициализация session_state
if 'art_ind' not in st.session_state:
    st.session_state['art_ind'] = art_ind
if 'num_similar' not in st.session_state:
    st.session_state['num_similar'] = num_similar

# Обновление session_state
st.session_state['art_ind'] = art_ind
st.session_state['num_similar'] = num_similar

# Фильтрация датафрейма по выбранным категориям
if selected_categories:
    filtered_df = df[df['category'].isin(selected_categories)].reset_index(drop=True)
else:
    filtered_df = df.reset_index(drop=True)

if generate or st.session_state['art_ind'] != art_ind or st.session_state['num_similar'] != num_similar:
    # Информация об исходной статье
    article_id = df.loc[art_ind, 'id']
    url = f"https://www.thevoicemag.ru/-{article_id}"
    title = df.loc[art_ind, 'title']
    category = df.loc[art_ind, 'category']
    date = df.loc[art_ind, 'date']
    st.write(f"Запрос: [Статья {article_id}](https://www.thevoicemag.ru/-{article_id})")
    st.write(f"Название: {title}")
    st.write(f"Категория: {category}")
    st.write(f"Дата: {date}")

    if filtered_df.empty:
        st.write("Похожие статьи не найдены в выбранных категориях.")
    else:
        # Используем индекс отфильтрованного датафрейма
        filtered_art_ind = filtered_df.index[filtered_df['id'] == df.loc[art_ind, 'id']].tolist()

        if not filtered_art_ind:
            st.write("Похожие статьи не найдены в выбранных категориях.")
        else:
            filtered_art_ind = filtered_art_ind[0]
            embeddings_text_filtered = embeddings_text[filtered_df.index]
            embeddings_title_filtered = embeddings_title[filtered_df.index]
            index_text_filtered = initialize_faiss_index_text(embeddings_text_filtered)
            index_title_filtered = initialize_faiss_index_title(embeddings_title_filtered)

            filtered_indices, filtered_distances = find_similar_articles(index_text_filtered, index_title_filtered,
                                                                         embeddings_text_filtered,
                                                                         embeddings_title_filtered,
                                                                         filtered_df, filtered_art_ind, num_similar,
                                                                         use_text,
                                                                         use_title)

            if len(filtered_indices) == 0:
                st.write("Похожие статьи не найдены в выбранных категориях.")
            else:
                st.write("\nБлижайшие статьи к запросу:")
                for i, (idx, dist) in enumerate(zip(filtered_indices, filtered_distances)):
                    article_id = filtered_df.loc[idx, 'id']
                    url = f"https://www.thevoicemag.ru/-{article_id}"
                    title = filtered_df.loc[idx, 'title']
                    category = filtered_df.loc[idx, 'category']
                    date = filtered_df.loc[idx, 'date']
                    similarity = dist
                    st.write(f"{i + 1}. [Статья {article_id}]({url})")
                    st.write(f"Заголовок: {title}")
                    st.write(f"Категория: {category}")
                    st.write(f"Дата: {date}")
                    st.write(f"Косинусное сходство: {similarity}")

# Вывод таблицы с количеством статей по категориям и сохранение в HTML файл
category_counts = df['category'].value_counts().reset_index()
category_counts.columns = ['Category', 'Count']

st.write("Количство статей по категориям")
st.write(category_counts)

# Сохранение таблицы в HTML файл
#html_table = category_counts.to_html(index=False)
#html_file_path = "/Users/marinakochetova/Downloads/category_counts.html"
#with open(html_file_path, 'w') as f:
#    f.write(html_table)
