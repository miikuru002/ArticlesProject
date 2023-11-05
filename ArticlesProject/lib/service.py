import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ArticlesProject.database.articles import get_all_articles
from ArticlesProject import settings
from datetime import datetime

articles_data = get_all_articles()


# función para filtrar los artículos según los keywords
def do_filter(keywords: list):
    filtered_articles = [
        article for article in articles_data if
        any(keyword.lower() in article.title.lower() or
            keyword.lower() in article.subtitle.lower() or
            keyword.lower() in article.publication.lower()
            for keyword in keywords)
    ]

    return filtered_articles


# Función para combinar el título y el subtítulo
def combine_title_subtitle(article):
    return f"{article.title} {article.subtitle}"


# Función para calcular la similitud coseno entre los artículos y un conjunto de palabras clave
def calculate_cosine_similarity_with_keywords(filtered_articles, keywords):
    combined_texts = [combine_title_subtitle(article) for article in filtered_articles]
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_vectorizer.fit(combined_texts)

    # Crear la representación TF-IDF de los artículos
    article_tfidf_matrix = tfidf_vectorizer.transform(
        [combine_title_subtitle(article) for article in filtered_articles])

    # Transformar las palabras clave en una representación TF-IDF
    keywords_string = ' '.join(keywords).lower()
    keywords_tfidf = tfidf_vectorizer.transform([keywords_string])

    # Calcular la similitud coseno entre las palabras clave y los artículos
    cosine_similarities = linear_kernel(keywords_tfidf, article_tfidf_matrix).flatten()

    return cosine_similarities


# Función para ajustar la similitud coseno basada en la coincidencia de la publicación
def adjust_similarity_for_publication(cosine_similarities, filtered_articles, keywords):
    # Ajuste basado en la coincidencia de la publicación
    adjusted_similarities = cosine_similarities.copy()
    for i, article in enumerate(filtered_articles):
        if article.publication.lower() in keywords:
            adjusted_similarities[i] += 0.1  # Ajuste para la coincidencia de la publicación

    # Asegurarse de que la similitud sigue estando en el rango [0, 1]
    adjusted_similarities = np.clip(adjusted_similarities, 0, 1)

    return adjusted_similarities


# Función para construir el grafo con la similitud coseno como peso de los bordes
def create_graph(filtered_articles: list, keywords: list):
    # Calcula la similitud coseno entre los artículos y las palabras clave
    cosine_similarities = calculate_cosine_similarity_with_keywords(filtered_articles, keywords)
    # Ajusta la similitud basada en la coincidencia de la publicación
    adjusted_similarities = adjust_similarity_for_publication(cosine_similarities, filtered_articles, keywords)

    # Ordena los artículos por similitud ajustada y selecciona los 100 principales
    top_indices = np.argsort(-adjusted_similarities)[:50]
    top_articles = [filtered_articles[i] for i in top_indices]
    top_similarities = adjusted_similarities[top_indices]

    main_graph = nx.Graph()

    # se agregan los nodos
    for article in top_articles:
        main_graph.add_node(article.id, title=article.title)

    # Encuentra el artículo con la mayor similitud (posible nodo de partida)
    starting_article_index = top_similarities.argmax()
    starting_article_id = top_articles[starting_article_index].id

    # Añadir bordes con pesos basados en la similitud coseno ajustada
    for i, article in enumerate(top_articles):
        if i != starting_article_index:  # No conectamos el artículo con sí mismo
            weight = round(top_similarities[i], 3)
            if weight > 0:  # Agrega un borde si la similitud es mayor que cero
                main_graph.add_edge(starting_article_id, article.id, weight=weight)

    # # Añadir bordes con pesos basados en la similitud coseno ajustada
    # for i, article1 in enumerate(top_articles):
    #     for j, article2 in enumerate(top_articles):
    #         if i < j:  # Para evitar duplicados y no comparar un nodo consigo mismo
    #             weight = round(min(adjusted_similarities[i], adjusted_similarities[j]), 3)
    #             if weight > 0:  # Agrega un borde si la similitud es mayor que cero
    #                 main_graph.add_edge(article1.id, article2.id, weight=weight)

    # se dibuja el grafo si no está vacío
    if len(main_graph) > 1:  # Más de un nodo en el grafo
        pos = nx.spring_layout(main_graph)  # para la disposición de los nodos
        labels = nx.get_edge_attributes(main_graph, 'weight')
        nx.draw(main_graph, pos, with_labels=True)
        nx.draw_networkx_edge_labels(main_graph, pos, edge_labels=labels)
        plt.title("Grafico de hospitales del Peru")
        plt.show()

        # se guarda el grafo en una imagen
        graphs_path = os.path.join(settings.BASE_DIR, 'graphs')
        if not os.path.exists(graphs_path):
            os.makedirs(graphs_path)

        now = datetime.now()
        fecha_hora_str = now.strftime("%Y-%m-%d-%H-%M-%S")

        # guardar la figura
        graph_path = os.path.join(graphs_path, f'graph_{fecha_hora_str}.png')
        plt.savefig(graph_path)
        plt.close()
    else:
        print("El grafo no tiene nodos conectados para dibujar.")

    return main_graph


