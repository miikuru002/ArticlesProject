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
    # se itera sobre cada artículo en 'articles_data' que son todos los artículos del csv usando comprensión de lista
    filtered_articles = [
        article for article in articles_data if
        # verifica si algun kw se encuentra en el titulo, subtitulo o el nombre de la publicacion del articulo
        any(keyword.lower() in article.title.lower() or
            keyword.lower() in article.subtitle.lower() or
            keyword.lower() in article.publication.lower()
            for keyword in keywords)
    ]

    return filtered_articles


# Función para combinar el título y el subtítulo
def combine_title_subtitle(article):
    return f"{article.title} {article.subtitle}"


# Función para calcular el promedio del mínimo y máximo de "claps" en una lista de artículos
def calculate_average_claps(filtered_articles: list):
    min_claps = min(article.claps for article in filtered_articles)
    max_claps = max(article.claps for article in filtered_articles)
    average_claps = (min_claps + max_claps) / 2
    return average_claps


# Función para calcular la similitud coseno entre los artículos y un conjunto de palabras clave
def calculate_cosine_similarity(filtered_articles):
    # obtiene los títulos de los artículos
    titles = [article.title for article in filtered_articles]

    # stop_words -> quita las palabras en inglés como "the", "a", "an", etc. que no son útiles para la similitud
    # TidfVectorizer = TF-IDF (term frequency - inverse document frequency)
    # TF -> mide la frecuencia en la que aparece una palabra en un string (dividiendo por el total de palabras)
    # IDF -> mide la importancia de una palabra en un string
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # tfidf_matrix -> matriz de similitud  coseno entre los artículos (cada fila es un titulo y cada columna un termino)
    tfidf_matrix = tfidf_vectorizer.fit_transform(titles)

    # linear_kernel -> calcula el producto punto entre la matriz
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # retornar la matriz de similitud coseno donde [i][j] es la similitud entre el artículo I y el artículo J
    return cosine_sim


# Función para ajustar la similitud coseno basada en la coincidencia de la publicación
def adjust_similarity(cosine_similarities, filtered_articles, keywords):
    # calcula el promedio de "claps" (likes) en los artículos filtrados
    average_claps = calculate_average_claps(filtered_articles)

    # crea una copia de la matriz de similitud coseno
    adjusted_similarities = cosine_similarities.copy()

    # se itera sobre los artículos filtrados (similar al for each)
    for i, article in enumerate(filtered_articles):
        # verifica si el texto de la publicación del artículo está en los keywords
        if article.publication.lower() in keywords:
            # aumenta la similitud
            adjusted_similarities[i] += 0.1

        # si el número de "claps" (likes) es mayor que el promedio
        if article.claps > average_claps:
            # aumenta la similitud
            adjusted_similarities[i] += 0.1

    # se limita los valores de la matriz entre 0 y 1 (si es que < 0 se pone 0 y si es que > 1 se pone 1)
    adjusted_similarities = np.clip(adjusted_similarities, 0, 1)

    return adjusted_similarities


# Función para construir el grafo con la similitud coseno como peso de los bordes
def create_graph(filtered_articles: list, keywords: list):
    # se calcula la similitud coseno entre los artículos filtrados en base a los títulos
    cosine_similarities = calculate_cosine_similarity(filtered_articles)
    # se ajusta la similitud coseno
    adjusted_similarities = adjust_similarity(cosine_similarities, filtered_articles,
                                              [keyword.lower() for keyword in keywords])

    # se crea un nuevo grafo
    graph = nx.Graph()

    # se añaden nodos para cada artículo en el grafo
    for article in filtered_articles:
        graph.add_node(article.id, title=article.title)

    # se añaden los bordes con pesos ajustados
    for i, article1 in enumerate(filtered_articles):
        for j, article2 in enumerate(filtered_articles):
            if i < j:  # Evita conectarse a sí mismo y duplicar bordes
                # Uso el promedio de las similitudes ajustadas como peso
                weight = (adjusted_similarities[i][j] + adjusted_similarities[j][i]) / 2

                # agrega el borde si el peso es mayor que 0
                if weight > 0:
                    graph.add_edge(article1.id, article2.id, weight=round(weight, 3))  # Redondeo a 3 decimales

    return graph


def save_graph(graph):
    # se dibuja el grafo si no está vacío
    if len(graph) > 1:  # Más de un nodo en el grafo
        pos = nx.spring_layout(graph)  # para la disposición de los nodos

        # Convertir los pesos a porcentajes y crear etiquetas de borde
        edge_weights = nx.get_edge_attributes(graph, 'weight')
        edge_labels = {k: f"{v * 100:.2f}%" for k, v in edge_weights.items()}

        nx.draw(graph, pos, with_labels=True, node_size=700, node_color='lightblue')
        nx.draw_networkx_edge_labels(graph, pos, font_size=8, edge_labels=edge_labels)
        # plt.show()

        # se guarda el grafo en una imagen
        graphs_path = os.path.join(settings.BASE_DIR, 'graphs')
        if not os.path.exists(graphs_path):
            os.makedirs(graphs_path)

        now = datetime.now()
        fecha_hora_str = now.strftime("%Y-%m-%d-%H-%M-%S")

        # guardar la figura
        graph_path = os.path.join(graphs_path, f'graph_{fecha_hora_str}.png')
        plt.savefig(graph_path)
        # plt.close()
    else:
        print("El grafo no tiene nodos conectados para dibujar.")
