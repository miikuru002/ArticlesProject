import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ArticlesProject.database.articles import get_all_articles
from ArticlesProject import settings
from datetime import datetime
from queue import Queue

articles_data = get_all_articles()


# función para filtrar los artículos según los keywords
def do_filter(keywords: list):
    filtered_articles = [
        article for article in articles_data if
        any(keyword.lower() in article.title.lower() or
            keyword.lower() in article.subtitle.lower()
            for keyword in keywords)
    ]

    return filtered_articles


# Función para obtener las categorías de los artículos filtrados
def filter_categories(filtered_articles):
    filtered_categories = []
    for article in filtered_articles:
        if article.category not in filtered_categories:
            filtered_categories.append(article.category)
    return filtered_categories


# Función para combinar el título y el subtítulo
def combine_title_subtitle(article):
    return f"{article.title} {article.subtitle}"


# Función para calcular el promedio del mínimo y máximo de "claps" en una lista de artículos
def calculate_average_claps(filtered_articles: list):
    min_claps = min(article.claps for article in filtered_articles)
    max_claps = max(article.claps for article in filtered_articles)
    average_claps = (min_claps + max_claps)/2
    return average_claps


# Función para calcular la similitud coseno entre los artículos y un conjunto de palabras clave
def calculate_cosine_similarity_with_keywords(filtered_articles):
    titles = [article.title for article in filtered_articles]

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(titles)

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim


# Función para ajustar la similitud coseno basada en la coincidencia de la publicación

def adjust_similarity(cosine_similarities, filtered_articles, keywords):
    average_claps = calculate_average_claps(filtered_articles)

    # Ajuste basado en la coincidencia de la publicación
    adjusted_similarities = cosine_similarities.copy()
    for i, article in enumerate(filtered_articles):
        if article.publication.lower() in keywords:
            adjusted_similarities[i] += 0.1  # Ajuste para la coincidencia de la publicación

        # Ajuste basado en el número de "claps" (likes)
        if article.claps > average_claps:
            # Ajusta la similitud en función del número de "claps" (likes)
            adjusted_similarities[i] += 0.1

    # Asegurarse de que la similitud sigue estando en el rango [0, 1]
    adjusted_similarities = np.clip(adjusted_similarities, 0, 1)

    return adjusted_similarities


# Función para construir el grafo con la similitud coseno como peso de los bordes
def create_graph(filtered_articles: list, keywords: list):
    # Calcular la similitud coseno entre los artículos
    cosine_similarities = calculate_cosine_similarity_with_keywords(filtered_articles)
    # Ajustar la similitud coseno basada en criterios específicos
    adjusted_similarities = adjust_similarity(cosine_similarities, filtered_articles, [keyword.lower() for keyword in keywords])

    # Crear un nuevo grafo
    graph = nx.Graph()

    # Añadir nodos para cada artículo en el grafo
    for article in filtered_articles:
        graph.add_node(article.id, title=article.title)

    # Añadir bordes con pesos ajustados
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



def find_most_similar_article(keywords: list, filtered_articles):
    # Combinar título y subtítulo (si es necesario) y extraerlos
    combined_texts = [combine_title_subtitle(article) for article in filtered_articles]

    # Vectorizar los textos de los artículos
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_texts)

    # Convertir la lista de palabras clave en una cadena y vectorizar
    keywords_text = ' '.join(keywords)
    keywords_vector = tfidf_vectorizer.transform([keywords_text])

    # Calcular la similitud coseno usando linear_kernel
    cosine_similarities = linear_kernel(keywords_vector, tfidf_matrix).flatten()

    # Encontrar el índice del artículo más similar
    most_similar_article_idx = np.argmax(cosine_similarities)

    # Devolver el id del artículo más similar
    return filtered_articles[most_similar_article_idx].id

def similar_articles_bfs(graph, start_id, id_to_article_map):
    visited = set()  # Conjunto para almacenar los nodos visitados
    queue = Queue()  # Cola para BFS

    # Iniciar BFS
    queue.put(start_id)
    visited.add(start_id)

    similar_articles = []  # Lista para almacenar los artículos similares

    while not queue.empty():
        current_id = queue.get()
        # Añadir el objeto de artículo correspondiente al ID actual
        similar_articles.append(id_to_article_map[current_id])

        # Obtener todos los vecinos del nodo actual
        for neighbor_id in graph.neighbors(current_id):
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                queue.put(neighbor_id)

    return similar_articles