# from Grafo_Hospitales import create_grafo_general
# from Grafo_Hospitales import all_hospitales
# from main_pacientes import filtrad_sintomas
# from operator import itemgetter
# import networkx as nx
# import matplotlib.pyplot as plt
# import math
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#
# hospitales = all_hospitales()
# global_list = []
#
#
# class Arista:
#     def __init__(self, origen, destino, peso):
#         self.origen = origen
#         self.destino = destino
#         self.peso = peso
#
#
# def kruskal(grafo):
#     # Crear un bosque de conjuntos disjuntos
#     conjuntos_disjuntos = {nodo: {nodo} for nodo in grafo.nodes}
#
#     # Ordenar las aristas del grafo por peso de forma ascendente
#     aristas_ordenadas = sorted(grafo.edges.data('weight'), key=itemgetter(2))
#
#     # Crear un grafo vacío para almacenar el árbol de expansión mínima
#     arbol_expansion = nx.Graph()
#
#     for u, v, peso in aristas_ordenadas:
#         # Verificar si los nodos u y v pertenecen a conjuntos diferentes
#         if conjuntos_disjuntos[u] != conjuntos_disjuntos[v]:
#             # Agregar la arista al árbol de expansión mínima
#             arbol_expansion.add_edge(u, v, weight=peso)
#
#             # Unir los conjuntos de u y v en un solo conjunto
#             conjuntos_disjuntos[u] = conjuntos_disjuntos[u].union(conjuntos_disjuntos[v])
#
#             # Actualizar los conjuntos de los nodos vecinos
#             for nodo in conjuntos_disjuntos[v]:
#                 conjuntos_disjuntos[nodo] = conjuntos_disjuntos[u]
#
#     return arbol_expansion
#
#
# def haversine(lat1, lon1, lat2, lon2):
#     """
#     Calcula la distancia haversine entre dos puntos geográficos en kilómetros.
#     """
#     # Convertir de grados a radianes
#     lat1 = math.radians(lat1)
#     lon1 = math.radians(lon1)
#     lat2 = math.radians(lat2)
#     lon2 = math.radians(lon2)
#
#     # Diferencia de latitud y longitud
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
#
#     # Fórmula haversine
#     a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#     distancia = 6371 * c  # Radio de la Tierra en kilómetros
#
#     return distancia
#
#
# def main_aux(sintoma1, sintoma2):
#     global global_list
#     main_hospitales = list(filtrad_sintomas(sintoma1, sintoma2))
#     lista_hospitales = set()
#
#     lista_pacientes_form = []
#     for i in main_hospitales:
#         lista_hospitales.add(i.id_eess)
#         lista_pacientes_form.append(
#             'El paciente' + str(i.id) + ' se atendio en el hospital ' + str(hospitales[i.id_eess - 1].nombre))
#     global_list = list(lista_hospitales)
#
#     return lista_pacientes_form
#
#
# def BFS(graph, s, t, parent):
#     visited = [False] * len(graph)
#     queue = []
#
#     queue.append(s)
#     visited[s] = True
#
#     while queue:
#         u = queue.pop(0)
#         for ind in range(len(graph[u])):
#             if visited[ind] is False and graph[u][ind] > 0:
#                 queue.append(ind)
#                 visited[ind] = True
#                 parent[ind] = u
#
#     return True if visited[t] else False
#
#
# def FordFulkerson(graph, source, sink):
#     parent = [-1] * (len(graph))
#     max_flow = 0
#
#     while BFS(graph, source, sink, parent):
#
#         path_flow = float("Inf")
#         s = sink
#
#         while s != source:
#             path_flow = min(path_flow, graph[parent[s]][s])
#             s = parent[s]
#
#         max_flow += path_flow
#
#         v = sink
#         while v != source:
#             u = parent[v]
#             graph[u][v] -= path_flow
#             graph[v][u] += path_flow
#             v = parent[v]
#
#     return max_flow
#
#
# def mostrar_grafico1():
#     global global_list
#     grafo_general_hospitales = create_grafo_general()
#     subgraph1 = nx.Graph()
#     for i in range(len(global_list) - 1):
#         nodo1 = global_list[i]
#         nodo2 = global_list[i + 1]
#         distancia = haversine(hospitales[nodo1].latitud, hospitales[nodo1].longitud, hospitales[nodo2].latitud,
#                               hospitales[nodo2].longitud)
#         subgraph1.add_edge(nodo1, nodo2, weight=str(round(distancia, 2)) + "Km")
#     subgraph1.add_node(global_list[-1])
#     ##Buscamos el MTS de los hospitales filtrados en base a los pacientes.
#     MTS = kruskal(subgraph1)
#
#     fig = plt.figure(figsize=(7, 9))
#     # Dibujar grafo General
#     pos = nx.get_node_attributes(grafo_general_hospitales, "pos")
#     nx.draw_networkx_nodes(grafo_general_hospitales, pos=pos, node_color="green", node_size=30)
#
#     # Dibujar subgrafo
#     nx.draw_networkx_nodes(MTS, pos=pos, node_color="red", node_size=30)
#     # # Dibujar las aristas del subgrafo completo
#     nx.draw_networkx_edges(MTS, pos, alpha=0.9, edge_color="black")
#     # #Agregar etiquetas de peso a las aristas
#     edge_labels = nx.get_edge_attributes(MTS, "weight")
#     # #para que se muestren los pesos
#     nx.draw_networkx_edge_labels(MTS, pos, edge_labels=edge_labels, font_size=6)
#
#     plt.title("Grafico de hospitales del Peru")
#     plt.xlabel("Longitud")
#     plt.ylabel("Latitud")
#
#     return fig
#
#
# def mostrar_flujo_max():
#     global global_list
#     subgraph2 = nx.Graph()
#     for i in range(len(global_list) - 1):
#         nodo1 = global_list[i]
#         nodo2 = global_list[i + 1]
#         capacidad = int(hospitales[nodo1].capacidad / 1851)
#         subgraph2.add_node(nodo1)
#         subgraph2.add_node(nodo2)
#
#         # Agregar la capacidad como atributo de la arista entre los nodos
#         subgraph2[nodo1][nodo2]['capacidad'] = capacidad
#
#     FlujoMaximo = FordFulkerson(subgraph2, 1, 1851)
#
#     return print("El Maximo Flujo Posible es: ", FlujoMaximo)
