from ArticlesProject.lib.service import do_filter, create_graph, save_graph


# función para filtrar los artículos según los keywords
def filter_articles(keywords: list):
    filtered_articles = do_filter(keywords)
    graph = create_graph(filtered_articles, keywords)
    save_graph(graph)
    return filtered_articles
