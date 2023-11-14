from ArticlesProject.lib.service import do_filter, create_graph, save_graph, filter_categories, find_most_similar_article, similar_articles_bfs


# función para filtrar los artículos según los keywords
def filter_articles(keywords: list):
    filtered_articles = do_filter(keywords)[:50]
    graph = create_graph(filtered_articles, keywords)
    save_graph(graph)
    most_similar_article = find_most_similar_article(keywords, filtered_articles)
    id_to_article_map = {article.id: article for article in filtered_articles}
    similar_articles = similar_articles_bfs(graph, most_similar_article, id_to_article_map)
    return similar_articles


def categories(similar_articles: list):
    filtered_categories = filter_categories(similar_articles)
    return filtered_categories
