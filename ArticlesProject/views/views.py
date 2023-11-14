from django.shortcuts import *
from ArticlesProject.database.articles import get_all_articles
from ArticlesProject.controllers.controller import filter_articles, filter_categories
from django.urls import resolve


def get_current_path(request):
    return {
        'current_path': resolve(request.path_info).url_name
    }


def index(request):
    # se obtienen los artículos y la similitud coseno
    articles_list = get_all_articles()[:10]
    return render(request, 'index.html', {
        'articles_list': articles_list,
    })


def search_view(request):
    # se obtiene el parámetro de consulta 'kw', si es que existe, o usa un valor por defecto
    keywords = request.GET.get('kw', '')
    keywords_list = keywords.split(' ')  # se convierte el string en una lista de keywords

    filtered_articles = filter_articles(keywords_list)[:50]
    filtered_categories = filter_categories(filtered_articles)

    return render(request, 'search.html', {
        'keywords': keywords,
        'filtered_articles': filtered_articles,
        'filtered_categories': filtered_categories
    })


def about(request):
    authors = [
        {
            'fullname': 'Jamutaq Piero Ortega Vélez',
            'code': 'u201911703',
        },
        {
            'fullname': 'Lucía Guadalupe Aliaga Trevejo',
            'code': 'u20211a452',
        },
        {
            'fullname': 'Luis Alberto Siancas Reategui',
            'code': 'U2021g156',
        },
    ]
    return render(request, 'about.html', {
        'authors': authors,
    })
