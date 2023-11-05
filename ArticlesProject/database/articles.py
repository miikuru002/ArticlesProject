import csv
from ArticlesProject.models.Article import Article


def get_all_articles():
    aux = ''
    readed_articles = []

    # se lee el archivo csv
    try:
        with open(file='static/medium_data.csv', newline='', encoding="utf8") as file:
            data = csv.reader(file, delimiter=',')
            next(data, None)  # se omite el encabezado
            aux = list(data)
    except FileNotFoundError:
        print("[x] Error -> Archivo no encontrado")
    except Exception as e:
        print(f"[x] Error -> Ocurrió un error al leer el archivo: {e}")

    # se crea una lista de objetos de tipo Article
    for file in aux:
        readed_articles.append(
            Article(
                id=int(file[0]),
                url=str(file[1]),
                title=str(file[2]),
                subtitle=str(file[3]),
                image=str(file[4]),
                claps=int(file[5]),
                responses=str(file[6]),
                reading_time=int(file[7]),
                publication=str(file[8]),
                date=str(file[9]),
            )
        )

    return readed_articles
