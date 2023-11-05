class Article:
    def __init__(self, id, url, title, subtitle, image, claps, responses, reading_time, publication, date):
        self.id = id
        self.url = url
        self.title = title
        self.subtitle = subtitle
        self.image = image
        self.claps = claps
        self.responses = responses
        self.reading_time = reading_time
        self.publication = publication
        self.date = date

    def __str__(self):
        return f"Article: {self.id} - {self.title} - {self.subtitle} - {self.claps} - {self.date}"
