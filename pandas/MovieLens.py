import pandas as pd
import numpy as np


# ratings = pd.read_csv('data/ratings.csv', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], sep=':')
tags = pd.read_csv('data/tags.csv', header=None, names=['UserID', 'movieId', 'tag', 'Timestamp'], sep=':')
movies = pd.read_csv('data/movies.csv', header=None, names=['MovieID', 'Title', 'Genres'], sep=':')

data = pd.merge(tags, movies)
print(data)
print(data[data.UserID == 1])
