import pandas as pd 
import numpy as np 
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request
import requests


# Initialize Flask app
app = Flask(__name__)

# Load the movie and credit data
movies = pd.read_csv("tmdb_5000_movies.csv")
cred = pd.read_csv("tmdb_5000_credits.csv")

# Merge movie and credit data
movies = movies.merge(cred, on="title")

# Select necessary columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Drop rows with missing data
movies.dropna(inplace=True)

# Function to convert genres (JSON string to list)
def convert(text):
    genres_list = json.loads(text)
    return [i["name"] for i in genres_list]

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Convert cast to list of top 4 actors
def convert_cast(text):
    cast_list = json.loads(text)
    return [i["name"] for i in cast_list[:4]]

movies['cast'] = movies['cast'].apply(convert_cast)

# Extract director from crew information
def get_director(crew):
    crew_list = json.loads(crew)
    director = [member for member in crew_list if member['job'] == 'Director']
    return [director[0]["name"]] if director else None

movies['crew'] = movies['crew'].apply(get_director)

# Split overview into words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Drop rows with any remaining missing data
movies.dropna(inplace=True)

# Clean data by removing spaces
def collapse(L):
    return [i.replace(" ", "") for i in L]

movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

# Combine all text features into a 'tags' column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Drop original columns used to create 'tags'
movies = movies.drop(columns=['overview', 'genres', 'keywords', 'cast', 'crew'])

# Join list into a single string for each movie
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

# Convert the 'tags' column into a matrix of token counts
cv = CountVectorizer(max_features=5000, stop_words='english')
vec = cv.fit_transform(movies['tags']).toarray()

# Compute cosine similarity between movies
similarity_matrix = cosine_similarity(vec)

# Function to get the top 'n' most similar movies
def get_similar_movies(movie_title, n=10):
    movie_title = movie_title.lower()
    
    movie_index = movies[movies['title'].str.lower() == movie_title].index

    if movie_index.empty:
        return f" "
    movie_index = movie_index[0]
    similarity_scores = list(enumerate(similarity_matrix[movie_index]))
    sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_similar_movies = sorted_similar_movies[1:n+1] 
    return [(movies.iloc[i[0]].title) for i in top_similar_movies]

# Example: Get the 10 most similar movies to "Avatar"
# movie_title = "Avatar"
# similar_movies = get_similar_movies(movie_title, n=10)

# Print the results
# for movie in similar_movies:
#     print(f"Movie: {movie}")


def retriveData(title):
	data = requests.get(f'http://www.omdbapi.com/?i=tt3896198&apikey=fbfb13bb&t={title}').json()
	return data

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to show recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['movie_title']
    similar_movies = get_similar_movies(movie_title, n=10)  
    
    posters = []  

    for movie in similar_movies:
        data = retriveData(movie)  
        if "Poster" in data and data["Poster"] != "N/A":
            posters.append(data["Poster"])  
        else:
            posters.append("https://via.placeholder.com/300x450?text=No+Image")

    return render_template('recommendations.html', 
                           movie_title=movie_title, 
                           similar_movies=similar_movies, 
                           posters=posters, zip = zip)

if __name__ == '__main__':
    app.run()
