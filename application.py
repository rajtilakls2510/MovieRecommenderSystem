from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd

app=Flask(__name__)
nn=pickle.load(open('engine.pkl','rb'))
movie_tag_pivot=pickle.load(open('movie_tag_pivot_table.pkl','rb'))
scores_movie=pickle.load(open('movie_names.pkl','rb'))


@app.route('/')
def index():
    return render_template('index.html')




def search_movie(search):
    # Extracting all the words
    search = search.strip().split()

    # Creating random substrings
    searchsub = []

    for word in search:
        if len(word) > 4:
            for i in range(int(0.8 * len(word))):
                low = np.random.randint(low=0, high=len(word))
                high = np.random.randint(low=low, high=len(word))
                if high - low <= 0.8 * len(word):
                    searchsub.append(word[low:high + 1].lower())
        else:
            low = 0
            high = len(word) - 1
            searchsub.append(word[low:high + 1].lower())

    # Sorting the substrings based on their lengths: Lowest length first
    lengths = []
    searchsub = np.array(searchsub)
    for sub in searchsub:
        lengths.append(len(sub))

    searchsub = searchsub[np.array(lengths).argsort()]

    # Finding all the rows that match the substrings
    results = pd.DataFrame(scores_movie[scores_movie['title'].str.contains(searchsub[0], case=False)], columns=scores_movie.columns)

    for subs in searchsub:
        new = scores_movie[scores_movie['title'].str.contains(subs, case=False)]
        if new.shape[0] != 0:
            results = pd.merge(results, new, how='inner')
    return results


@app.route('/search', methods=['POST'])
def search():
    search=request.form.get('search')
    search_results=search_movie(search).values
    return render_template('index.html',search=1,search_results=search_results)

def recommend(movie_id):
    distances,suggestions=nn.kneighbors(movie_tag_pivot.loc[movie_id,:].values.reshape(1,-1),n_neighbors=16)
    return movie_tag_pivot.iloc[suggestions[0]].index.to_list()

@app.route('/movie',methods=['GET'])
def movie():
    movie_id=int(request.args.get('movie_id'))
    recommendations = recommend(movie_id)
    movies_rec=[]
    for movie_id in recommendations:
        movies_rec.append(scores_movie[scores_movie['movieId']==movie_id].values[0])

    return render_template('show_recommendations.html',movies=movies_rec)

if __name__ == '__main__':
    app.run()