from flask import Flask, render_template, request
import models.model_clus as model
import string
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


app = Flask(__name__)


# index route
@app.route('/')
def index():
    return render_template('index.html')


# master run API route
@app.route('/api/v1.0/run')
def run():
    # TODO:
    # run model
    model.train_clus_model()
    # update global values
    # return required value
    return

# get error count from cluster number route
@app.route('/api/v1.0/get-err-count/<cluster_no>')
def get_error_count(cluster_no):
    # TODO: get count from global value and return actual data
    # dummy count
    error_count = int(cluster_no) + 10
    return str(error_count)

# driver function
if __name__ == '__main__':
    app.run(debug=True)
