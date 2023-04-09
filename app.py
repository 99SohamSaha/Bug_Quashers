from flask import Flask, render_template, request
import models.model_clus as model
import string
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import os


app = Flask(__name__)


# index route
@app.route('/')
def index():
    return render_template('buganalyzer.html')


# cluster route
@app.route('/cluster')
def cluster():
    return render_template('cluster.html')


# master run API route
@app.route('/api/v1.0/run')
def run():
    # TODO:
    # run model
    err_count_arr, result = model.train_clus_model()
    # update global values
    # return required value
    result = result.drop(['Clean data'], axis=1)
    return result.to_json()


# get error count from cluster number route
@app.route('/api/v1.0/get-err-count/<cluster_no>')
def get_error_count(cluster_no):
    # TODO: get count from global value and return actual data
    # dummy count
    error_count = int(cluster_no) + 10
    return str(error_count)


@app.route("/upload", methods=["POST"])
def upload():
    if request.method == "POST":
        first_key = next(iter(request.files))
        file = request.files[first_key]
        if file.filename != "":
            filename = file.filename
            file.save(os.path.join("uploads", filename))
            with open(os.path.join("uploads", filename), "r") as f:
                contents = f.read()
            return f"<h1>CSV file contents:</h1><pre>{contents}</pre>"
    return "<h1>No file selected!</h1>"


# driver function
if __name__ == '__main__':
    app.run(debug=True)
