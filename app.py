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


# FAQ route
@app.route('/FAQ')
def FAQ():
    return render_template('FAQ.html')


# scope route
@app.route('/scope')
def scope():
    return render_template('scope.html')


# master run API route
@app.route('/api/v1.0/run')
def run():
    # TODO:
    # run model
    err_count_arr, result = model.train_clus_model()
    # update global values
    # return required value
    result = result.drop(['Clean data'], axis=1)
    return err_count_arr, result.to_json()


# get error count from cluster number route
@app.route('/api/v1.0/get-err-count/<cluster_no>')
def get_error_count(cluster_no):
    # TODO: get count from global value and return actual data
    # dummy count
    error_count = int(cluster_no) + 10
    return str(error_count)


# upload csv, train model, and then return results
@app.route("/cluster/run", methods=["POST"])
def upload():
    if request.method == "POST":
        first_key = next(iter(request.files))
        file = request.files[first_key]
        if file.filename != "":
            filename = file.filename
            filePath = os.path.join("uploads", filename)
            file.save(filePath)
            
            # run model for dynamic file
            err_count_arr, result = model.train_clus_model(filePath)
            result = result.drop(['Clean data'], axis=1)
            return render_template('cluster.html', err_count_arr0 = err_count_arr[0], err_count_arr1 = err_count_arr[1], err_count_arr2 = err_count_arr[2], err_count_arr3 = err_count_arr[3], err_count_arr4 = err_count_arr[4], err_count_arr5 = err_count_arr[5], result0 = (result[result.cluster==0])['Error Description'].values, result1 = (result[result.cluster==1])['Error Description'].values, result2 = (result[result.cluster==2])['Error Description'].values, result3 = (result[result.cluster==3])['Error Description'].values, result4 = (result[result.cluster==4])['Error Description'].values, result5 = (result[result.cluster==5])['Error Description'].values)

        return "<h1>No file selected!</h1>"


# driver function
if __name__ == '__main__':
    app.run(debug=True)
