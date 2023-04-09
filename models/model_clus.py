import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sentence_transformers
from sentence_transformers import SentenceTransformer
#import nltkp
import string
import re
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt
#import seaborn as sns

def train_clus_model(csv_file):
    data_df = pd.read_csv(csv_file)

    reduced_data = []
    for i in range(0,data_df.shape[0]):
        line = data_df['Error Description'].iloc[i]
        line = line.lower()
        line = re.sub(r'\d+', '', line)
        translator = str.maketrans('', '', string.punctuation)
        line = line.translate(translator)
        line = " ".join(line.split())
        reduced_data.append(line)
    
    #adding the column of clean data in the actual dataframe beside raw data
    data_df["Clean data"] = reduced_data
    #converting the clean data lines to numerical vectors
    embedder = SentenceTransformer('distilbert-base-nli-mean-tokens')
    reduced_data_embeddings = embedder.encode(reduced_data)

    #finding the ideal number of clusters using Elbow method
    scores=[]
    for i in range(15):
        scores.append(KMeans(n_clusters=i+2).fit(reduced_data_embeddings).inertia_) 
    x=np.arange(2,17)

    #from the Graph, the optimal number of clusters can be taken as 6 from the elbow method
    #applying the clustering algorithm
    num_of_clusters = 6
    cl_model = KMeans(n_clusters=num_of_clusters)
    cl_model.fit(reduced_data_embeddings)
    labels = cl_model.labels_
    #adding the cluster column to the original dataframe
    data_df['cluster']=cl_model.labels_

    #sorting the dataset by cluster values
    #displaying the content of separate clusters
    rslt_df = data_df.sort_values(by = 'cluster')
    clus = 0
    count = 0
    err_count_arr = []
    
    for i in range(0,rslt_df.shape[0]):
        if rslt_df['cluster'].iloc[i]==clus :
            #print(rslt_df['Error Description'].iloc[i])
            count = count + 1
        else :
            clus = clus + 1
            #print('**************************************************************')
            #print('Contents of cluster '+str(clus))
            #print(rslt_df['Error Description'].iloc[i])
            err_count_arr.append(count)
            count=1
    # error count array
    err_count_arr.append(count)
    return err_count_arr, rslt_df