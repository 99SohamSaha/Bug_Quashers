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

def train_clus_model():
    data_df = pd.read_csv(r'error_data_csv.csv')
    #print(data_df.info())
    #print('******************************')
    #print('Shape of the data : ')
    #print(data_df.shape)
    #print('******************************')
    #print(data_df.head())

    reduced_data = []
    for i in range(0,data_df.shape[0]):
        line = data_df['Error Description'].iloc[i]
        line = line.lower()
        line = re.sub(r'\d+', '', line)
        translator = str.maketrans('', '', string.punctuation)
        line = line.translate(translator)
        line = " ".join(line.split())
        reduced_data.append(line)
        #print(line)

    #adding the column of clean data in the actual dataframe beside raw data
    data_df["Clean data"] = reduced_data
    #converting the clean data lines to numerical vectors
    embedder = SentenceTransformer('distilbert-base-nli-mean-tokens')
    reduced_data_embeddings = embedder.encode(reduced_data)
    #displaying the original dataframe with the added clean data column
    #print(data_df.head())
    #print('***************************************************')
    #displaying the numerical feature vectors
    #print('Numerical feature arrays :')
    #print(reduced_data_embeddings)
    #print('shape of the numerical feature array :')
    #print(reduced_data_embeddings.shape)

    #finding the ideal number of clusters using Elbow method
    scores=[]
    for i in range(15):
        scores.append(KMeans(n_clusters=i+2).fit(reduced_data_embeddings).inertia_) 
    x=np.arange(2,17)
    #plt.plot(x,scores)
    #plt.xlabel('Number of clusters')
    #plt.ylabel('inertia')

    #from the Graph, the optimal number of clusters can be taken as 6 from the elbow method
    #applying the clustering algorithm
    num_of_clusters = 6
    cl_model = KMeans(n_clusters=num_of_clusters)
    cl_model.fit(reduced_data_embeddings)
    labels = cl_model.labels_
    #adding the cluster column to the original dataframe
    data_df['cluster']=cl_model.labels_
    #print(data_df.head())

    #sorting the dataset by cluster values
    #displaying the content of separate clusters
    rslt_df = data_df.sort_values(by = 'cluster')
    clus = 0
    count = 0
    cluster_length = []
    #print('Contents of cluster '+str(clus))

    for i in range(0,rslt_df.shape[0]):
        if rslt_df['cluster'].iloc[i]==clus :
            #print(rslt_df['Error Description'].iloc[i])
            count = count + 1
        else :
            clus = clus + 1
            #print('**************************************************************')
            #print('Contents of cluster '+str(clus))
            #print(rslt_df['Error Description'].iloc[i])
            cluster_length.append(count)
            count=1
    cluster_length.append(count)

    #printing the details of the error clusters
    #print('Details about the error clusters :')
    #print('*******************************************************')
    #for i in range(0,num_of_clusters):
        #print('Number of errors in cluster '+str(i)+' : '+str(cluster_length[i]))
        #print('-----------------------------------------------------')