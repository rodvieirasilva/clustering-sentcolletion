# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:10:41 2017

@author: Wash
"""

from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA



with open('sklearn_bagofwords.json') as json_data:
        bagofwords = json.load(json_data)
        
with open('sklearn_featuresnames.json') as json_data:
        featuresNames = json.load(json_data)
        
with open('complete.json') as json_data:
        complete = json.load(json_data)
        
with open('processed.json') as json_data:
        processed = json.load(json_data)

condensed = PCA(n_components=10)
condensed.fit_transform(bagofwords)
#print(condensed.components_)
#print(condensed.singular_values_)

#distance = pairwise_distances(condensed.components_, Y=None, metric='euclidean', n_jobs=1)
distance = pairwise_distances(bagofwords, Y=None, metric='euclidean', n_jobs=1)

Z = linkage(distance[0:20], 'single')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z, orientation="top", labels=featuresNames)

"""plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')"""

plt.tight_layout() #show plot with tight layout
plt.savefig('word_clusters_Single-Link.png') #save figure