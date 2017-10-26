from sklearn.decomposition import PCA
import csv
import json
from textdict import TextDict
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

basec = ['#000080', '#FF00FF', '#40E0D0', '#006400']

def getcor(y):
    basec = ['#000080', '#FF00FF', '#40E0D0', '#006400']
    if(y=="-1"):
        return basec[0]
    return basec[int(y)]

def plotpca(pca, words, y_pred, classnames):
    plt.figure()
    XPCA = pca.transform(words)
    colors = [getcor(c) for c in y_pred]

    patches = [mpatches.Patch(color=basec[i], label=classn) for i,classn in enumerate(classnames)]
    plt.legend(handles=patches)

    plt.scatter(XPCA[:, 0], XPCA[:, 1], s=30, color=colors)
    
    plt.legend()

def main():
    with open('sklearn_bagofwords.json') as json_data:
        bagofwords = json.load(json_data)
    with open('complete.json') as json_data:
        complete = json.load(json_data)

    pca = PCA(n_components=2, svd_solver='full')
    pca.fit(bagofwords)

    #{"theme": "game", "product": "archeage", "tweet": "@tweetmee lol atleast i have csgo and archeage to keep me atleast a tad bit busy xp ahah", "class": "1"}
    dicts = {"game":0,"movie":1, "smartphone":2}
    Ys = [ [  item["class"], dicts[item["theme"]]  ] for item in complete]
    Yclazz = [item[0] for item in Ys]
    Ytheme = [item[1] for item in Ys]
    plotpca(pca, bagofwords,Yclazz , ["negative", "positive"])
    plotpca(pca, bagofwords, Ytheme, ["game", "movie", "smartphone"])
    plt.show()
    print("Finish")

if __name__ == '__main__':
    main()