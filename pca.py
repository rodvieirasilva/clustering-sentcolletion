from sklearn.decomposition import PCA
import csv
import json
from textdict import TextDict
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

basec = ["#896800",
"#c992ff",
"#00b86c",
"#f436a7",
"#017dd2",
"#b73500"]

def plotpca(title, pca, words, y_pred, classnames):
    plt.figure(num=title)
    plt.title(title)
    XPCA = pca.transform(words)
    colors = [basec[c] for c in y_pred]

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
    dictclass = {"-1":0,"1":1}
    dicttheme = {"game":0,"movie":1, "smartphone":2}
    Ys = [ [  dictclass[item["class"]], dicttheme[item["theme"]]  ] for item in complete]
    Yclazz = [item[0] for item in Ys]
    Ytheme = [item[1] for item in Ys]
    dictthemeclass = {0:{0:0,1:1},1:{0:2, 1:3}, 2:{0:4, 1:5}}
    Yclasstheme = [dictthemeclass[item[1]][item[0]] for item in Ys]    
    plotpca("pca_class", pca, bagofwords, Yclazz , ["negative", "positive"])
    plotpca("pca_theme", pca, bagofwords, Ytheme, ["game", "movie", "smartphone"])
    plotpca("pca_classtheme", pca, bagofwords, Yclasstheme , ["negative_game","positive_game","negative_movie","positive_movie", "negative_smartphone", "positive_smartphone"])
    plt.show()
    print("Finish")

if __name__ == '__main__':
    main()