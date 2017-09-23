import csv
import json
import numpy as np
#from wordcloud import WordCloud --> Módulo wordcloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer #--> Classe utilizada para converter o array de strings em um array de binários


def save(filename, data):
    with open(filename, 'w') as outfile:
            json.dump(data, outfile)

def readCSV(filename, theme, product):
    result = []
    with open(filename, 'r', encoding='utf-8') as reader:
        reader.readline()
        reader = csv.reader(reader, delimiter = ',', quotechar = '"')        
        for row in reader:
            result.append({"theme":theme, "product": product, "tweet": row[0], "class":row[1]})
    return result

def stats(name, data):
    print('-- {} --'.format(name))
    print('size: {}'.format(len(data)))
    print('--------')

def wordCloud(text):
    # Generate a word cloud image
    wordcloud = WordCloud(width=1024, height=768, background_color="white").generate(text)

    wordcloud.to_file('cloudword.png')
    # Display the generated image:
    # the matplotlib way:    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    # lower max_font_size
    # wordcloud = WordCloud(width=1024, height=768).generate(text)
    # plt.figure()
    # plt.imshow(wordcloud, interpolation="bilinear")
    # plt.axis("off")
    plt.show()    

def main():
    print('Started')
    archeage = readCSV("SentCollection - ARCHEAGE.csv", "game", "archeage")
    stats("archeage", archeage)
    save('archeage.json', archeage)
    hobbit = readCSV("SentCollection - HOBBIT.csv", "movie", "hobbit")
    stats("hobbit", hobbit)
    save('hobbit.json', hobbit)
    iphone6 = readCSV("SentCollection - IPHONE6.csv", "smartphone", "iphone6")
    stats("iphone6", iphone6)
    save('iphone6.json', iphone6)

    complete = archeage + hobbit + iphone6
    stats("complete", complete)
    save('complete.json', complete)
    
    tweets = [item['tweet'] for item in complete]    
    #wordCloud(' '.join(tweets)) --> Juntar as palavras no gráfico wordcloud

    allwords = set()
    for tweet in tweets:
        tweet = tweet.split()
        for word in tweet:
            allwords.add(word)       
    allwords = list(allwords)
    stats("allwords", allwords)
    save('allwords.json', allwords)
    
    #print(tweets)
    
    #Utilizando o CountVectorizer para criar a bag-of-words
    vectorizer = CountVectorizer(stop_words='english') #--> Inicializa a função CountVectorizer passando o parâmetro de extração de stop_words (Em inglês)

    save('sklearn_stoplist.json', list(vectorizer.get_stop_words()))   

    sklearn_bagofwords = vectorizer.fit_transform(tweets).todense()
    save('sklearn_bagofwords.json', sklearn_bagofwords.tolist())
    featuresnames = list(vectorizer.get_feature_names())
    save('sklearn_featuresnames.json', featuresnames)
    print('sklearn_featuresnames, size: {}'.format(len(featuresnames)))
    #save ('sklearn_vocabulary_.json', vectorizer.vocabulary_) #--> Exibe o vocabulario extraido pela função fit_transform()
    
    """Teste inicial de conversão
    corpus = ['UNC played Duke in basketball', 'Duke lost the basketball game']

    vectorizer = CountVectorizer()
    print (vectorizer.fit_transform(corpus).todense())
    print (vectorizer.vocabulary_)"""
    

    print('Finished')

if __name__ == '__main__':
    main()          