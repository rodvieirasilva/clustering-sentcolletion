import matplotlib.pyplot as plt
import numpy as np
from util import mkdir

class SimpleWordCloud:
    DEFAULT_PREFIX = 'wordcloud/'
    prefix = DEFAULT_PREFIX
    wordcloud = None
    
    def __init__(self):
        pass

    def plot(self, text, filename):
        # Generate a word cloud image
        allText = ' '.join(text)
        filename = self.prefix + filename
        mkdir(filename)
        try:
            from wordcloud import WordCloud #--> Módulo wordcloud
            self.wordcloud = WordCloud(width=1024, height=768, background_color="white", stopwords=[], collocations=False).generate(allText)
            self.wordcloud.to_file(filename)
            
        except Exception as e:
            print("WordCloud não suportado, Error: " + str(e))
    
    def plotLabels(self, texts, labels, name, title):
        npTexts = np.array(texts)
        clusters = np.unique(labels)
        self.prefix = name + '/' + self.DEFAULT_PREFIX
        for cluster in clusters:
            text = npTexts[labels == cluster]
            self.plot(text, 'wordcloud-' + title + '-' + str(cluster) + '.png')
        self.prefix = self.DEFAULT_PREFIX
    
    def show(self):
        plt.figure()
        # Display the generated image:
        # the matplotlib way:    
        plt.imshow(self.wordcloud, interpolation='bilinear')
        plt.axis("off")
        # lower max_font_size
        plt.show()

def main():
    wordCloud = SimpleWordCloud()


if __name__ == '__main__':
    main()            