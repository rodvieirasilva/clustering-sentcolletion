class TextDict:
    def __init__(self, filename):
        self.dict = {}
        with open(filename) as file:
            for line in file:
                splitline = line.split('\t')
                self.dict[splitline[0]] = splitline[-1].strip()
    def contains(self, word):
        lowered_word = word.lower()
        return lowered_word in self.dict

    def translate(self, word):
        lowered_word = word.lower()
        if lowered_word in self.dict:
            return self.dict[lowered_word]
        return word