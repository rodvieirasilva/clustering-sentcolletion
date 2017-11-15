"""
-- Sent Collection v.1 para análise de agrupamento --
--                  Grupo 1                        --
--Marciele de Menezes Bittencourt                  --
--Rodrigo Vieira da Silva                          --
--Washington Rodrigo Dias da Silva                 --
-----------------------------------------------------
"""
class TextDict:
    def __init__(self, filename):
        self.dict = {}
        with open(filename) as file:
            for line in file:
                splitline = line.split('\t')
                self.dict[splitline[0].strip()] = splitline[-1].strip()
    def contains(self, word):
        lowered_word = word.lower()
        return lowered_word in self.dict

    def translate(self, word):
        lowered_word = word.lower()
        if lowered_word in self.dict:
            return self.dict[lowered_word]
        return word