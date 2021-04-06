
import numpy as np
from scipy import spatial

if __name__ == "__main__":
    glove = "glove.6B/glove.6B.50d.txt"
else:
    glove = "../glove.6B/glove.6B.50d.txt"


class Preprocess():

    def __init__(self):

        self.embeddings_dict = {}
    	        
    def parse(self):

        with open(glove, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings_dict[word] = vector
    
    def find_emb(self,word):
        k = lambda word: self.embeddings_dict[word]
        return k(word)

    def find_close(self,emb):
        k = lambda word: self.embeddings_dict[word]

if __name__ == "__main__":
        
    pre = Preprocess()
    pre.parse()
    #print(pre.find_emb(["business main"]))
    print(pre.embeddings_dict["ai"])



