import numpy as np
from util import FNCData,pipeline_test,pipeline_train

if __name__ != "__main__":
    file_train_instances = "../train_stances.csv" 
    file_train_bodies = "../train_bodies.csv"                                                
    file_test_instances = "../test_stances_unlabeled.csv"                                    
    file_test_bodies = "../test_bodies.csv"                                                  
    file_predictions = '../predictions_test.csv'
else:
    file_train_instances = "train_stances.csv"                                                
    file_train_bodies = "train_bodies.csv"                                                    
    file_test_instances = "test_stances_unlabeled.csv"                                        
    file_test_bodies = "test_bodies.csv"                                                      
    file_predictions = 'predictions_test.csv'


class DataLoader():

    def __init__(self):

        self.raw_train = FNCData(file_train_instances, file_train_bodies)
        self.raw_test = FNCData(file_test_instances, file_test_bodies)

    
    def dataset(self):
        
        # optimize
        self.train_headline = [i["Headline"] for i in self.raw_train.instances]
        self.train_stance = [i["Stance"] for i in self.raw_train.instances]
        self.body_id  = [i["Body ID"] for i in self.raw_train.instances]
        self.train_body = [self.raw_train.bodies[i] for i in self.body_id]

        return self.train_headline ,self.train_stance ,self.train_body

    def test(self):
        
        # optimize
        self.test_headline = [i["Headline"] for i in self.raw_test.instances]
        self.body_id  = [i["Body ID"] for i in self.raw_test.instances]
        self.test_body = [self.raw_test.bodies[i] for i in self.body_id]

        return self.test_headline,self.body_id ,self.test_body


if __name__ == "__main__":

    dataloader = DataLoader()
    head,stance,body = dataloader.dataset()




