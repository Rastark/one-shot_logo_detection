import numpy as np

class Evaluator:

    def __init__(self, model: Model):
        self.model = model
        self.dataset = model.dataset    # the Dataset may be useful to convert ids to names

    def eval(self,
            samples: np.array,
            write_output:bool = False):

        # run prediction on the samples

    def _write_output(self, samples, ranks, predictions):

    @staticmethod
    def precision:

    @staticmethod
    def recall:
    
    @staticmethod
    def accuracy:
