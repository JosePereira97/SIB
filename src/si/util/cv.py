from .util import train_test_split
import numpy as np
import intertools


class CrossValidationScore:

    def __init__(self, model, dataset, **kwargs):
        self.model = model
        self.dataset = dataset
        self.cv = kwargs.get('cv', 3)
        self.split = kwargs.get('split', 0.8)
        self.train_scores = None
        self.test_scores = None
        self.ds = None

    def run(self):
        train_scores = []
        test_scores = []
        ds = []
        for _ in range(self.cv):
            train, test = train_test_split(self.dataset, self.split)
            ds.append((train, test))
            self.model.fit(train)
            train_scores.append(self.model.cost())
            test_scores.append(self.model.cost(test.X, test.Y))
        self.train_scores = train_scores
        self.test_scores = test_scores
        self.ds = ds
        return train_scores, test_scores

    def toDataframe(self):
        import pandas as pd
        assert self.train_scores and self.test_Scores, 'Need to run code first'
        return pd.DataFrame({'Train Scores': self.train_scores, 'Test Scores': self.test_scores})



class GridSearchCV:

    def __init__(self, model, dataset, parameters, **kwargs):
        self.model = model
        self.dataset = dataset
        hasparam = [hasattr(self.model, param) for param in parameters]
        if np.all(hasparam):
            self.parameters = parameters
        else:
            index = hasparam.index(False)
            keys = list(parameters.keys())
            raise ValueError (f"Wrong parameters: {keys[index]}")
        self.kwargs = kwargs
        self.results = None
    
    def run(self):
        self.results = []
        attrs = list(self.parameters.keys())
        values = list(self.parameters.values())
        for conf in itertools.product(*value):
            for i in range(len(attrs)):
                setattr(self.model, attrs[i], conf[i])
            scores = CrossValidationScore(self.model, self.dataset, **self.kwargs).run()
            self.results.append((conf, scores))
        return self.results

    def toDataFrame(self):
        import pandas as pd
        assert self.results, 'The grid search needs to be ran'
        date = dict()
        for i,k in enumerate(self.parameters.keys()):
            v=[]
            for r in self.results:
                v.append(r[0][i])
            date[k] = v
        return 