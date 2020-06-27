from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from models.constant import FPR,TPR,DUMMY_CLASSIFIER
from matplotlib import pyplot
from sklearn.model_selection import RepeatedKFold


class Model:

    def __init__(self,name,instance,parameters):
        self.name = name
        self.instance = instance
        self.parameters = parameters
        self.folds = []

    def train(self,X,Y,verbose=0,splits=5,repeats=1):
        self.instance = GridSearchCV(self.instance,self.parameters,refit=True,verbose=verbose,cv=RepeatedKFold(n_splits=splits,n_repeats=repeats))
        self.instance = self.instance.fit(X,Y)
        for train,test in self.instance.cv.split(X):
            self.folds.append((train,test))

    def predict(self,dataset):
        return self.instance.predict(dataset)

    # Considering positive class by default
    def draw_roc_curve(self, X, Y, target_class=1):
        ns_probs = [0 for _ in range(len(Y))]
        lr_probs = self.instance.predict_proba(X)
        lr_probs = lr_probs[:, target_class]
        ns_fpr, ns_tpr, _ = roc_curve(Y, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(Y, lr_probs)
        pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label=DUMMY_CLASSIFIER)
        pyplot.plot(lr_fpr, lr_tpr, marker='.', label=self.name)
        pyplot.xlabel(FPR)
        pyplot.ylabel(TPR)
        pyplot.legend()
        pyplot.show()