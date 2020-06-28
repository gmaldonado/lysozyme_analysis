from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from models.constant import FPR,TPR,DUMMY_CLASSIFIER
from matplotlib import pyplot
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix

class Model:

    def __init__(self,name,instance,parameters):
        self.name = name
        self.instance = instance
        self.parameters = parameters

    def train(self,X,Y,verbose=False,splits=5):
        self.instance = GridSearchCV(self.instance,self.parameters,refit=True,cv=StratifiedKFold(n_splits=splits))

        if verbose:
            print("Starting verbose mode")
            #TODO check the object references and this implementation in general
            temporary = Model(self.name,self.instance,self.parameters)
            i = 1
            for train,test in temporary.instance.cv.split(X,Y):
                print("Split {}: \n Train: {} Test: {} ".format(i, train,test))
                temporary.instance = temporary.instance.fit(X.iloc[train],Y.iloc[train])
                pred = temporary.instance.predict(X.iloc[test])
                print(confusion_matrix(Y.iloc[test],pred))
                print(classification_report(Y.iloc[test], pred))
                temporary.draw_roc_curve(X.iloc[test],Y.iloc[test])
                i+=1

        self.instance = self.instance.fit(X,Y)



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