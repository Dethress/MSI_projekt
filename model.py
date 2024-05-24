import numpy as np

from LogisticRegression import OurLogisticRegression
from mlp import OurMLP


from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone

from sklearn.datasets import make_classification, load_breast_cancer





datasets_names = ["make_classification", "breast_cancer", "indiand_diabetes"]



results = np.zeros((3, 10, 4))
data = np.loadtxt("pima_indians_diabetes.csv", delimiter=",")
X = data[:, :-1]
y = data[:, -1]



datasets = [make_classification(
            n_samples=500,
            n_features=5,
            n_classes=2),
            
            load_breast_cancer(return_X_y=True),

            (X, y)]


rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2)

for dataset_i, dataset in enumerate(datasets):
    X, y = dataset


    for fold, (train, test) in enumerate(rskf.split(X, y)):
        clasisfiers = [OurLogisticRegression(), OurMLP(input_size=X[train].shape[1], hidden_size=5, output_size=1), LogisticRegression(), MLPClassifier()]
        for cls_i, cls in enumerate(clasisfiers):
            if cls_i == 1:
                model = cls
            model = cls
            model.fit(X[train], y[train])
            predicts = model.predict(X[test])
            results[dataset_i, fold, cls_i] = balanced_accuracy_score(y[test], predicts)




results = np.mean(results, axis=0)
 
np.savetxt("results.csv", results, delimiter=",")
 
