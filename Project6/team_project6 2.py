import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import time


def train_and_evaluation(datasets, pipelines):
    for dataset_name, dataset in datasets.items():
        X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        y = pd.Series(dataset.target, name='test_set')
        X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3, random_state=1)
        print(f"\nDATASET NAME: {dataset_name}")
        print(f'TRAIN AMOUNT(Data, Charctristic): {X_train.shape}')
        print(f' TEST AMOUNT(Data, Charctristic): {X_test.shape}\n')

        # ------- accuracy, precision, recall, f1_score for test_data------

        scores = {}
        for pipe_name, pipeline in pipelines.items():
            start = time.perf_counter()
            pipeline.fit(X_train, y_train)
            end = time.perf_counter()
            print("計測時間：" + str(end-start) + "(s)")
            print(f'MODEL NAME: {pipe_name}')
            print(classification_report(y_test, pipeline.predict(X_test)))


def pipeline():
    pipelines = {
        'SVM':
            Pipeline([('scl',StandardScaler()),
                      ('est',SVC(kernel='linear',random_state=1))]),
        'MLP':
            Pipeline([('scl',StandardScaler()),
                      ('est',MLPClassifier(hidden_layer_sizes=(10,),
                                           max_iter=1000,
                                           random_state=1))])
    }
    return pipelines


def dataset():
    datasets = {}
    datasets['iris'] = load_iris()
    datasets['wine'] = load_wine()
    datasets['breast_cancer'] = load_breast_cancer()
    datasets['digits'] = load_digits()
    return datasets


def main():
    datasets = dataset() # generate datasets of 3 components ("iris", "wine", "cancer")
    pipelines = pipeline() # generate pipelines consist of SVM and MLP
    train_and_evaluation(datasets, pipelines)


if __name__ == '__main__':
    main()
    
