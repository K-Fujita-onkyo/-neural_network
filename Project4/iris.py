import seaborn as sns
import numpy as np

# データセット読み込み
from sklearn.datasets import load_iris
 
iris_data = load_iris()
iris = iris_data.data

print(iris)

from sklearn.mixture import GaussianMixture as gm 
model = gm(n_components=3,covariance_type='full')
model.fit(iris)
y_gmm = model.predict(iris)
print(y_gmm)