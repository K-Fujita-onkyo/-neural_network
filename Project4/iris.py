import seaborn as sns
import numpy as np

#Loading data
from sklearn.datasets import load_iris
 
iris_data = load_iris()
iris = iris_data.data

print(iris)

#unsupervised learning with Gaussian Mixture Model
from sklearn.mixture import GaussianMixture as gm 
model = gm(n_components=3,covariance_type='full')
model.fit(iris)
gmm = model.predict(iris)
print(gmm)