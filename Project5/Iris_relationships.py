import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['kind'] = pd.DataFrame(data=iris.target)
mapping = {0:'setosa', 1:'versicolor', 2:'virginica'}
df['kind'] = df['kind'].map(mapping)

sns.countplot('kind',data=df)
sns.pairplot(df, hue='kind')
plt.show()