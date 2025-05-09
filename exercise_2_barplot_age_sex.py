import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cleveland.csv', header=None)
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang',
              'oldpeak', 'slope', 'ca', 'thal', 'target']

df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
df['thal'] = df.thal.fillna(df.thal.mean())
df['ca'] = df.ca.fillna(df.ca.mean())

# barplot of age vs sex with hue = target

ax = sns.catplot(kind='bar', data=df, y='age', x='sex', hue='target')
ax.ax.set_xticks(np.arange(0, 5, 5))
plt.title("barplot of age vs sex with hue = target")
plt.show()
