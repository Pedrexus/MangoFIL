import pandas as pd
import matplotlib.pyplot as plt

path = r'hist.csv'

df = pd.read_csv(path)
cls = df.iloc[:, 1]

cls = cls.replace(['apenas Antracnose', 'Colapso e Antracnose'], 'Antracnose')
cls = cls.replace(['sem Colapso e sem Antracnose', 'apenas Colapso'], 'Sem Antracnose')

cls.hist()
plt.show()