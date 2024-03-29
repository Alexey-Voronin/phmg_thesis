import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_pickle("results.pkl")

for r in list(df['resids'].values):
    plt.semilogy(r)

plt.show()
