import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.othermod.betareg as betareg


actuals = pd.read_pickle("C:/Users/kpfs/Projects/Windpower-Correlation-Structure/Data/NABQR/actuals.pkl")
basis = pd.read_pickle("C:/Users/kpfs/Projects/Windpower-Correlation-Structure/Data/NABQR/corrected_ensembles.pkl")

actuals[actuals < 0.01] = 0.01


def trans(x):
    return x / 3600


def detrans(x):
    return 3600*x


endog = trans(actuals["DK1-onshore"])
exog = basis["DK1-onshore"]
exog.loc[:, "const"] = 1

train_size = int(len(exog)*0.8)
X_train = exog.iloc[:train_size]
y_train = endog.iloc[:train_size]
X_test = exog.iloc[train_size:]
y_test = endog.iloc[train_size:]


# %%
# no precision
betares = betareg.BetaModel(y_train, X_train).fit()


quants = pd.DataFrame({str(q): betares.get_distribution(X_test, exog_precision=np.ones(
    (len(X_test), 1))).ppf(q) for q in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]})
plt.figure(figsize=(14, 8), layout="tight")
plt.plot(y_test.index, quants["0.5"], color='black')
plt.plot(y_test.index, quants[["0.25", "0.75"]], color='blue', linestyle=":")
plt.plot(y_test.index, quants[["0.1", "0.9"]], color='blue', linestyle="--")
plt.plot(y_test.index, quants[["0.01", "0.99"]], color='red')
plt.scatter(y_test.index, y_test, color="black", marker="x")


# %%
betares2 = betareg.BetaModel(y_train, X_train, exog_precision=X_train).fit()
quants = pd.DataFrame({str(q): betares2.get_distribution(X_test, exog_precision=X_test).ppf(q)
                      for q in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]})

plt.figure(figsize=(14, 8), layout="tight")
plt.plot(y_test.index, quants["0.5"], color='black')
plt.plot(y_test.index, quants[["0.25", "0.75"]], color='blue', linestyle=":")
plt.plot(y_test.index, quants[["0.1", "0.9"]], color='blue', linestyle="--")
plt.plot(y_test.index, quants[["0.01", "0.99"]], color='red')
plt.scatter(y_test.index, y_test, color="black", marker="x")


# %%

means1 = betares.predict(X_test)
means2 = betares2.predict(exog=X_test, exog_precision=X_test)
pres1 = betares.predict(X_test, exog_precision=np.ones((len(X_test), 1)), which="precision")
pres2 = betares2.predict(X_test, exog_precision=X_test, which="precision")

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(means1)
plt.plot(means2)
plt.subplot(1, 2, 2)
plt.plot(pres1)
plt.plot(pres2)

plt.figure()
plt.scatter(means2, pres2)
