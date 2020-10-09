# Einfache lineare Regression auf Basis einer bereits gefilterten .csv
# Diese enth‰lt Informationen von Gebrauchtwagen aus der Platform "eBay Kleinanzeigen".

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

df = pd.read_csv("./autos_prepared.csv")

print(df.head())


# Scatter-Plot zeichnen

plt.scatter(df["kilometer"], df["price"])
plt.show()

# Lineare Regression 

model = LinearRegression()
model.fit(df[["kilometer"]], df[["price"]])

print("Intercept: " + str(model.intercept_))
print("Coef: " + str(model.coef_))

# Y = Intercept + Coef * [km] 

predicted = model.predict([[0], [130000]])
print(predicted)

# Vorhersage in Grafik einzeichnen

plt.scatter(df["kilometer"], df["price"])
plt.plot([0, 130000], predicted, color="red")
plt.show()

# Vorhersage f√ºr 50.000km machen

print(model.predict([[50000]]))