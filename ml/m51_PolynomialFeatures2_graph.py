import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
plt.rcParams['font.family'] = 'Malgun Gothic'

np.random.seed(888)
X = 2 * np.random.rand(100, 1) - 1      # -1 ~ 1 범위의 난수 생성
y = 3 * X**2 + 2 * X + 1 + np.random.randn(100, 1)

pf = PolynomialFeatures(degree=15, include_bias=False)
X_p = pf.fit_transform(X)

model = LinearRegression()
model2 = LinearRegression()

model.fit(X, y)
model2.fit(X_p, y)

print("acc : ", model.score(X, y))
print("acc : ", model2.score(X_p, y))

plt.scatter(X,y, c='blue', label='원래 데이터')
plt.xlabel("X")
plt.ylabel("y")
plt.title('Polynomial Regression Example')

X_plot = np.linspace(-1, 1, 100).reshape(-1, 1)
X_plot_poly = pf.transform(X_plot)
y_plot = model.predict(X_plot)
y_plot2 = model2.predict(X_plot_poly)
plt.plot(X_plot, y_plot, c='red',label='Poly' )
plt.plot(X_plot, y_plot2, c='green', label='그냥')
plt.legend()
plt.show()