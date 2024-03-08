import numpy as np
import pandas as pd
from sklearn.datasets import load_linnerud
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

X, y = load_linnerud(return_X_y=True)

# print(X, X.shape)
# print(y, y.shape)

model = RandomForestRegressor(random_state=42)
model.fit(X, y)
y_pred = model.predict(X)
print(model.__class__.__name__, "mae : ", round(mean_absolute_error(y, y_pred), 4))
print(model.predict([X[-1]]))

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
print(model.__class__.__name__, "mae : ", round(mean_absolute_error(y, y_pred), 4))
print(model.predict([X[-1]]))

model = XGBRegressor()
model.fit(X, y)
y_pred = model.predict(X)
print(model.__class__.__name__, "mae : ", round(mean_absolute_error(y, y_pred), 4))
print(model.predict([X[-1]]))

# model = LGBMRegressor()     # 에러
# model.fit(X, y)
# y_pred = model.predict(X)
# print(model.__class__.__name__, "mae : ", round(mean_absolute_error(y, y_pred), 4))
# print(model.predict([X[-1]]))

from sklearn.multioutput import MultiOutputRegressor
model = MultiOutputRegressor(LGBMRegressor(loss='MultiRMSE'))     # 성공
model.fit(X, y)
y_pred = model.predict(X)
print(model.__class__.__name__, "mae : ", round(mean_absolute_error(y, y_pred), 4))
print(model.predict([X[-1]]))
