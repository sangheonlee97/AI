from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV

parameters = [
    {"rf__n_estimators": [100, 200], "rf__max_depth": [6, 10, 12], "rf__min_samples_leaf": [3, 10]},
    {"rf__max_depth": [6, 8, 10, 12], "rf__min_samples_leaf": [3, 5, 7, 10]},
    {"rf__min_samples_leaf": [3, 5, 7, 10], "rf__min_samples_split": [2, 3, 5, 10]},
    {"rf__min_samples_split": [2, 3, 5, 10]},
]

datasets = load_iris()
X = datasets.data
y = datasets.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y )

pipe = Pipeline([('mm', MinMaxScaler()), ('rf', RandomForestClassifier( random_state=42))])
model = GridSearchCV(pipe, parameters, cv=5, verbose=1,)


model.fit(X_train,y_train)

results = model.score(X_test, y_test)

print("model : ", model, ", ",'score : ', results)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("model : ", model, ", ","acc : ", acc)
print("\n")