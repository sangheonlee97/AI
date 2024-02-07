from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



datasets = load_iris()
X = datasets.data
y = datasets.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y )
models = []
models.append(LinearSVC())
models.append(Perceptron())
models.append(LogisticRegression())
models.append(KNeighborsClassifier())
models.append(DecisionTreeClassifier())
models.append(RandomForestClassifier())

for model in models:

    model.fit(X_train,y_train)

    results = model.score(X_test, y_test)

    print("model : ", model, ", ",'score : ', results)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("model : ", model, ", ","acc : ", acc)
    print("\n")