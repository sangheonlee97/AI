from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.svm import LinearSVC


datasets = load_diabetes()
X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1226)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
allAlgorithms = all_estimators(type_filter='classifier')
best = [0, 'no', 9999]
for i, v in enumerate(allAlgorithms):
    name, algorithm = v
    try:
        model = algorithm()
        
        model.fit(X_train,y_train)

        # results = model.score(X_test, y_test)

        # print("model : ", model, ", ",'score : ', results)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print("idx : ", i, ", model : ", name, ", ","acc : ", acc)
        if best[0] < acc:
            best[0] = acc
            best[1] = name
            best[2] = i
    except:
        print("qudtlsrkxdms dkfrhflwma : ", name)
print("best model : ", best[1], ", idx [", best[2],"]", "\nbest acc : ", best[0])