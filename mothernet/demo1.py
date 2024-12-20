from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from mothernet.prediction import MotherNetClassifier, EnsembleMeta

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# MotherNetClassifier encapsulates a single instantiation of the model.

classifier = MotherNetClassifier(device="cpu", model_path="path/to/model.pkl")

classifier.fit(X_train, y_train)
y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)

print("Accuracy", accuracy_score(y_test, y_eval))

# Ensembling as described in the TabPFN paper an be performed using the EnsembleMeta wrapper
ensemble_classifier = EnsembleMeta(classifier)
# ...
