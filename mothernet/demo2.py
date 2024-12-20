from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from ticl.utils import get_mn_model
from ticl.prediction import MotherNetClassifier, EnsembleMeta


X, y = load_breast_cancer(return_X_y=True)
print("X shape", X.shape)
print(X)
print("y shape", y.shape)
print(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# MotherNetClassifier encapsulates a single instantiation of the model.
# This will automatically download a model from blob storage
model_string = "mn_d2048_H4096_L2_W32_P512_1_gpu_warm_08_25_2023_21_46_25_epoch_3940_no_optimizer.pickle"
model_path = get_mn_model(model_string)
classifier = MotherNetClassifier(device="cpu", path=model_path)

classifier.fit(X_train, y_train)
y_eval = classifier.predict(X_test)

print("Accuracy", accuracy_score(y_test, y_eval))
