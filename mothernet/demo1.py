from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from mothernet.utils import get_mn_model
from mothernet.prediction import MotherNetClassifier, EnsembleMeta


def train_and_evaluate(device="cpu", test_size=0.33, random_state=42):

    # 加载数据集并划分训练集和测试集：
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 模型参数的检查点文件路径：
    model_string = "mn_d2048_H4096_L2_W32_P512_1_gpu_warm_08_25_2023_21_46_25_epoch_3940_no_optimizer.pickle"
    model_path = get_mn_model(model_string)
    # 初始化分类器模型：
    classifier = MotherNetClassifier(device=device, path=model_path)

    # 模型训练：
    classifier.fit(X_train, y_train)

    # 模型预测与评估：
    y_eval = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_eval)
    print("Accuracy", accuracy)

    # 请注意，MotherNetClassifier 并不进行任何集成，你需要使用 EnsembleMeta 来获得论文所述的集成。
    # 集成学习：
    ensemble_classifier = EnsembleMeta(classifier)
    # 集成学习的具体实现可以在这里添加：

    return accuracy, ensemble_classifier


# 调用函数进行训练和评估：
train_and_evaluate(device="cpu")
