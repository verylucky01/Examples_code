from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from ticl.utils import get_mn_model
from ticl.prediction import MotherNetClassifier


def evaluate_model(model_string, test_size=0.33, random_state=42):
    """
    将数据加载、模型加载、训练、预测和评估的步骤整合至一个函数中，以提升代码的可读性和可维护性。
      - 这段代码针对乳腺癌诊断这一二分类问题，利用包含肿瘤大小、形状等多个特征的数据集，
      - 通过基于超网络 Transformer 架构的预训练深度学习模型进行分类，并对模型的准确率进行了评估。
    """

    # 加载数据集：
    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    print("X shape", X.shape)
    print(X)
    print("y shape", y.shape)
    print(y)

    # 划分数据集：
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 加载预训练模型：
    model_path = get_mn_model(model_string)
    classifier = MotherNetClassifier(device="cpu", path=model_path)

    # 微调训练模型：
    classifier.fit(X_train, y_train)

    # 预测与评估：
    y_eval = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_eval)

    return accuracy


# 评估模型：
model_string = "mn_d2048_H4096_L2_W32_P512_1_gpu_warm_08_25_2023_21_46_25_epoch_3940_no_optimizer.pickle"
accuracy = evaluate_model(model_string)
print("Accuracy", accuracy)
