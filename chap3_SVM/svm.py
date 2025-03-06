# python: 3.5.2
# encoding: utf-8

import numpy as np
import cvxopt


def load_data(fname):
    """
    载入数据。
    """
    with open(fname, 'r') as f:
        data = []
        line = f.readline()
        for line in f:
            line = line.strip().split()
            x1 = float(line[0])
            x2 = float(line[1])
            t = int(line[2])
            data.append([x1, x2, t])
        return np.array(data)


def eval_acc(label, pred):
    """
    计算准确率。
    """
    return np.sum(label == pred) / len(pred)

class SVM():
    """
    SVM模型。
    """

    def __init__(self):
        # 请补全此处代码
        self.C = 1 # 惩罚因子
        self.w = None
        self.b = None
        pass

    def train(self, data_train):
        """
        训练模型。
        """
        # 请补全此处代码
        X, t = data_train[:, :-1], data_train[:, -1]
        N = X.shape[0] # alpha个数
        # 数据标准化
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X = (X - mean) / std
        # Gram矩阵H作为二次规划问题的二次项系数矩阵
        y = t.reshape(-1, 1)
        X_dash = y * X
        H = np.dot(X_dash, X_dash.T) 
        # 二次规划目标函数
        f = (-1) * np.ones(N) # 线性项  -sum(alpha)
        # 约束条件 不等式约束 Gx <= h
        G = np.vstack([-np.eye(N), np.eye(N)]) # (2N * N) 约束矩阵 约束条件的左侧
        LB = np.zeros(N) # alpha_n >= 0
        UB = np.ones(N) * self.C # alpha_n <= self.C
        h = np.hstack([-LB, UB]) # (2N,) # 约束条件的右侧
        # 约束条件 等式约束 Ax = b
        A = t.reshape((1, N)).astype(np.double) # (1 * N) 
        b = np.double(0)
        # 求解标准二次规划问题
        sol = cvxopt.solvers.qp(P=cvxopt.matrix(H), q=cvxopt.matrix(f), 
                                G=cvxopt.matrix(G), h=cvxopt.matrix(h),
                                A=cvxopt.matrix(A), b=cvxopt.matrix(b))
        # 得到最优alpha值和支持向量
        alpha = np.array(sol['x']).reshape((-1,)) 
        sv = np.where(alpha > 1e-4, True, False) # 支持向量的选择
        if ~sv.any():
            raise ValueError('No support vectors found.')
        # 最终分类器
        self.w = np.sum((alpha[sv] * y[sv]).dot(X[sv]), axis=0) #  (d,)
        self.b = np.mean(y[sv] - self.w.dot(X[sv].T))
        # 计算分类结果
        result = X.dot(self.w.T) + self.b
        result[result >= 0] = 1
        result[result < 0] = -1
        # 松弛变量
        #slack = np.where(alpha > self.C - 1e-6, True, False)
        # slack = np.maximum(0, 1 - y * (result))
        # print("Slack variables:", slack)  # 打印松弛变量

    def predict(self, x):
        """
        预测标签。
        """
        # 请补全此处代码
        # 数据标准化
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        x = (x - mean) / std

        result = x.dot(self.w) + self.b
        result[result >= 0] = 1
        result[result < 0] = -1
        return result


if __name__ == '__main__':
    # 载入数据，实际实用时将x替换为具体名称
    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'
    data_train = load_data(train_file)  # 数据格式[x1, x2, t]
    data_test = load_data(test_file)

    # 使用训练集训练SVM模型
    svm = SVM()  # 初始化模型
    svm.train(data_train)  # 训练模型

    # 使用SVM模型预测标签
    x_train = data_train[:, :2]  # feature [x1, x2]
    t_train = data_train[:, 2]  # 真实标签
    t_train_pred = svm.predict(x_train)  # 预测标签
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    t_test_pred = svm.predict(x_test)

    # 评估结果，计算准确率
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))
