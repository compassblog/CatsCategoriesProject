"""
    用逻辑回归实现一个猫分类器，输入一张图片x，预测图片是否为猫，输出该图片中存在猫的概率结果y
"""

# 导入用到的包
import numpy as np  # 进行科学计算的基础包
import matplotlib.pyplot as plt # 绘图库
import h5py     # 读取 HDF5 二进制数据格式文件的接口

# 导入数据
# 本项目训练及测试图片集是以 HDF5 二进制数据格式储存的


def load_dataset():
    train_dataset = h5py.File("train_cat.h5","r") # 读取训练数据
    test_dataset = h5py.File("test_cat.h5", "r") # 读取测试数据

    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # 原始训练集，train_set_x：图像矩阵
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # 原始训练集的标签集（y=0非猫,y=1是猫），train_set_y：标签列表

    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # 原始训练集，train_set_x：图像矩阵
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # 原始训练集的标签集（y=0非猫,y=1是猫），train_set_y：标签列表

    train_set_y_orig = train_set_y_orig.reshape((1,train_set_y_orig.shape[0])) #原始训练集的标签集设为（1*m）
    test_set_y_orig = test_set_y_orig.reshape((1,test_set_y_orig.shape[0])) #原始测试集的标签集设为（1*m）

    classes = np.array(test_dataset["list_classes"][:]) # list_classes：[b'non-cat' b'cat']
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# sigmoid函数:用sigmoid 函数来约束线性拟合（逻辑回归）中 y^ 的值域[0,1]


def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

# 初始化参数w,b ：定一个nx 维向量 w 和一个值 b 作为参数，以得到线性回归的表达式


def initialize_with_zeros(dim):
    w = np.zeros((dim,1)) # w为一个dim*1矩阵
    b = 0
    return w, b

# 计算 Y_hat,成本函数 J 以及 dw，db


def propagate(w, b, X, Y):
    m = X.shape[1] # 样本个数
    Y_hat = sigmoid(np.dot(w.T,X)+b)
    cost = -(np.sum(np.dot(Y,np.log(Y_hat).T)+np.dot((1-Y),np.log(1-Y_hat).T)))/m #成本函数

    dw = (np.dot(X,(Y_hat-Y).T))/m
    db = (np.sum(Y_hat-Y))/m

    cost = np.squeeze(cost) # 压缩维度
    grads = {"dw": dw,
             "db": db} # 梯度

    return grads, cost

# 梯度下降找出最优解
# num_iterations-梯度下降次数 learning_rate-学习率，即参数ɑ


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):#num_iterations-梯度下降次数 learning_rate-学习率，即参数ɑ
    costs = [] # 记录成本值

    for i in range(num_iterations): # 循环进行梯度下降
        grads, cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db

        # 每100次记录一次成本值
        if i % 100 == 0:
            costs.append(cost)

        # 打印成本值
        if print_cost and i % 100 == 0:
            print ("循环%i次后的成本值: %f" %(i, cost))

    # 最终参数值
    params = {"w": w,
              "b": b}

    # 最终梯度值
    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

# 预测出结果


def predict(w, b, X):
    m = X.shape[1] # 样本个数
    Y_prediction = np.zeros((1,m)) # 初始化预测输出
    w = w.reshape(X.shape[0], 1) # 转置参数向量w-->T

    Y_hat = sigmoid(np.dot(w.T,X)+b) # 最终得到的参数代入方程

    for i in range(Y_hat.shape[1]):
        if Y_hat[:,i]>0.5:
            Y_prediction[:,i] = 1
        else:
            Y_prediction[:,i] = 0

    return Y_prediction

# 建立整个预测模型
# num_iterations-梯度下降次数 learning_rate-学习率，即参数ɑ


def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):

    w, b = initialize_with_zeros(X_train.shape[0]) # 初始化参数w，b

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost) #梯度下降找到最优参数

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_train = predict(w, b, X_train) # 训练集的预测结果
    Y_prediction_test = predict(w, b, X_test) # 测试集的预测结果

    train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100 # 训练集识别准确度
    test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100 # 测试集识别准确度

    print("训练集识别准确度: {} %".format(train_accuracy))
    print("测试集识别准确度: {} %".format(test_accuracy))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}

    return d

# 初始化数据


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_x_orig.shape[0] # 训练集中样本个数
m_test = test_set_x_orig.shape[0] # 测试集总样本个数

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T # 原始训练集的设为（12288*209）
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T # 原始测试集设为（12288*50）

train_set_x = train_set_x_flatten/255. # 将训练集矩阵标准化
test_set_x = test_set_x_flatten/255. # 将测试集矩阵标准化

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

# 画出学习曲线
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

# 学习率不同时的学习曲线
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("学习率: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()