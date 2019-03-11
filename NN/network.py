import numpy as np 
import pickle
import gzip
import random




class Network(object):

    def __init__(self, sizes):
        # 网络层数
        self.num_layers = len(sizes)
        self.sizes = sizes
        # biases从第二层起, y行1列
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        # n与n+1层权重总数:n*(n+1)
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        # zip逐层读w,b
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b) # dot乘
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        # 随机梯度法
        if test_data: n_test = len(test_data)
        n = len(training_data)
        # 迭代次数
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta) # eta学习速率
            if test_data:
                print("Epoch {0}:{1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        # 初始化w,b  用于梯度复制
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            # 梯度计算
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        # 1.输入x：为输入层设置对应的激活值a(L)
        # 2.前向传播：计算z(L)=w(L)a(L-1)+b(L) and a(L)=sigmoid(z(L))
        # 3.输出层误差c: 误差向量c(L)=round(aC) dot sigmoid_prime(z(L))
        # 4.反向误差传播：c(L)=(w(L+1)Tc(L+1)) dot sigmoid_prime(z(L))
        # 5.输出：代价函数的梯度由 w的偏导=a(k)(l-1)c(j)(l) and b的偏导=c(j)(l)得出
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        activation = x
        activations = [x]
        zs = []
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1],y)*sigmoid_prime(zs[-1])
        nabla_b[-1] = delta 
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(),delta) * sp 
            nabla_b[-l] = delta 
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activatives, y):
        # a对C的偏导，推导结果式a(L)-y
        return (output_activatives-y)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    # sigmoid的导数
    return sigmoid(z)*(1-sigmoid(z))


def load_data():
    # Load the dataset
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f,encoding='bytes')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    # y增加到10维
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

if __name__ == '__main__':
    training_data, validation_data, test_data = load_data_wrapper()
    net = Network([784,30,10])
    net.SGD(training_data, epochs=1, mini_batch_size=10, eta=3.0, test_data=test_data)
