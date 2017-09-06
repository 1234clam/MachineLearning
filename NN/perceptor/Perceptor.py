from functools import reduce
class Perceptron(object):
    def __init__(self,input_num,activator):
        '''
        :param input_num:
        :param activator:  激活函数的类型，激活函数的类型为double->double
        :return:
        '''
        self.activator = activator

        #输入的长度是二维的数据，初始化权重矩阵也是一个二维的数据结果
        self.weights = [0.0 for _ in range(input_num)]

        #初始化偏置量的值
        self.bias = 0.0

    #重定义父类的__str__函数
    def __str__(self):
        #打印学习到的权重和偏置量
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights,self.bias)

    def train(self,input_vecs,labels,iteration,rate):
        for i in range(iteration):
            self._one_iteration(input_vecs,labels,rate)

    def predict(self,input_vec):
        '''
        :param input_vec: 对于每一个输入向量都是二维的，对input_vec中存储的值进行计算
        :return: 不存在返回值
        '''
        #在这通过map的值求出input_vec中的每一个数据和self.weight中的每一个结果相乘
        #通过reduce将通过map得到的结果相加，在此基础之上加上self.bias
        #返回预测的结果
        return self.activator(
            reduce(lambda a, b: a + b,
                   map(lambda x, w: x * w,
                       input_vec, self.weights)
                   , 0.0) + self.bias)

    def _one_iteration(self,input_vecs,labels,rate):
        '''
        :param input_vecs: 每次迭代输入的input_vecs
        :param labels: 每次迭代对应的标签
        :param rate: rate表示学习的速率
        :return:
        '''
        samples = zip(input_vecs,labels)
        #对数据集里面的每一个结果
        for (input_vec,label) in samples:
            #预测出输出的结果
            output = self.predict(input_vec)
            #更新矩阵的权重
            self._update_weights(input_vec,output,label,rate)

    def _update_weights(self,input_vec,output,label,rate):
        '''
        :param input_vec: 预测时输入的值
        :param output: 对应的输出值
        :param label: 对应的标签值
        :param rate: 学习速率
        :return:
        '''
        #计算出来的误差
        delta = label - output
        #新的权重计算公式中：新权重 = 原来的权重+误差*学习速率*输入
        # Wn = Wn-1+ delta*rate*x
        self.weights = list(map(
            lambda x, w: w + rate * delta * x,
            input_vec, self.weights))
        #更新bias 新的偏置量 = 学习效率*误差值 + 原有的偏置量
        #Bn = Bn-1+ rate *
        self.bias += rate*delta