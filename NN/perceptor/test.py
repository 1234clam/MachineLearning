from NN.perceptor.Perceptor import Perceptron


def f(x):
    #定义激活函数
    return 1 if x > 0 else 0

def get_training_dataset():
    #构建训练数据集
    #输出向量列表
    input_vecs = [[1,1],[0,0],[1,0],[0,1]]

    #期望的输出列表，注意要与输入一一对应
    labels = [1,0,0,0]
    return input_vecs,labels

def train_and_perceptron():
    #创建感知器，输入参数的个数为2(因为and是二元运算)，激活函数未f
    p = Perceptron(2,f);
    input_vecs,labels = get_training_dataset()
    p.train(input_vecs,labels,10,0.2)
    return p

if __name__ == '__main__':
    # 训练and感知器
    and_perceptron = train_and_perceptron()
    # 打印获取的权重的结果
    print(and_perceptron)
    #测试
    print('1 and 1 = %d'%and_perceptron.predict([1, 1]))
    print('0 and 0 = %d'%and_perceptron.predict([0, 0]))
    print('1 and 0 = %d'%and_perceptron.predict([1, 0]))
    print('0 and 1 = %d'%and_perceptron.predict([0, 1]))

