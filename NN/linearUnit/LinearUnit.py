
#定义激活函数
from NN.perceptor.Perceptor import Perceptron

#在线性感知器中,我们通过得到的权重如果为W，得到的偏置量如果为 B那么得到的最后的结果
# Y = W*X+
f = lambda x :x

#在and感知器的基础之上我们换用 y = x作为激活函数来实现线性单元
class LinearUnit(Perceptron):
    def __init__(self,input_num):
        Perceptron.__init__(self,input_num,f)