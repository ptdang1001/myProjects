# -*- coding: utf-8 -*

import matplotlib.pyplot as plt


def draw(result, epochs):
    #我这里迭代了epochs次，所以x的取值范围为(0，epochs)，然后再将每次相对应的准确率以及损失率附在x上
    x1 = range(0, epochs)
    y1 = result
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('train loss vs. epochs')
    plt.xlabel('train loss vs. epoches')
    plt.ylabel('train loss')
    plt.show()


def test():
    print("hello!")