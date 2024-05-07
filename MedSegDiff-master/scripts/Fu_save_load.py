import torch
import torchvision.models as models
import torch.optim as optim
#保存模型权重
# model = models.vgg16(pretrained=True)
# torch.save(model.state_dict(),'model_weight.pth')
#
# #load model weights
# model = models.vgg16(pretrained=True)
# model.load_state_dict(torch.load('model_weight.pth'))
# model.eval() #模型 inference阶段
#
# #save the general checkpoint   ：collect 阿拉蕾relevant information and build your dictionary。
#
# EPOCH = 5
# PATH = 'model.pt'
# LOSS = 0.4
# optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
# torch.save({
#     'epoch': EPOCH,
#     'model_state_dict':model.state_dict(),
#     'optimizer':optimizer.state_dict()
#     'loss':LOSS,
# },PATH)
#
# #load the general checkpoint
# model = ()
# optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
# checkpoint = torch.load(PATH)
# optimizer.load_state_dict(checkpoint['optimizer'])
# model.load_state_dict(checkpoint['model_state_dict'])

#所有继承自MODULE的类都会有state_dict()函数和load_state_dict()函数.包含当前当前module的参数和buffer量.


#to函数:将模型的权重和buffer放到divice变量里面.、


# class Student:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age
#
#
#     def get_name(self):
#         return name
#
#
# ryan = Student("ryan", 18)
#
# print(ryan.get_name())
import numpy as np
#numpy.ndarray的数组中numpy.ndarray的list和int居然可以直接进行比较并把结果作为索引
b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
a = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
print(type(a))
for i in range(3):

    # numpy.ndarray的list和int类型的在数组中
    # 居然可以直接比较并当做索引使用
    print(b[a == i])

for i in range(3):
    print(a)
    print('\n')
    print(a == i)
    print(type(a == i))




class NumpyStudy:
    def lnFunction(self):
        const = np.e
        result = np.log(const)
        print("np.e=",np.e)
        print("np.log10(100)=",np.log10(100))
        print("函数ln(e)的值为：")
        print(result)


if __name__ == "__main__":
    main = NumpyStudy()
    main.lnFunction()

    import torch

    a = torch.rand(2, 3) * 10
    print(a)
    a = a.clamp(5, 8)
    print(a)