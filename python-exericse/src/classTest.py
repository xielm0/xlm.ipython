# -*- coding:utf-8 -*-

class Person(object):  # 定义一个父类
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.weight = 'weight'

    def talk(self):  # 父类中的方法
        print("person is talking....")


class Chinese(Person):  # 定义一个子类， 继承Person类
    def __init__(self, name, age, language):  # 先继承，在重构
        # 继承父类的构造方法,从而继承父类的属性。
        Person.__init__(self, name, age)
        # 也可以使用supper方法
        # super(Chinese,self).__init__(name,age)
        self.language = language    # 定义类的本身属性

    def walk(self):  # 在子类中定义其自身的方法
        print(self.name,self.language )

    #将类当函数使用
    def __call__(self, x):
        print(x)


# c = Chinese('bigberg', 22)
c = Chinese('bigberg', 22, 'Chinese')
c.talk()  # 调用继承的Person类的方法
# c.walk()  # 调用本身的方法
c("I am a function")