# -*- coding: utf-8 -*-

#等差
def d(n):
    if n==0:
        return n
    else:
        return d(n-1)+2

def dispaly_d():
    print(d(100))
    for i in range(10):
        print(d(i))


#等同于
def d2(n):
    c=0
    for i in range(n):
        c += 2
    return c

#Fibonacci数
def f(n):
    if n<=1:
        return n
    else:
        return f(n-1)+f(n-2)

def display_f():
    for i in range(10):
        print(f(i))


if __name__ == '__main__':
    dispaly_d()
