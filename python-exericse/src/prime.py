# -*- coding: utf-8 -*-

from math import sqrt
import hashlib

def is_prime(n):
    if n == 1:
        return False
    for i in range(2, int(sqrt(n))+1):
        if n % i == 0:
            return False
    return True

# 找素数
def find_prime(n):
    for x in range(2,n):
        if is_prime(x):
            print(x)

# find_prime(1000)
# print(is_prime(31)) #2^6-1




