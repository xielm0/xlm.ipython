# -*- coding: utf-8 -*-

from math import sqrt
import hashlib

# 采用递归
def hashCode(s):
    h = 0
    for i, c in enumerate(s):
        h = 31* h + ord(c)  # ord
    return h


def hash64(s):
    m = hashlib.sha256()
    m.update(s.encode('utf-8'))  #hash
    v = m.hexdigest()  # 返回的是16进制的字符串
    # v = int(v,16) #转成10进制
    return v


print(hashCode("a") )
print(hash("a:b") % 10000)  #不同次运行的哈希值不同
print(hashCode("a:b") % 10000)
print(hash64("a:b") )



