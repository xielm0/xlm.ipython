# -*- coding: utf-8 -*-

class Trie:
    """
    python没有指针，通过多层嵌套实现
    """
    def __init__(self):
        self.root = {}
        # word_end主要用于find时，每个word必须要有一个结束符。
        # 假设2个单词"abc","ab",如果没有结束符，只要add("abc"),find("ab")也能return True,这是不对的。
        self.word_end = '-1'  #不属于字符就好。

    def add(self, word):
        curNode = self.root
        for c in word:
            if c not in curNode:
                curNode[c] = {}  #创建子树。子树也用字典表示，初始是空字典。
                curNode = curNode[c]  #进入子树
            else:
                curNode = curNode[c]  #直接进入子树
        # 插入结束符
        curNode[self.word_end]= self.word_end

    def find(self, word):
        curNode = self.root
        for c in word:
            if c not in curNode:
                return False
            else:
                curNode = curNode[c] #进入进入子树
        # 遍历完后，确认是否找到结束符
        if self.word_end not in curNode:
            return False
        else:
            return True


    def display(self):
        curNode = self.root # 从root开始
        for (k, v) in curNode.items():
            print(k,v)


def test():
    trie = Trie()
    trie.add('abc')
    trie.add('ab')
    trie.add('abcd')
    trie.add('hello')
    print(trie.find("abc"))
    trie.display()


if __name__ == '__main__':
    test()