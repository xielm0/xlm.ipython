# -*- coding: utf-8 -*-

#ctr平滑的代码

import scipy.special as special
import pandas as pd
import numpy as np
import random

class HyperParam(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def update(self, imp_click, iter_num, epsilon):
        for i in range(iter_num):
            # 增加采样，加快速度
            sample_imp_click=imp_click.sample(100000).values
            new_alpha, new_beta = self.__fixed_point_iteration(sample_imp_click, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta
            print(new_alpha,new_beta)


    def __fixed_point_iteration(self, imp_click, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0
        n=len(imp_click)
        for i in range(n):
            imp,click=imp_click[i]
            numerator_alpha += (special.digamma(click+alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imp-click+beta) - special.digamma(beta))
            denominator += (special.digamma(imp+alpha+beta) - special.digamma(alpha+beta))

        return alpha*(numerator_alpha/denominator), beta*(numerator_beta/denominator)

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        sample = np.random.beta(alpha, beta, num)
        imp_click = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            click = imp * click_ratio
            imp_click.append((imp,click))
        return pd.DataFrame(imp_click)


def test():
    hyper = HyperParam(1, 1)
    imp_click = hyper.sample_from_beta(1, 50, int(1e6), 10000)

    hyper.update(imp_click, 100, 1e-8)
    print(hyper.alpha, hyper.beta)


def download():
    sql ="""
use tmp;
create table tmp_szad_imp_click
ROW FORMAT DELIMITED  FIELDS TERMINATED BY '\t'
as
select  expose_nums,click_nums
from app.app_szad_m_dyrec_sku_ctr_train
where  adspec_id = '255' and expose_nums>0
    """
    cmd ="hdfs dfs -get hdfs://ns3/user/jd_ad/tmp.db/tmp_szad_imp_click/000000_0"
    cmd ="mv 000000_0 imp_click.txt"
    pass


def ctr_alpha_beta():
    df=pd.read_csv('data/imp_click.txt' ,sep="\t")
    imp_click=df
    print(len(imp_click))

    bs = HyperParam(1, 50)
    print("start train alpha & beta")
    bs.update(imp_click, 1000, 1e-8)
    print(bs.alpha, bs.beta)

def compute_ctr():
    df = pd.read_csv('data/imp_click.txt', sep="\t")
    imp = df.iloc[:,0]
    click = df.iloc[:,1]
    ctr=sum(click)/sum(imp)
    print(ctr)


if __name__ == '__main__':
    test()