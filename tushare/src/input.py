# -*- coding:utf-8 -*-
import tushare as ts
import datetime
import os
import numpy as np
import pandas as pd
import func
import conf as cf

now = datetime.datetime.now()
DAY = now.strftime('%Y-%m-%d')
DAY = func.get_trade_date(DAY, 0)
DAY_01 = func.get_trade_date(DAY, -1)
DAY_05 = func.get_trade_date(DAY_01, -5)
DAY_20 = func.get_trade_date(DAY_01, -20)

def get_code_list():
    """
    根据过滤条件选出一部分code
    :return:
    """
    df_stock = ts.get_stock_basics()
    code_list= df_stock.index
    return code_list

CODE_LIST=get_code_list()


def get_basic_feature(df_stock, code, day=DAY):
    """
    对ts.get_stock_basics(day) 做一些预处理处理
    :param df_stock:
    :return:
    """
    df = df_stock.loc[code,["industry","area","timeToMarket","pe","pb","outstanding","reservedPerShare","bvps"]]
    industry,area,timeToMarket,pe,pb,outstanding,reservedPerShare,bvps=df
    day_60 = func.get_trade_date(day, -60)
    day_60 = datetime.datetime.strptime(day_60, '%Y-%m-%d').strftime('%Y%m%d')
    day_60 = int(day_60)
    new_flag = 0 if df["timeToMarket"]<day_60 else 1

    # 对pe进行分段
    #

    return [outstanding,industry,area,timeToMarket,pe,pb,reservedPerShare,bvps,new_flag]


def get_day_pic(code,day, n=20):
    """
    根据close,及前n个基本的交易数据，生成按顺序的向量，可以看作图像的像素。
    :param code:
    :param day:
    :param n:
    :return:
    """
    start_date= func.get_trade_date(day,-n)
    df = ts.get_k_data(code,start=start_date, end=day)
    if len(df)<20:
        return None
    df1=df[["open", "close", "high", "low"]]
    df2=df["volume"]
    # close的价格作为标准
    c = df.iloc[-1,1]
    v = df.iloc[-1,4]
    df1 = df1/c - 1
    df2 = df2/v - 1

    # 转成向量
    s1= func.df_to_vector(df1,n)
    s2= func.series_to_vector(df2,n)
    return  s1 + s2


def get_day_trade(code,day ):
    """
    获取衍生特征。
    date 交易日期 (index)  open 开盘价  close 收盘价 high  最高价 low 最低价 df 成交量 (amount 成交额 turnover 换手率)
    :return: +　price_change　p_change　ma5  ma20 ma60 v_ma5  v_ma20  v_ma60
    :param outstanding:  用来计算换手率
    """
    df = ts.get_k_data(code,"2017-01-01", end=day)
    #计算每日的涨跌幅
    df["p_change"]= df["close"]/df["close"].shift(1) - 1

    # ma
    df["ma5"]=df["close"].rolling(window=5).mean()
    df["ma20"]=df["close"].rolling(window=20).mean()
    df["ma60"]=df["close"].rolling(window=60).mean()
    df["v_ma5"]=df["volume"].rolling(window=5).mean()
    df["v_ma20"]=df["volume"].rolling(window=20).mean()
    df["v_ma60"]=df["volume"].rolling(window=60).mean()
    #
    df["close_smoth"]=df["close"].rolling(window=3).mean()
    df["trend_diff"] = df["close_smoth"]/df["close_smoth"].shift(1) - 1
    df["trend_second_diff"] = df["trend_diff"]- df["trend_diff"].shift(1)

    return df


def get_day_more(code,day, outstanding):

    df= get_day_trade(code,day)
    #
    df["turnover"]= df["volume"]/outstanding
    #
    df["open_ma5"] = df["open"]/df["ma5"] - 1
    df["open_ma20"]= df["open"]/df["ma20"] - 1
    df["open_ma60"] = df["open"]/df["ma60"] - 1
    df["close_ma5"]  = df["close"]/df["ma5"] - 1
    df["close_ma20"] = df["close"]/df["ma20"] - 1
    df["close_ma60"] = df["close"]/df["ma60"] - 1

    return df


def get_feature(day):
    """
    将day对应的特征保存成文件
    :param day:
    :return:
    """
    # 获取股票列表
    code_list=CODE_LIST
    # 基本面数据
    df_stock = ts.get_stock_basics(day)
    #


    # 获取日线数据
    X=[]
    for code in code_list:
        basic = get_basic_feature(df_stock,code,day)
        df_k=ts.get_k_data(code,start=day_60,end=day)
        #
        trade,trade_more,trade_flag =get_day_more(code,df_k)
        pic=get_day_pic(code,df_k,60)
        turnover= trade_more[0]/basic[0]

        label = get_label(code,day,3,1)
        x= [code] + trade + trade_more + trade_flag + pic + turnover

        X.append(x)

    return X

def write_feature(feature ,day):
    filename =cf.train_path + day +".csv"
    with open(filename,"w") as f:
        for line in feature:
            f.writelines(func.mkstring(line,",")+"\n")



LABEL_LIST=[-0.08, -0.03, -0.01, 0.01, 0.03, 0.08]
LABEL_N=len(LABEL_LIST)+1
def get_label(code,day,n,win=1):
    """
    买入后，会有最大收益，和最大风险。
    并且划分为10段
    :param win:  win=1，代表计算最大收益，win=0 ,计算最大损失。
    :return:
    """
    end_date=func.get_trade_date(day, n)
    df=ts.get_k_data(code,start=day, end=end_date)

    ind=list(range(len(df)))
    df.index=ind
    cur=df.loc[0,"close"]
    if win ==1 :
        y=np.max(df.loc[1:,["open","close"]].values)
    else :
        y=np.min(df.loc[1:,["open","close"]].values)

    label = round(y/cur,4)-1
    print(label)
    return func.bucket(label,LABEL_LIST,LABEL_N)



def get_train_data(day):
    feature =get_feature(day)
    label=get_label()





def main():
    get_train_data(DAY_01)
    # df_stock = ts.get_stock_basics(DAY_01)
    # get_basic_feature(df_stock,"000333",DAY_01)
    # get_trade_data("603701",DAY_01)
    # print(get_label("000333",DAY_05,3,1))
    # print(get_label("000333",DAY_05,3,2))


if __name__ == '__main__':
    main()