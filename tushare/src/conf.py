# -*- coding:utf-8 -*-
import tushare as ts
import datetime
import os
import func

now = datetime.datetime.now()
DAY = now.strftime('%Y-%m-%d')
DAY = func.get_trade_date(DAY,0)
print(DAY)

basic_path="../data/stock_basics/"
his_path="../data/his/"
today_path= "../data/today/"+DAY
tick_path="../data/tick/"
train_path="../data/train/"
apply_path="../data/apply/"

if not os.path.exists(basic_path):
    os.mkdir(basic_path)
if not os.path.exists(his_path):
    os.mkdir(his_path)
if not os.path.exists(today_path):
    os.mkdir(today_path)
if not os.path.exists(tick_path):
    os.mkdir(tick_path)
if not os.path.exists(train_path):
    os.mkdir(tick_path)


