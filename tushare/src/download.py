# -*- coding:utf-8 -*-
import tushare as ts
import datetime
import os
import conf as cf
import func


now = datetime.datetime.now()
DAY = now.strftime('%Y-%m-%d')

basic_path=cf.basic_path
his_path=cf.his_path
today_path= cf.today_path
tick_path= cf.tick_path

if not os.path.exists(basic_path):
    os.mkdir(basic_path)
if not os.path.exists(his_path):
    os.mkdir(his_path)
if not os.path.exists(today_path):
    os.mkdir(today_path)
if not os.path.exists(tick_path):
    os.mkdir(tick_path)


def download_hist_data():
    """
    调用get_hist_data下载数据，但下载数度慢
    :return:
    """
    df_stock = ts.get_stock_basics()
    #
    DAY_01=func.get_trade_date(DAY,-1)
    filename= DAY_01+".csv"
    df_stock.to_csv(os.path.join(basic_path, filename))
    # 历史数据
    print("download stock hist ....")
    code_list= df_stock.index
    for code in code_list:
        # df=ts.get_k_data(code)  # return :date ,open close high low vol code
        # df=ts.get_h_data(code)
        try:
            df=ts.get_hist_data(code)
            df = df.sort_index(ascending=True)
            filename=code+".csv"
            df.to_csv(os.path.join(his_path, filename))
        except :
            print("code="+code+" is not exists!")
            pass

    # 当日行情
    # df_today=ts.get_today_all()

    # 分笔数据





def main():
    download_hist_data()


if __name__ == '__main__':
    main()