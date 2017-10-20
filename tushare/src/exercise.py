# -*- coding:utf-8 -*-
import tushare as ts


def main():
    df=ts.get_k_data('000333',start='2017-08-01')
    df_his=ts.get_h_data('000333',start='2017-08-01')
    df[:10].open

    df_info=ts.get_stock_basics()
    df[:10]

    df_cpi=ts.get_cpi()

    df_news = ts.get_latest_news()

    df_today=ts.get_today_all()


if __name__ == '__main__':
    main()