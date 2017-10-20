import pybacktest
import pandas as pd


def main():
    # 从yahoo下载数据
    ohlc = pybacktest.load_from_yahoo('SPY')
    ohlc.tail()

    # MA快线上穿慢线时，买进做多
    # MA快线下穿慢线时，卖出做空
    short_ma = 50
    long_ma = 200

    ms = pd.rolling_mean(ohlc.C, short_ma)
    ml = pd.rolling_mean(ohlc.C, long_ma)

    buy  = (ms > ml) & (ms.shift() < ml.shift())  # ma cross up
    sell = (ms < ml) & (ms.shift() > ml.shift())  # ma cross down

    print('>  Short MA\n%s\n' % ms.tail() )
    print( '>  Long MA\n%s\n' % ml.tail() )
    print( '>  Buy/Cover signals\n%s\n' % buy.tail() )
    print( '>  Short/Sell signals\n%s\n' % sell.tail())


    bt = pybacktest.Backtest(locals(), 'ma_cross')


    print( filter(lambda x: not x.startswith('_'), dir(bt))     )
    print( '\n>  bt.signals\n%s' % bt.signals.tail()             )
    print( '\n>  bt.trades\n%s' % bt.trades.tail()               )
    print( '\n>  bt.positions\n%s' % bt.positions.tail()         )
    print( '\n>  bt.equity\n%s' % bt.equity.tail()               )
    print( '\n>  bt.trade_price\n%s' % bt.trade_price.tail()     )

    bt.summary()

    # 净资产曲线
    # figsize(10, 5)
    bt.plot_equity()

    #
    bt.plot_trades()
    pd.rolling_mean(ohlc.C, short_ma).plot(c='green')
    pd.rolling_mean(ohlc.C, long_ma).plot(c='blue')
    legend(loc='upper left')


