# coding: utf-8

if __name__ == '__main__':
    import jqsdk
    params = {
        'token':'c767fdeaa9ab2c6446fc0ce3a7153287',
        'algorithmId':6,
        'baseCapital':1000000,
        'frequency':'day',
        'startTime':'2019-04-26',
        'endTime':'2019-04-27',
        'name':"Test1",
    }
    jqsdk.run(params)

##### 下面是策略代码编辑部分 #####

import jqdata

def initialize(context):
    print("init........")
    run_daily(bfopen, time='before_open')

def bfopen(context):
    print("open........")
    l = get_billboard_list(stock_list=None, end_date=context.previous_date, count=1)
    print(l)
    l.to_csv("blb.csv", encoding="utf_8_sig")
