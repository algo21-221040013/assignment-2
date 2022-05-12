# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 08:44:47 2022

@author: DZQQQQ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from WindPy import w
# %%

# %%
def data_import(data_path): #数据导入
    data = pd.read_table(data_path, sep= '\t')
    data.columns = ['stock','date','holder','quantity','p']
    data.date = pd.to_datetime(data.date.astype(str))
    data = data.dropna(subset = ['holder']) #去掉非券商和银行的机构
    data = data[~data.stock.str.contains('HK')] #去掉残余的港股
    data.loc[:,'com'] = data.holder.apply(lambda x: x[0])
    return data.reset_index(drop = True)

hold_list = os.listdir(r'C:\Users\DZQQQQ\Desktop\北向\hold')    #导入收益率数据值
path_row = r'C:\Users\DZQQQQ\Desktop\北向\hold'
hold_28_20 = pd.DataFrame()
for i in hold_list:
    hold_28_20 = pd.concat([hold_28_20,data_import(os.path.join(path_row, i))])   #18.7-20.12完整持仓数据
    
# hold_28_20.to_csv('hold_18_20.txt')
#陆港通中出现的所有股票
target_stock = list(set(hold_28_20.stock))
# %% 持仓比例最大
def hold_prop(data, N1, n, company = ['B','C']):  #每个月底，按照持仓比例，N1日内类型company的机构持仓比例最大的n只股票
    data = data[data.com.isin(company)]   #选取银行或券商机构
    data = data.groupby(['date','stock']).sum()  #先按照日期表达出每天每只股票的机构持仓比例和数量
    data = data.unstack(level = 1).stack(level = 0)
    data.index.names = ['date','att']
    data_p = data.query('att == "p"')  #取出持仓比例dataframe
    data_p.fillna(method = 'ffill',inplace = True)
    data_p = data_p.rolling(N1).mean()  #计算个股N1日的持仓比例的平均值
    
    data_p = data_p.groupby(data_p.index.levels[0].month).apply(lambda x: x.iloc[-1])\
        .apply(lambda x: x.nlargest(n).index,axis = 1)  #输出每个月月底，过去N1天内平均持仓比例最高的n只股票名称
    
    return data_p
# %%增持最多组合
def speed_prop(data, N1, n, company = ['B','C']):  #每个月底，按照持仓比例，N1日内类型company的机构增持最多的n只股票
    data = data[data.com.isin(company)]   #选取银行或券商机构
    data = data.groupby(['date','stock']).sum()  #先按照日期表达出每天每只股票的机构持仓比例和数量
    data = data.unstack(level = 1).stack(level = 0)
    data.index.names = ['date','att']
    data_p = data.query('att == "quantity"')  #取出持仓数量dataframe
    data_p.fillna(method = 'ffill',inplace = True)
    data_p = (data_p - data_p.shift(N1)) / data_p.shift(N1)
    data_p = data_p.groupby(data_p.index.levels[0].month).apply(lambda x: x.iloc[-1])\
        .apply(lambda x: x.nlargest(n).index,axis = 1)  #输出每个月月底，过去N1天内平均持仓比例最高的n只股票名称
    
    return data_p
# %%
#计算全样本组合收益率
r_all = pd.read_csv(r'C:\Users\DZQQQQ\Desktop\北向\return_all.csv',index_col = 0,header = 0,usecols=target_stock
                    ).replace('--',np.nan).astype(float)
size_all = pd.read_csv(r'C:\Users\DZQQQQ\Desktop\北向\size_all.csv',index_col = 0,header = 0
                       ).replace('--',np.nan).astype(float)
r_all = r_all.loc[:,target_stock]
size_all = size_all.loc[:,target_stock]
r_all.index = pd.DatetimeIndex(r_all.index)
size_all.index = pd.DatetimeIndex(size_all.index)
profit_all = (r_all*(size_all.apply(lambda x: x / np.nansum(x),axis =1 ))).apply(lambda x:x.sum(),axis = 1) / 100 +1
profit_all = profit_all.cumprod()
# profit_all.index = pd.DatetimeIndex(profit_all.index)
# %%机构持仓比例超额收益
# res =
# hold_prop(data_import(r'C:\Users\DZQQQQ\Desktop\北向\18.txt'),15,5,['C','B'])
# res1 =  pass
# speed_prop(data_import(r'C:\Users\DZQQQQ\Desktop\北向\18.txt'),15,5,['C'])

def profit(year,sample):
    profit = pd.DataFrame()
    for i in sample.index:    #2018年数据从8月开始
        profit1 = (r_all.loc['%d-%d' %(year,i+1),sample.loc[i]] * (size_all.loc['%d-%d' %(year,i),sample.loc[i]]\
            .apply(lambda x: x / np.nansum(x),axis =1 ))).apply(lambda x:x.sum(),axis = 1)
        profit1 = (profit1/100 + 1).cumprod()
        profit = pd.concat([profit,profit1],axis = 0)
    return profit

b_top_prop = profit(2018,res1.iloc[:-1])
plt.plot(b_top_prop.sub(profit_all,axis = 0))
print(b_top_prop.sub(profit_all,axis = 0).mean())

# %%

        


