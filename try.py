# -*- coding: utf-8 -*-
"""
Created on Sat May  7 14:12:25 2022

@author: DZQQQQ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# import os
# %%
# price_target = pd.read_csv('target_price.csv',header = 0,index_col =0).fillna(method = 'ffill')         #读取目标股票收盘价
# price_target.index = pd.DatetimeIndex(price_target.index)                       #转换索引为日期

# price_target = price_target.resample('B').fillna(method = 'ffill')

# price_target_ave = pd.read_csv('target_price_ave.csv',header = 0,index_col =0).fillna(method = 'ffill')
# price_target_ave.index = pd.DatetimeIndex(price_target_ave.index)                       #转换索引为日期
# price_target_ave = price_target_ave.resample('B').fillna(method = 'ffill')


#

# =============================================================================
# new
# def data_import(data_path): #数据导入
#     data = pd.read_table(data_path, sep= '\t')
#     data.columns = ['stock','date','holder','quantity','p']
#     data.date = pd.to_datetime(data.date.astype(str))
#     data = data.dropna(subset = ['holder']) #去掉非券商和银行的机构
#     data = data[~data.stock.str.contains('HK')] #去掉残余的港股
#     data.loc[:,'com'] = data.holder.apply(lambda x: x[0])
#     return data.reset_index(drop = True)

# a_21 = data_import(r'hold\21.txt')
# hold_18_20 = pd.DataFrame()
# for i in [r'hold\18.txt',r'hold\19.txt',r'hold\20.txt']:
#     hold_18_20 = pd.concat([hold_18_20,data_import(i)],axis = 0)

# =============================================================================


# hold_18_20.date = pd.DatetimeIndex(hold_18_20.date)
# hold_18_20 = hold_18_20.sort_values(by = 'date').set_index('date')
# hold_18_20 = hold_18_20[hold_18_20.index.isin(price_target_ave.index)]


hold_18_20 = pd.read_hdf('hold_data.hdf5', 'hold_18_20')
# target_date = sorted(list(set(hold_18_20.index)))

# %%

# price_target = price_target[price_target.index.isin(target_date)]
# price_target_ave = price_target_ave[price_target_ave.index.isin(target_date)]
# price_target=pd.read_hdf('return_target.hdf5','close_price').sort_index(axis = 1)
# price_target_ave=pd.read_hdf('return_target.hdf5','ave_price').sort_index(axis = 1)

price_target = pd.read_csv('close_price.csv', header=0, index_col=0)
price_target.index = pd.DatetimeIndex(price_target.index)

price_target_ave = pd.read_csv('ave_price.csv', header=0, index_col=0)
price_target_ave.index = pd.DatetimeIndex(price_target_ave.index)

# 获取机构持仓日内标的股票的收盘价

# return_target = ((price_target - price_target.shift(1)) / price_target.shift(1))
# return_target_5 = ((price_target - price_target.shift(5)) / price_target.shift(5))
# return_target_m = ((price_target - price_target.shift(20)) / price_target.shift(20))

return_target = pd.read_hdf('return_target.hdf5', 'daily')
return_target_5 = pd.read_hdf('return_target.hdf5', 'weekly')
return_target_m = pd.read_hdf('return_target.hdf5', 'monthly')

# 计算持仓比例因子
# 总股本数导入
# total_num = pd.read_csv('total_num.csv',header = 0,index_col = 0).sort_index(axis = 1)
# total_num.index = pd.DatetimeIndex(total_num.index)
# total_num = pd.merge(pd.DataFrame(target_date,columns = ['date']),total_num,left_on = 'date',right_index = True,how = 'left').set_index('date')
# 流通市值导入
total_value = pd.read_csv('total_value.csv', header=0, index_col=0).sort_index(axis=1)
total_value.index = pd.DatetimeIndex(total_value.index)


# total_value = total_value.resample('B').fillna(method = 'ffill')
# total_value = total_value[total_value.index.isin(target_date)]
# %% 算出不同机构持有的市值
# def dsum_company(company = ['B','C']):   #计算不同机构股票持仓总市值
#     date = hold_18_20.index
#     if len(company) == 1:
#         date = hold_18_20[hold_18_20.com == company].index
#         dsum = pd.DataFrame(hold_18_20[hold_18_20.com == company].groupby([date,'stock']).sum().loc[:,'quantity'])
#     else:
#         dsum = pd.DataFrame(hold_18_20.groupby([date,'stock']).sum().loc[:,'quantity'])
#     dsum.loc[:,'market_value'] = np.nan
#     for i in dsum.index:
#         dsum.loc[i,'market_value'] = dsum.loc[i,'quantity'] * price_target_ave.loc[i[0],i[1]]
#     return dsum

# %%  全体北向资金IC计算

# dsum_all = dsum_company().loc[:,'market_value'].unstack(level  = 1 )   #全体北向资金
# dsum_B = dsum_company('B').loc[:,'market_value'].unstack(level  = 1 )  #券商增持市值
# dsum_C = dsum_company('C').loc[:,'market_value'].unstack(level  = 1 )    #银行增持市值

# dsum_all=pd.read_hdf('chicangshizhi.hdf5','dsum_all')
# dsum_C=pd.read_hdf('chicangshizhi.hdf5','dsum_C')
# dsum_B=pd.read_hdf('chicangshizhi.hdf5','dsum_B')

# #%%
# #增持市值比例

# dsum_all_add = (dsum_all - dsum_all.shift(1)) / dsum_all.shift(1)
# dsum_all_add = dsum_all_add - dsum_all_add.shift(1)
# dsum_B_add = (dsum_B - dsum_B.shift(1)) / dsum_B.shift(1)
# dsum_C_add = (dsum_C - dsum_C.shift(1)) / dsum_C.shift(1)

# #增持市值

# dsum_all_add = (dsum_all - dsum_all.shift(1))
# dsum_B_add = (dsum_B - dsum_B.shift(1))
# dsum_C_add = (dsum_C - dsum_C.shift(1))

# %%ic计算
def corr_calculate(x):  # 计算不同机构持有市值的ic
    corr_frame = pd.DataFrame(index=x.index, columns=['corr'])
    for i in range(len(x) - 20):
        corr_frame.iloc[i, 0] = x.iloc[i, :].rank(ascending=False).corr(
            return_target_m.iloc[i + 20, :].rank(ascending=False))
    return corr_frame


# %%计算不同机构增持比例的IC结果

# ic_all = corr_calculate(dsum_all_add.replace(0,np.nan))
# ic_B = corr_calculate(dsum_B_add.replace(0,np.nan))
# ic_C = corr_calculate(dsum_C_add.replace(0,np.nan))
# result_ic = pd.DataFrame([[ic_all.mean(),ic_all.std()],[ic_C.mean(),ic_C.std()],[ic_B.mean(),ic_B.std()]],columns = ['mean','std'],index = ['all','C','B'])

# %%计算持仓市值最高的机构的IC

hold_price = pd.read_hdf('hold_data.hdf5', 'hold_price')  # 有每个股票和机构持仓市值的data
# holder_sum = hold_price.groupby(['date','holder']).sum() #按照机构持仓总量group

# holder_sum.to_hdf('hold_data.hdf5','holder_sum',format = 'fixed')
holder_sum = pd.read_hdf('hold_data.hdf5', 'holder_sum')


# =============================================================================
# 计算每天持仓市值最大的n家机构
def n_most_holder(n):
    return holder_sum.value.groupby(level=0, group_keys=False).apply(
        lambda x: x.sort_values(ascending=False)[:n])  # 前三分之一


# 计算持仓市值最小的n家机构
def n_least_holder(n):
    return holder_sum.value.groupby(level=0, group_keys=False).apply(lambda x: x.sort_values()[:n])


# =============================================================================

# %%计算持仓市值最高的机构IC
# h = hold_18_20.reset_index().set_index('date').loc[:,['stock','holder','quantity']].loc['2021']
# h = h.set_index([h.index,'holder'])
TOP_result = pd.DataFrame(index=np.arange(1, 21), columns=['mean', 'std'])
TOP_result_week = pd.DataFrame(index=np.arange(1, 21), columns=['mean', 'std'])
TOP_result_month1 = pd.DataFrame(index=np.arange(1, 20), columns=['mean', 'std', 'ir'])


def corr_calculate(x):  # 计算不同机构持有市值的ic
    corr_frame = pd.DataFrame(index=x.index, columns=['corr'])
    for i in range(len(x) - 20):
        corr_frame.iloc[i, 0] = x.iloc[i, :].rank(ascending=False).corr(
            return_target_m.iloc[i + 20, :].rank(ascending=False))
    return corr_frame


for a in range(1, 20):
    # h5 = n_most_holder(a)

    # holder_top_company = pd.merge(h,pd.DataFrame(h5),left_index = True,right_index = True)
    # holder_top_company = holder_top_company.iloc[:,:2]
    # holder_top_company.columns = ['stock','quantity']

    # top_company = holder_top_company.reset_index().groupby([holder_top_company.reset_index().date,'stock']).sum()
    # top_company = top_company.unstack(level = 1).droplevel(0,axis = 1)
    # top_company = pd.read_hdf('top_company.hdf5','h%d' %(a+1)).append(top_company)  #注意这里打错了，h2对应最高的一家,h21对应最高的20家
    # top_company.to_hdf('top_company.hdf5','h%d' %(a))
    top_company = pd.read_hdf('top_company.hdf5', 'h%d' % (a))
    top_company = top_company * price_target_ave / total_value
    # n = 1
    # top_company = (top_company.fillna(0) - top_company.shift(n).fillna(0)) / top_company.shift(n).fillna(0)

    ic_TOP = corr_calculate(top_company.replace(0, np.nan))
    # print(ic_TOP.mean()[0],ic_TOP.std()[0],ic_TOP.mean()[0] / ic_TOP.std()[0])
    TOP_result_month1.iloc[a - 1, 0] = ic_TOP.mean()[0]
    TOP_result_month1.iloc[a - 1, 1] = ic_TOP.std()[0]
    TOP_result_month1.iloc[a - 1, 2] = ic_TOP.mean()[0] / ic_TOP.std()[0]

# holder6_8=n_most_holder(7).append(n_most_holder(6)).drop_duplicates(keep=False)
# pd.DataFrame(holder6_8.index.get_level_values(1)).value_counts()

# %%史丹利持仓
n = 5  # 周频增加
hold6_8 = h[h.holder == 'C00093 ']
hold6_8 = hold6_8.groupby([hold6_8.index, 'stock'])['quantity'].sum().unstack(level=1)
hold_weight = hold6_8 * price_target_ave  # 市值权重
hold6_8 = hold6_8 * price_target_ave / total_value
print(corr_calculate(hold6_8).mean())
# hold6_8 = (hold6_8.fillna(0) - hold6_8.shift(n).fillna(0))
# ic_TOP = corr_calculate(top_company.replace(0,np.nan))
# print(ic_TOP.mean()[0],ic_TOP.std()[0],ic_TOP.mean()[0] / ic_TOP.std()[0])

# %%计算持仓市值最低的机构IC
BOTTOM_result = pd.DataFrame(index=np.arange(1, 21), columns=['mean', 'std'])
BOTTOM_result_week = pd.DataFrame(index=np.arange(1, 21), columns=['mean', 'std', 'ir'])
BOTTOM_result_month = pd.DataFrame(index=np.arange(1, 40), columns=['mean', 'std', 'ir'])
h = pd.read_hdf('hold_data.hdf5', 'hold_quantity')
h = h.set_index([h.index, 'holder'])
for a in range(1, 40):
    # h5 = n_least_holder(a)
    # # holder_bottom_company = pd.DataFrame()  #每天持仓市值最多的的机构的持仓信息

    # holder_bottom_company = pd.merge(h,pd.DataFrame(h5),left_index = True,right_index = True)
    # holder_bottom_company = holder_bottom_company.iloc[:,:2]
    # holder_bottom_company.columns = ['stock','quantity']

    # bottom_company = holder_bottom_company.reset_index().groupby([holder_bottom_company.reset_index().date,'stock']).sum()
    # bottom_company = bottom_company.unstack(level = 1).droplevel(0,axis = 1)

    # bottom_company.to_hdf('bottom_company.hdf5','h%d' %(a))

    bottom_company = pd.read_hdf('bottom_company.hdf5', 'h%d' % (
        a))  # .append(bottom_company)#.fillna(0) - pd.read_hdf('bottom_company.hdf5','h1').fillna(0)

    bottom_company = bottom_company * price_target_ave / total_value
    # bottom_company = (bottom_company.fillna(0) - bottom_company.shift(1).fillna(0)) / bottom_company.shift(1).fillna(0)
    # bottom_company = (bottom_company.fillna(0) - bottom_company.shift(1).fillna(0)) * price_target / \
    # bottom_company.shift(1) * price_target.shift(1)
    ic_bottom = corr_calculate(bottom_company.replace(0, np.nan))
    BOTTOM_result_month.iloc[a - 1, 0] = ic_bottom.mean()[0]
    BOTTOM_result_month.iloc[a - 1, 1] = ic_bottom.std()[0]
    BOTTOM_result_month.iloc[a - 1, 2] = ic_bottom.mean()[0] / ic_bottom.std()[0]

# a_bottom = pd.read_hdf('bottom_company.hdf5','h2').fillna(0) - pd.read_hdf('bottom_company.hdf5','h1').fillna(0)
# a_bottom = a_bottom * price_target_ave
# a_bottom = a_bottom.replace(0,np.nan)
# a_ic_bottom = corr_calculate(a_bottom)
# print(a_ic_bottom.mean(),a_ic_bottom.std())
# %%
#   top - bottom对比
top_bottom_result = pd.DataFrame(columns=['mean', 'std', 'ratio'])
target_stock = pd.DataFrame(columns=total_value.columns)

a_top0 = pd.read_hdf('top_company.hdf5', 'h8') * price_target_ave
a_top0 = (a_top0.fillna(0) - a_top0.shift(1).fillna(0)) / a_top0.shift(1).fillna(0)
for i in range(15, 40):
    # a_top = pd.read_hdf('top_company.hdf5','h8') - target_stock.append(pd.read_hdf('bottom_company.hdf5','h%d' %i)).fillna(0)#pd.read_hdf('top_company.hdf5','h9').fillna(0) - pd.read_hdf('top_company.hdf5','h7').fillna(0) #- hold6_8 #- ( pd.read_hdf('bottom_company.hdf5','h%d' %i).fillna(0))
    # a_top = a_top * price_target_ave / total_value

    a_top1 = target_stock.append(pd.read_hdf('bottom_company.hdf5', 'h%d' % i)).fillna(0) * price_target_ave
    a_top1 = (a_top1.fillna(0) - a_top1.shift(1).fillna(0)) / a_top1.shift(1).fillna(0)
    a_top = a_top0 - a_top1
    ic_TOP = corr_calculate(a_top.replace(0, np.nan))
    # print(i,ic_TOP.mean()[0].round(4),ic_TOP.std()[0].round(4),(ic_TOP.mean()/ic_TOP.std())[0].round(4))
    t_b_result = pd.DataFrame(
        [[ic_TOP.mean()[0].round(4), ic_TOP.std()[0].round(4), (ic_TOP.mean() / ic_TOP.std())[0].round(4)]],
        columns=['mean', 'std', 'ratio'])
    top_bottom_result = pd.concat([top_bottom_result, t_b_result], axis=0)

top_bottom_result = top_bottom_result.set_index(np.arange(15, 40))

# %%画出三年的不同类型机构表现情况
df = pd.DataFrame(columns=['18', '19', '20'], index=['all', 'C', 'B'])
for i, i1 in enumerate(['2018', '2019', '2020']):
    for j, j1 in enumerate([ic_all, ic_C, ic_B]):
        df.iloc[j, i] = j1.loc[i1].mean()

plt.plot(df)

a_top = pd.read_hdf('top_company.hdf5', 'h9').fillna(0) - pd.read_hdf('top_company.hdf5', 'h8').fillna(0)
a_top *= price_target_ave
ic_TOP = corr_calculate(a_top.replace(0, np.nan))
ic_TOP.mean()


# %%
# 计算持仓比例
def chicangfenlei(company=['B', 'C']):
    date = hold_18_20.index
    if len(company) == 1:
        date = hold_18_20[hold_18_20.com == company].index
        dsum = pd.DataFrame(hold_18_20[hold_18_20.com == company].groupby([date, 'stock']).sum().loc[:, 'quantity'])
    else:
        dsum = pd.DataFrame(hold_18_20.groupby([date, 'stock']).sum().loc[:, 'quantity'])

    return (dsum.unstack(level=1).droplevel(0, axis=1))  # * price_target_ave


# 持仓ic计算
# %%ic计算
# return_target = return_target.loc['2018-07-31':,:]
def corr_calculate(x):  # 计算不同机构持有市值的ic
    corr_frame = pd.DataFrame(index=x.index, columns=['corr'])
    for i in range(len(x) - 1):
        corr_frame.iloc[i, 0] = x.iloc[i, :].rank(ascending=False).corr(return_target.iloc[i + 1].rank(ascending=False))
    return corr_frame


# prop_all = chicangfenlei()
# prop_C = chicangfenlei('C')
# prop_B = chicangfenlei('B')
prop_all = pd.read_hdf('chicang.hdf5', 'all_num')
prop_C = pd.read_hdf('chicang.hdf5', 'C_num')
prop_B = pd.read_hdf('chicang.hdf5', 'B_num')

# 持仓比例ic
prop_all = (prop_all) * price_target_ave  # / total_value
prop_B = (prop_B) * price_target_ave / total_value
prop_C = (prop_C) * price_target_ave / total_value

# 持仓占比作差，乘以本月和上一个月的平均价格
n = 1
# price_target_a = (price_target_ave + price_target_ave.shift(20)) / 2
prop_add_all = (prop_all.fillna(0) - prop_all.shift(n).fillna(0)) / (prop_all.shift(n).fillna(0))
prop_add_B = (prop_B.fillna(0) - prop_B.shift(n).fillna(0)) / (prop_B.shift(n).fillna(0))
prop_add_C = (prop_C.fillna(0) - prop_C.shift(n).fillna(0)) / (prop_C.shift(n).fillna(0))

# 计算持仓市值最高的机构的持仓比例ic
# a_dsum = pd.DataFrame(holder_top_company.groupby([holder_top_company.index,'stock']).sum().loc[:,'quantity'])
# prop_top = (a_dsum.unstack(level = 1).droplevel(0, axis=1)) / total_num
# ic_p_top = corr_calculate(prop_top)
ic_p_all = corr_calculate(prop_add_all.replace(0, np.nan))
ic_p_C = corr_calculate(prop_add_C.replace(0, np.nan))
ic_p_B = corr_calculate(prop_add_B.replace(0, np.nan))
result_p_ic = pd.DataFrame([[ic_p_all.mean()[0], ic_p_all.std()[0], ic_p_all.mean()[0] / ic_p_all.std()[0]],
                            [ic_p_C.mean()[0], ic_p_C.std()[0], ic_p_C.mean()[0] / ic_p_C.std()[0]],
                            [ic_p_B.mean()[0], ic_p_B.std()[0], ic_p_B.mean()[0] / ic_p_B.std()[0]]],
                           columns=['mean', 'std', 'ir'], index=['all', 'C', 'B'])

# %%

ic_all_month = ic_p_all.groupby(pd.Grouper(freq='M')).apply(lambda x: x.mean())
ic_B_month = ic_p_B.groupby(pd.Grouper(freq='M')).apply(lambda x: x.mean())
ic_C_month = ic_p_C.groupby(pd.Grouper(freq='M')).apply(lambda x: x.mean())
ic_result_month = pd.concat([ic_all_month, ic_B_month, ic_C_month], axis=1)
ic_result_month.columns = ['北向', '券商', '银行']
ic_result_month.index = ic_result_month.index.strftime('%Y%m')

# plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
# plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置
# ic_result_month.plot.bar()

# %%
# most100stocks1 = prop_add_B.apply(lambda x:x.nlargest(100).index,axis = 1)
# most100stocks1 = most100stocks1.apply(lambda x: pd.Series(x)).stack().droplevel(1)
# most100stocks1 = pd.DataFrame(most100stocks1,columns = ['stock']).set_index([most100stocks1.index,'stock'])

# 先选出市值前100做样本池子，再看增持的IC
return_target = return_target.loc['2018-07-30':, :]
# port_ic = pd.DataFrame(columns = ['n','ic','ir'])
for n in range(1100, 2301, 100):
    # n=100
    most100stocks = prop_all.shift(19).apply(lambda x: x.nlargest(n).index.values, axis=1).apply(lambda x: pd.Series(x))
    most100stocks = most100stocks.stack().droplevel(1)
    # most100stocks.columns = ['stock']
    most100stocks = pd.DataFrame(most100stocks, columns=['stock']).set_index([most100stocks.index, 'stock'])
    add_B = pd.DataFrame(prop_add_all.stack())
    add_B.index.names = ['date', 'stock']
    merge_result = pd.merge(most100stocks, add_B, left_index=True, right_index=True).unstack().droplevel(0, axis=1)
    merge_result = merge_result.replace(0, np.nan)
    new = pd.DataFrame([[n, corr_calculate(merge_result).mean()[0],
                         corr_calculate(merge_result).mean()[0] / corr_calculate(merge_result).std()[0]]],
                       columns=['n', 'ic', 'ir'])
    port_ic = port_ic.append(new, ignore_index=True)
# print(n,corr_calculate(merge_result).mean()[0],corr_calculate(merge_result).std()[0],corr_calculate(merge_result).mean()[0] / corr_calculate(merge_result).std()[0])

# %%
n = 200
most100stocks = prop_add_B.apply(lambda x: x.nlargest(n).index.values, axis=1).apply(lambda x: pd.Series(x))
most100stocks = most100stocks.stack().droplevel(1)
# most100stocks.columns = ['stock']
most100stocks = pd.DataFrame(most100stocks, columns=['stock']).set_index([most100stocks.index, 'stock'])
add_B = pd.DataFrame(prop_all.shift(20).stack())
add_B.index.names = ['date', 'stock']
merge_result = pd.merge(most100stocks, add_B, left_index=True, right_index=True).unstack().droplevel(0, axis=1)
print(n, corr_calculate(merge_result).mean()[0], corr_calculate(merge_result).std()[0],
      corr_calculate(merge_result).mean()[0] / corr_calculate(merge_result).std()[0])

# %%持仓信息-持仓市值高的减低的机构持仓市值比因子
n = 100
# p0 = pd.read_hdf('top_company.hdf5','h8') - pd.read_hdf('bottom_company.hdf5','h35').fillna(0)
# p0 = p0 * price_target / total_value
a_top0 = pd.read_hdf('top_company.hdf5', 'h8') * price_target_ave
a_top0 = (a_top0.fillna(0) - a_top0.shift(1).fillna(0)) / a_top0.shift(1).fillna(0)
a_top1 = target_stock.append(pd.read_hdf('bottom_company.hdf5', 'h26')).fillna(0) * price_target_ave
a_top1 = (a_top1.fillna(0) - a_top1.shift(1).fillna(0)) / a_top1.shift(1).fillna(0)
a_top = a_top0 - a_top1

p = a_top.astype('float').apply(lambda x: x.nlargest(n).index, axis=1)
p = p.apply(lambda x: pd.Series(x))
p = p[p.index.isin(target_date.date)]
p = pd.DataFrame(p.stack().droplevel(1), columns=['stock'])
p = p.set_index([p.index, 'stock'])
t = total_value.stack()
t.index.names = ['date', 'stock']
t = pd.DataFrame(t)
p = p.merge(t, left_index=True, right_index=True)
p1 = p.unstack().div(p.unstack().sum(axis=1), axis=0)
p1 = p1.droplevel(0, axis=1).stack()
p1.to_csv('t_sub_b_add_prop%d.csv' % n)

# %%银行资金持仓市值比最高的TOP100
target_date = pd.read_csv('date.csv')
# for n in [6,7,8,9]:
n = 100
p = (prop_all).apply(lambda x: x.sort_values(ascending=False)[:n].index, axis=1)
p = p.apply(lambda x: pd.Series(x))
p = p[p.index.isin(target_date.date)]
p = pd.DataFrame(p.stack().droplevel(1), columns=['stock'])
p = p.set_index([p.index, 'stock'])
p.index.names = ['date', 'stock']
t = (prop_all * total_value).stack()  # 权重
t.index.names = ['date', 'stock']
t = pd.DataFrame(t)
p = p.merge(t, left_index=True, right_index=True)
p1 = p.unstack().div(p.unstack().sum(axis=1), axis=0)
p1 = p1.droplevel(0, axis=1).stack()
p1.to_csv('北向持仓占比持仓加权_top%d.csv' % n)

# %%
p = hold6_8 * price_target_ave
p = p[p.index.isin(target_date.date)]
# %%全体北向持仓top500之中 周增持top100
a_all = prop_C[prop_C.index.isin(target_date.date)].stack()
a_all1 = a_all.groupby('date').apply(lambda x: x.sort_values(ascending=False)[1500:2000]).droplevel(1)
a_add_all = prop_add_all[prop_add_all.index.isin(target_date.date)].stack()
a_result = pd.merge(pd.DataFrame(a_all1), pd.DataFrame(a_add_all), left_index=True, right_index=True)
a_result.columns = ['hold', 'add']
a_result.index.names = ['date', 'stock']
t = (prop_all * total_value).stack()  # 权重
t.index.names = ['date', 'stock']
t = pd.DataFrame(t)
a_result = a_result.merge(t, left_index=True, right_index=True)
a_result = a_result.drop(columns=['hold', 'add'])
a_result = a_result.groupby('date').apply(lambda x: x.sort_values(ascending=False, by=0)[:100]).droplevel(0).unstack()
a_result = a_result.div(a_result.sum(axis=1), axis=0)
a_result.droplevel(0, axis=1).stack().to_csv('银行top1500_2000北向增持100.csv')


# %%验证正态分布厚尾性
def mod(x):
    median = x.quantile(0.5)
    diff_median = ((x - median).abs()).quantile(0.5)
    max_range = median + 4 * diff_median
    min_range = median - 4 * diff_median
    return np.clip(x, min_range, max_range)


def z_score(x):
    var = x.std()
    return (x - var) / x.mean()


# %%持仓量最大的和最小的股票价格分布
n = 100
s = prop_all.apply(mod, axis=1)
s1 = s.iloc[n].sort_values()
length = (~s1.isnull()).sum()
s1 = s1.iloc[200:length - 200]
s1 = z_score(s1)  # z_score标准化
plt.hist(s1, bins=50)

m_v = prop_all * price_target_ave / total_value  # 北向资金持仓市值,total_value是个股流通市值

less_hold = prop_all.apply(lambda x: x.sort_values()[:100].index, axis=1).apply(lambda x: pd.Series(x))
less_hold_stack = less_hold.stack().droplevel(1)
less_hold_stack_frame = pd.DataFrame(index=[less_hold_stack.index, less_hold_stack.values])
less_hold_stack_frame.index.names = ['date', 'stock']

price_stack = m_v.stack()
price_stack.index.names = ['date', 'stock']
less_hold_value = less_hold_stack_frame.merge(pd.DataFrame(price_stack), left_index=True, right_index=True)
less_hold_value_mean = less_hold_value.unstack().apply(lambda x: x.mean(), axis=1)  # 最小的100个股平均北向资金持仓市值

most_hold = prop_all.apply(lambda x: x.sort_values(ascending=False)[:100].index, axis=1).apply(lambda x: pd.Series(x))
most_hold_stack = most_hold.stack().droplevel(1)
most_hold_stack_frame = pd.DataFrame(index=[most_hold_stack.index, most_hold_stack.values])
most_hold_stack_frame.index.names = ['date', 'stock']

price_stack = m_v.stack()
price_stack.index.names = ['date', 'stock']
most_hold_value = most_hold_stack_frame.merge(pd.DataFrame(price_stack), left_index=True, right_index=True)
most_hold_value_mean = most_hold_value.unstack().apply(lambda x: x.mean(), axis=1)  # 最大的100个股平均北向资金持仓市值


