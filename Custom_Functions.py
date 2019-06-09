
# coding: utf-8

# 本文件记录了 Risk Pre2 项目中使用的自定义函数。

# In[10]:


import numpy as np
import pandas as pd
import MDLP_discretization as MDLP 


# In[2]:


# Calnan 函数

# 定义了一个计算 缺失值数目的函数 Calnan。
# df 是待计算的 dataframe，本函数可以计算 df 的每一列的缺失值 nan 的数目。
# 返回的是一个 dataframe，有三列，第一列是 df的每个列名，第二列 nancount是 df每一列有多少个缺失值。
# nanpercent是 df 的每一列的缺失值占 df 总条目数的比例。

def Calnan(df):
    num = df.shape[0]     # df 的总条目数
    outcome = []
    for i in df.columns:  # 对 df 的每列进行循环
        nancount = np.sum(df[i].isnull())  
        nanpercent = nancount/num
        outcome.append([i, nancount, nanpercent])
    return pd.DataFrame(outcome, columns=["column_name","nancount","nanpercent"])


# In[3]:


# discrete_and_substitute, test_substitute, nan_substitute, continuousvar_change, bank_detail_change函数 用来对
# bank_detail 部分进行最优分箱和转化。bank_detail_change函数 返回转化后的训练集和未知样本集。

# 本函数实现对无缺失的连续数值变量进行MDLP最优分箱，并把各箱从上到下用0,1,2,,等数值取代。
# 准确来说，本函数适用于 训练集，而且是待分箱的连续变量和对应的目标变量都无缺失的部分训练集。

# 本函数需提供参数：df_train，即上述数据集。ID，注意，此数据集除了待分箱连续变量和目标变量外，还必须有一个对应的“无重复值”的列，
# 因为最后返回的替换后的字段，还是要结合到总表上的，但此字段的值已被替换，所以就必须有一个参照列，能够把此字段的每一行和原来总表
# 的行对应起来。目标变量不能充当参照列，因为目标变量是0/1二值的有大量重复。参照列必须是“无重复”的列，才能准确无误的匹配。
# 针对本赛题，userID可以很好地充当这一角色。
# varname，待分箱的连续变量的列名。tarname，目标变量列名。a，MDLP 算法的 正则系数，可调整分箱数。
# 根据df_train这部分数据集，进行MDLP分箱，然后把分得的箱分别用数字0,1,2,,,代替，则最后得到的连续变量里都是0,1,2,,等值，不再是以前的
# 连续数值。另外注意，df_train 无需事先排序。

# 返回值为三个构成的一个元组，分别是 binsinfo，各个分箱的情况（参 MDLP算法介绍）。
# binsvalue，即各个箱的首尾索引对应的varname变量的值，即每个箱的值域，这个在之后给未知样本分箱时需要。此为二维list格式，与binsinfo一样。
# tmp，被用0,1,2,,等值替换了的 dataframe。


def discrete_and_substitute(df_train, ID, varname, tarname, a):  # 分箱加替换
    tmp = df_train.loc[df_train[varname].notnull(),[ID, varname, tarname]]   # 选出训练集中 varname 不为nan的那些行。
    tmp.sort_values(by=varname, inplace=True)   # 原地对varname字段排序。（但此时行索引可能是5,3,1,2,,杂乱无序的）
    tmp.index = pd.RangeIndex(0,tmp.shape[0])   # 原地修改索引为0,1,2,3,,否则调用 Discretization函数时会出错。
    Ps,binsinfo = MDLP.Discretization(tmp, varname, tarname, 0, tmp.shape[0]-1, a)
    
    bins_num = len(binsinfo)  # 得到箱数
    binsvalue = []   # 创建 binsvalue记录各个箱的首尾索引对应的varname变量的值。
    for i in range(bins_num):
        # 将第i个箱的首尾索引对应的varname变量的值作为列表添加到 binsvalue里。binsvalue最后是二维列表。
        b = binsinfo[i][0]
        e = binsinfo[i][1]
        binsvalue.append([tmp[varname][b],tmp[varname][e]]) 
        tmp.loc[b:e, varname] = i   # 在 loc 选取中，是前闭后闭。第一个箱的值用0替换，第二个箱用1，第三个箱用2,,,
    
    tmp[varname] = tmp[varname].astype(np.int64)  # 转换数据类型为整数。
    print("binsvalue = ",binsvalue)  
    return (binsinfo, binsvalue, tmp)  


# In[4]:


# 准确来说，本函数适用于 未知样本集，而且是待分箱的连续变量无缺失的部分未知样本集。
# 训练集 已调用discrete_and_substitute函数进行最优分箱和替换后，需要根据各个箱和其对应的替换值，对未知样本集中此连续变量上的值进行替换。
# 即若未知样本集上此连续变量值不为nan时，这个值落在哪个分箱？应该用什么值替换？（落在第一个箱用0替换，第二个箱用1替换，第三个...）
# 本函数做这个判断工作和实质的替换工作。

# 本函数需提供参数：df_test，即上述数据集。ID，同discrete_and_substitute函数，是必须有的，且是“非重复”的列。
# 针对本赛题，userID可以很好地充当这一角色。
# varname，待分箱的连续变量的列名。binsvalue，用MDLP 算法对训练集进行分箱时得到的分箱信息，即每个箱的值域，本函数中要用到。
# 另外注意，df_train 无需事先排序。

# 返回值为两个构成的一个元组，分别是 binsvalue，其实它是作为参数传进来的，现在再传出去。
# tmp，被用0,1,2,,等值替换了的 未知样本部分集（也就是把传进来的 df_test 进行值替换）。


def test_substitute(df_test, ID, varname, binsvalue):
    tmp = df_test.loc[df_test[varname].notnull(),(ID, varname)]   # 选出未知样本集中 varname 不为nan的那些行。
    tmp.sort_values(by=varname, inplace=True)   # 原地对varname字段排序。（但此时行索引可能是5,3,1,2,,杂乱无序的）
    tmp.index = pd.RangeIndex(0,tmp.shape[0])   # 原地修改索引为0,1,2,3,,否则之后 tmp[var][k] 会出错。
    
    bins_num = len(binsvalue)  # 得到箱数
    tmparray = tmp[[ID,varname]].values  # 得到底层 ndarray，用 ndarray判断与替换，速度会快很多很多，直接用 tmp[varname][k] 基本无法运行
    for k in range(tmparray.shape[0]):    # 为选出的未知样本部分集中 的每一个 varname值进行检查
        for i in range(bins_num):
            s = binsvalue[i][0]
            l = binsvalue[i][1]
            if (s<=tmparray[k][1])&(tmparray[k][1]<=l):
                tmparray[k][1]=i
            elif tmparray[k][1]<binsvalue[0][0]:
                tmparray[k][1]=0
            elif tmparray[k][1]>binsvalue[bins_num-1][1]:
                tmparray[k][1]=bins_num-1
    
    tmp = pd.DataFrame(tmparray,columns=[ID,varname])
    tmp[ID]=tmp[ID].astype(np.int64)   # 因为使用了ndarray，导致userID也变为float型，所以这里更改为int64
    tmp[varname] = tmp[varname].astype(np.int64)    # varname也转成int64
    return binsvalue, tmp


# In[5]:


# 此函数负责从训练集和未知样本集里提取出var字段里为nan的样本，将它们视为“缺失箱”，并把它们的值由nan替换成类别数值。
# 此函数的参数：df_train训练集，df_test未知样本集，ID同discrete_and_substitute 函数，是必须有的，且是“非重复”的列。
# 针对本赛题，userID可以很好地充当这一角色。varname，被分箱的连续变量的列名。binsinfo，用MDLP 算法对训练集进行分箱时得到的分箱信息，
# 本函数要用到箱数，所以需要它。另外注意，df_train和df_test无需事先排序。
# 此函数的返回值：tmp1，是训练集中var为nan的那些训练集样本被替换为数字类别值之后返回的dataframe，
# tmp2是 未知样本集中var为nan的那些未知样本集样本被替换为数字类别值之后返回的dataframe。


def nan_substitute(df_train, df_test, ID, varname, binsinfo):
    bins_num = len(binsinfo)
    tmp1 = df_train.loc[df_train[varname].isnull(),[ID, varname]]
    tmp1[varname] = bins_num  # 例如bins_num=3，即分3箱，则它们的代号为0,1,2. 而此处的缺失值作为“缺失值箱”，代号就是 3，即需要
                              # 用 3 替换。而 3 正是 bins_num。
    tmp2 = df_test.loc[df_test[varname].isnull(),[ID, varname]]
    tmp2[varname] = bins_num 
    
    tmp1[varname] = tmp1[varname].astype(np.int64)  # 转换数据类型为整数。
    tmp2[varname] = tmp2[varname].astype(np.int64)  # 转换数据类型为整数。
    return tmp1, tmp2


# In[6]:


# 此函数通过对传入的训练集的非nan的var数据和对应的tar数据进行MDLP，找到最佳分箱，并把训练集的非nan的var部分进行转化。
# （第一个箱的var值用0替代，第二个箱的用1替代，第三个箱的用2替代….）。然后，把未知样本集的非nan的var部分也套用此分箱方式进行替换。
# 然后，把为nan的训练集var和测试集var部分另作为“缺失箱”，用3替换…（当然，这些数值是相对的，还是要看MDLP分得的箱数）。
# 反正，缺失箱 是独立于 MDLP分得的箱。若MDLP分得n箱，则缺失箱作为第n+1箱。
# 本函数的参数：df_train，即上述数据集。df_test，即未知数据集。ID，同discrete_and_substitute函数，是必须有的，且是“非重复”的列。
# 针对本赛题，userID可以很好地充当这一角色。varname，待分箱的连续变量的列名。tarname，目标变量列名。
# a，MDLP 算法的 正则系数，可调整分箱数。注意，df_train和df_test不用事先排序。但是，如果userID是这两表的行索引而非列，
# 则要先把userID变为列。
# 本函数的返回值：tmp_train_change是已经替换好的完整的训练集，以userID为行索引，以var为唯一列，且行索引已排序。
# tmp_test_change是已经替换好的完整的测试集，以userID为行索引，以var为唯一列，且行索引已排序。


def continuousvar_change(df_train, df_test, ID, varname, tarname, a):
    binsinfo, binsvalue, tmp_train = discrete_and_substitute(df_train, ID, varname, tarname, a)
    tmp_train.drop(tarname, axis=1, inplace=True)
    binsvalue, tmp_test = test_substitute(df_test, ID, varname, binsvalue)
    tmp_train_nan, tmp_test_nan = nan_substitute(df_train, df_test, ID, varname, binsinfo)
    tmp_train_change = pd.concat([tmp_train.set_index(ID),tmp_train_nan.set_index(ID)]).sort_index()
    tmp_test_change = pd.concat([tmp_test.set_index(ID),tmp_test_nan.set_index(ID)]).sort_index()
    return tmp_train_change,tmp_test_change


# In[9]:


# 此函数 实现了对 bank_detail部分（除了“是否有工资列”外）的各个列进行最优分箱和替换。
# 最终返回的 bank_detail_train_changed,bank_detail_test_changed 是各个列已经用 数值0,1,2等（给箱起的代号值）替换了的类别变量。

# 本函数的参数：df_train，即训练集。df_test，即未知数据集。ID，同discrete_and_substitute函数，是必须有的，且是“非重复”的列。
# 针对本赛题，userID可以很好地充当这一角色。varname，所有待分箱的连续变量的列名构成的列表。tarname，目标变量列名。
# a，MDLP 算法的 正则系数，可调整分箱数。注意，df_train和df_test不用事先排序。但是，如果userID是这两表的行索引而非列，
# 则要先把userID变为列。
# 本函数的返回值：bank_detail_train_changed是已经替换好的完整的训练集，以userID为行索引，且行索引已排序，varnames里有多少列，
# 这里就有多少列。bank_detail_test_changed是已经替换好的完整的未知样本集，以userID为行索引，且行索引已排序，varnames里有多少列，
# 这里就有多少列。

def bank_detail_change(df_train, df_test, ID, varnames, tarname, a):
    train_vars_changed = []
    test_vars_changed = []
    
    for v in varnames:
        train_v_changed,test_v_changed = continuousvar_change(df_train, df_test, ID, v, tarname, a)
        train_vars_changed.append(train_v_changed)
        test_vars_changed.append(test_v_changed)
    
    bank_detail_train_changed = pd.concat(train_vars_changed,axis=1)  # 此处用 concat只需要一步，用 merge要很多步
    bank_detail_test_changed = pd.concat(test_vars_changed,axis=1) 
    
    return bank_detail_train_changed,bank_detail_test_changed
    

