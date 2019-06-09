
# coding: utf-8

# 说明：  
# &ensp;&ensp;MDLP_discretization_(myself_improve--provide n+my characters)-Copy1 是
# MDLP_discretization_(myself_improve--provide n+my characters) 的复制版本，并改动了一些 print 的语句。
# 

# $$loss = 加权熵和值 + a*n$$ 

# In[4]:


import pandas as pd
import numpy as np

'''
计算一个数据集的熵
'''
# CalEnt函数的作用是计算一个数据集的熵。df是dataframe，varname是待分箱的连续变量的列名，
# 注意，整个df已经按照varname变量的值由小到大排序过了。tarname是目标类别变量的列名。
# b和e是数值型行索引（0,1,2,3,,,那种），指的是要把此连续变量从b到e截取出来（目标变量跟着截取）
# 把截取到的这一段作为一个数据集，计算此数据集的熵。（包括b和e两个索引处的数据）
# 返回值 Ent 就是这个数据集的熵。

def CalEnt(df, varname, tarname, b, e):
    if b>e :
        return 0.0
    
    # b<=e 时
    Ent = 0.0
    dfc = df.iloc[b:(e+1), :]  # dfc 是从 df 中截取到 的行索引从 b到 e的一截。（包含b,e）。
    cat_percent = dfc[tarname].value_counts()/dfc.shape[0]  # 此得到的是各个 pk
    
    for k in range(len(cat_percent)):
        if cat_percent.iloc[k] == 0 :
            Entk = 0           # 我认为Entk是for循环的内部变量。但是Ent是外部变量。
        else :
            Entk = cat_percent.iloc[k]*np.log2(cat_percent.iloc[k])
        Ent = Ent - Entk
    
    return Ent


'''
计算一个二分划分的熵
'''
# CalEnt_cutbins函数的作用是计算：当用 P 作为分割点将 df的 b~e 这一段（母箱）二分成两个子箱时，
# 两个子箱的加权熵和值，即 Ent_cutbins。此值越小，说明划分点 P 越好，划分得到的两个子箱越纯。
# 注意，P 也是索引，两个子箱，左箱是 b~P,右箱是 (P+1)~e。（边界都包含）
# 同时，用字典 d 记录两个子箱的信息，每个子箱的信息包括首位索引、此箱的熵值。
# 返回两个数据，一为字典d，二为 加权熵和值Ent_cutbins

def CalEnt_cutbins(df, varname, tarname, b, e, P):  # P的范围： b<=P<= e
    d={}  # 用来存储两个箱各自的首尾索引和熵值
    
    EntL = CalEnt(df, varname, tarname, b, P)     # 左箱的熵
    EntR = CalEnt(df, varname, tarname, P+1, e)   # 右箱的熵
    Ent_cutbins = (P-b+1)/(e-b+1)*EntL + (e-P)/(e-b+1)*EntR   # 左右两子箱的加权熵和值
    # print("划分点为{0}时，两子箱的加权熵和值为: {1}".format(P, Ent_cutbins))  # 打印加权熵和值
                                                                                 # 此值越小越好。
    d["L"] = [b, P, EntL]
    d["R"] = [P+1, e, EntR]
    
    return (d, Ent_cutbins)



'''
计算若干个分箱的加权熵和值。
'''
# CalEnt_bins函数的作用是：一个连续变量被分箱后，计算各个箱的加权熵和值，也就是这个连续变量（数据集）
# 被分箱之后的熵值。df是dataframe，varname是被分箱的连续变量的列名，注意，
# 整个df已经按照varname变量的值由小到大排序过了。tarname是目标类别变量的列名。
# b和e是数值型行索引（0,1,2,3,,,那种），指的是要把此连续变量从b到e截取出来（目标变量跟着截取）
# 把截取到的这一段作为一个数据集（母箱），对它调用 Discretization 函数进行分箱，返回存储各个子箱信息的二维列表
# binsinfo。对binsinfo执行循环，每次循环，i都取出一个子箱的信息，i[0],i[1]分别是此子箱的首位索引（包含），
# i[2]是此子箱的熵值。而 e-b+1 正是此连续变量从b到e这一段（即母箱）所含的数值个数。由此可以求出各个
# 子箱的加权熵和值，即 Ent_bins。

def CalEnt_bins(df, varname, tarname, b, e, binsinfo):  
    Ent_bins = 0
    for i in range(len(binsinfo)):
        Ent_bins = Ent_bins + (binsinfo[i][1]-binsinfo[i][0]+1)/(e-b+1)*binsinfo[i][2]
    return Ent_bins
    



'''
对一个已排序的数值连续变量进行分箱
'''
# Discretization函数对 df 的连续变量的 b~e 段这个箱进行循环分箱。参数 a 是正则系数。
# a=0，表示完全不考虑正则，那尽量往多了分，会分成最多的箱，因为这样的 loss 可以最低。
# a 较大时，表示对分箱的惩罚大，那尽量少分，会分成不是太多的箱。

def Discretization(df, varname, tarname, b, e, a):   
    
    Ps = []         # 存储最终的划分点
    binsinfo = [[b, e, CalEnt(df, varname, tarname, b, e)]]
     # 先把完整的箱加入到binsinfo，此二维list存储最终的 各个分箱 的信息.
     # 注意，我之前尝试了将 binsinfo 用二维ndarray存储，但是极不方便，有两个问题：
     # 添加 元素时，需要把元素先转换为 二维ndarray形式，如 [[X,X,X,X]]，才能 append到 binsinfo里。
     # 二维ndarray要求每个元素的类型相同，所以，因为熵是小数，所以b和e会是小数，所以在range(b,e)时会出问题，
     # 需要先把b，e转换为int才可以。
    
    
    '''
    计算二分划分的最佳划分点
    '''
    # best_cut_point函数是discretization函数的嵌套內建函数。作用是对 df 的 连续变量的 b~e段这个箱进行最佳二分。
    # 得到最佳二分点 P。计算分箱前 和 用 P 分箱后的 loss，如果分箱不能降低 loss，则不分箱，保持未分原样，返回
    # 状态码 1.如果分箱能降低 loss，则分箱，返回状态码 0，并将 P添加到 Ps。并将划分所得子箱添加到 binsinfo，
    # 给每个箱添加标记位：0表示可再分，1表示不可再分。
    # 此函数使用了 nonlocal 技术，在函数内部改变了 Ps 和 binsinfo两个变量。即使退出 best_cut_point函数，
    # 这种改变在discretization函数中依然可见。 但本函数还是返回了状态码，因为状态码可以指示接下来Discretization
    # 函数该怎么做。
    
    def best_cut_point(df, varname, tarname, b, e, a):  # 此函数借鉴源代码
        nonlocal Ps
        nonlocal binsinfo
        
        P = -1
        min_Ent_cutbins = 0
        Ent_cutbins = 0
        best_bins = {}  # 存储用最佳二分点划分后得到的两个箱的信息
        tmpindex = -1   # 把分得的子箱添加到母箱前，需要先从 binsinfo 中删掉母箱，tmpindex用来记录母箱的index
    
        for i in range(b, e):
            # print("----------遍历点为i: {}---------".format(i))  # 遍历点不一定等于划分点，遍历到i时，如果有重复元素，执行了i=i+1，
            # while (i < end) & (S[i] == S[i+1]):               # 则划分点不是这里的 i，而是改变后的 i。            
                # i += 1                      # 但 CalEnt_cutbins 计算熵时，还会打印一次划分点，这时才是真正的划分点，即改变后的 i
            while (i<e):
                if (df[varname][i] == df[varname][i+1])&(df[tarname][i] == df[tarname][i+1]): # 即第i和第i+1是重复元素
                    i+=1
                else:                                           # 此while循环确定了本轮使用的分割点索引值 i
                    break
                
            d, Ent_cutbins = CalEnt_cutbins(df, varname, tarname, b, e, i)
        
            if P == -1:                                         # 此 if-elif 找到了最佳二分点 P
                P = i
                min_Ent_cutbins = Ent_cutbins
                best_bins = d
            elif Ent_cutbins < min_Ent_cutbins:
                P = i
                min_Ent_cutbins = Ent_cutbins
                best_bins = d
        # print("The choosen P is: ", P)    
        # print("------------------------------")
        
        
        # 找到最佳二分点后，接下来，本该先从 binsinfo里删掉母箱，然后把此二分点划分得到的两个子箱添加进 binsinfo,
        # 但此处不这样做。而是先计算当前的（还没有用这一轮找到的最佳二分点划分的） binsinfo 里的所有箱的加权熵和值，
        # 加上正则项，即当前的binsinfo 里的箱数乘 a，记为 original_loss。然后，给 当前的 Ps 和 binsinfo 做个备份，
        # 即 Ps_tmp，binsinfo _tmp。然后，从 binsinfo 里删掉母箱，用找到的最佳二分点划分所得的子箱添加进 binsinfo，
        # 然后，再对这个新的 binsinfo 求 loss，记为 new_loss。比较 original_loss和 new_loss。如果前者大，说明二分
        # 有利于 loss 降低，故进行二分，使用二分后的binsinfo和Ps。如果前者小于等于后者，说明二分不能降低 loss，
        # 若是二分更好，则 return 0, 而一旦有某一个箱二分后出现 loss增加的情况，就不进行划分，而且返回1. 
        # 外函数 Discretization 可以根据返回值，决定下一步动作。若接受到 0，则对 binsinfo 里的箱选择熵最大的，进行分箱。
        # 如果接受到 1，则直接退出 Discretization。
        
        
        original_loss = CalEnt_bins(df, varname, tarname, b, e, binsinfo) + a*len(binsinfo) # 用 P二分前，原始的 loss
        Ps_tmp = Ps
        binsinfo_tmp = binsinfo
        
        
        for j in range(len(binsinfo)):                    # 找到 binsinfo 中的 [b  e]分箱，删掉它
            if (binsinfo[j][0]==b)&(binsinfo[j][1]==e):   # 因为接下来要把二分得到的子箱添加进 binsinfo，在此之前要删掉母箱
                tmpindex = j
        binsinfo.pop(tmpindex)          # 注意，千万不能把 删除这一步放到for循环里。
        
        
        if P == e :
            # best_bins.pop("R")     # 这步其实不用，因为最后添加到 binsinfo时，只添加左箱即可。
            best_bins["L"].append(1) # 注意这里标记左箱为 1，表示无法再分，因为这一轮划分就没能划分[b  e]，它已经是熵最低，以后也无法划分。
            binsinfo.append(best_bins["L"])
        
        else:
            Ps.append(P)   # P不等于e时，才把 P 纳入 Ps。所以 Ps里的 P值有： b <= P 。
            best_bins["L"].append(0)
            best_bins["R"].append(0)
            if P==b:
                best_bins["L"][3] = 1   # 即左箱只有一个元素时，肯定不可再分，所以标记为1
            if (P+1)==e:
                best_bins["R"][3] = 1   # 即右箱只有一个元素时，肯定不可再分，所以标记为1
            
            binsinfo.append(best_bins["L"])
            binsinfo.append(best_bins["R"])
        
        
        new_loss = CalEnt_bins(df, varname, tarname, b, e, binsinfo) + a*len(binsinfo) # 用 P 二分后的 loss
        
        if original_loss <= new_loss :
            Ps = Ps_tmp 
            binsinfo = binsinfo_tmp 
            return 1  # 状态码，表示不划分更好且没有划分，保持原状。
        else:
            return 0  # 状态码，表示划分更好且已划分。
        
        
                    
     # ----------------------------------------------------------------best_cut_point函数定义结束         

    
    if (b==e) :
        # print("The length of this variable is 0. It cannot be discretized.")   # 说明给的连续变量长度为0，无法划分
        return
    elif b>e :
        # print("Something wrong with the length of this variable.")
        return
    else:                           # 保证 b < e，然后调用best_cut_point进行第一次划分，得到两个 或 一个子分箱
        state = best_cut_point(df, varname, tarname, b, e, a)  
        if state == 1:  # 说明分箱使 loss 增加，那就不分箱，保持 binsinfo 和 Ps 原状，退出 Discretization 函数。
            #print("Discretization increase the loss. Undo the discretization and quit. ")
            Ps.sort()
            binsinfo.sort()
            print("-------","结果为 ",varname," -------")
            print("划分点： Ps = ",Ps)
            print("各分箱： binsinfo = ", binsinfo)
            return (Ps, binsinfo)
    
    # 若 state == 0，说明分箱使得 loss 减小，则分箱，然后接着对 binsinfo 选择熵最大的箱进行二分。
    while True :            # 两种可能终止此 while循环。1.binsinfo里所有箱都已最优，无法再分。
        max_Ent = 0         # 2.对 binsinfo里某个选中的箱进行分箱，出现 loss 增加的情况。
        choosebin = -1
        
        binsinfo_array = np.array(binsinfo)  # 为了用numpy的函数，先把二维列表转化为 ndarray
        if np.sum(binsinfo_array[:,3]) == len(binsinfo_array):    # 如果binsinfo的所有分箱的标记位都是1
            #print("Now all bins are optimal. They cannot be discretized.")  # 则所有分箱都已是最优，无法再分。
            Ps.sort()
            binsinfo.sort()
            print("-------","结果为 ",varname," -------")
            print("划分点： Ps = ",Ps)
            print("各分箱： binsinfo = ", binsinfo)
            return (Ps, binsinfo)
        
        for i in range(len(binsinfo)):     # 这段for循环是为了找出现有分箱中，标记为0且熵最大的箱，下一次分箱就对它分箱
            if binsinfo[i][3] == 0:
                if choosebin == -1 :
                    choosebin = i
                    max_Ent = binsinfo[i][2]
                elif binsinfo[i][2] > max_Ent:
                    choosebin = i
                    max_Ent = binsinfo[i][2]     
        
        state = best_cut_point(df, varname, tarname, int(binsinfo[choosebin][0]), int(binsinfo[choosebin][1]), a )
        if state == 1:  # 说明分箱使 loss 增加，那就不分箱，保持 binsinfo 和 Ps 原状，退出 Discretization 函数。
            #print("Discretization cannot decrease the loss. Undo the discretization and quit. ")
            Ps.sort()
            binsinfo.sort()
            print("-------","结果为 ",varname," -------")
            print("划分点： Ps = ",Ps)
            print("各分箱： binsinfo = ", binsinfo)
            return (Ps, binsinfo)
        # 若 state == 0，说明分箱使得 loss 减小，则分箱，然后接着执行while循环对 binsinfo 选择熵最大的箱进行二分。
        
        
 
    print("-------","结果为 ",varname," -------")
    print("划分点： Ps = ",Ps)
    print("各分箱： binsinfo = ", binsinfo)
    return (Ps, binsinfo)
                    
                    

