用户贷款风险预测竞赛
==================
一. 赛题简介
--------
**融360.com  用户贷款风险预测**

**赛题链接：**  
https://www.pkbigdata.com/common/cmpt/%E7%94%A8%E6%88%B7%E8%B4%B7%E6%AC%BE%E9%A3%8E%E9%99%A9%E9%A2%84%E6%B5%8B_%E8%B5%9B%E4%BD%93%E4%B8%8E%E6%95%B0%E6%8D%AE.html

**赛题介绍：**  
* 竞赛提供了近7万贷款用户的基本身份信息、消费行为、银行还款等数据信息，需要参赛者以此建立准确的风险控制模型，来预测用户是否会逾期还款。  
* 参赛者可使用的训练数据包括用户的基本属性user_info、银行流水记录bank_detail、用户浏览行为browse_history、信用卡账单记录bill_detail、放款时间loan_time，以及用户是否发生逾期行为的记录overdue。（注意，并非每一位用户都有非常完整的记录）。相应地，需要预测是否逾期的用户也有除overdue外的数据信息。参赛者根据训练数据（约5.5万个用户）构建模型，使用模型对预测集（约1.5万个用户）进行预测，采用Kolmogorov-Smirnov(KS)统计量值衡量预测结果。（KS是风险评分领域常用的评价指标，KS值越高表明模型对正负样本的区分能力越强）。注意，数据经过了脱敏处理。   
  
二. 解决方案
---------
## 1. 尝试1：Data clean1 – ModelLR1 – ModelXGB1   
### **Data clean1**  
* **导入数据**    
&ensp;&ensp;&ensp;&ensp;将训练集train和未知样本集test中的同名数据表导入，上下连接，共得6张长表（user_info, bank_detail, browse_history, bill_detail, loan_time, overdue）。注意：有些表的userID不唯一，说明每个userID对应多条记录，后期需进行groupby处理，使得一个userID对应一条记录。  另外，未知样本集用户的overdue缺失，待预测。

* **描述性统计**  
  * **缺失值统计：**  
&ensp;&ensp;&ensp;&ensp;从特征的角度：所有训练集和未知样本集用户都有user_info，loan_time，overdue信息（当然未知样本集的overdue是缺失的，待预测），但只有14.4%的用户有bank_detail信息，85.4%的有browse_history信息，96.2%的有bill_detail信息。
&ensp;&ensp;&ensp;&ensp;从用户的角度：有约0.07%的训练集用户，约0.01%的未知样本集用户同时缺乏bank_detail, browse_history, bill_detail 信息。
&ensp;&ensp;&ensp;&ensp;缺失比例关系到之后的缺失值处理方式。

  * **描述性统计**    
&ensp;&ensp;&ensp;&ensp;仅对训练集进行描述性统计。对user_info、loan_time、overdue进行单变量描述统计，包括：极端值识别、要素分析（即违约分布统计）、正态性检验、并用条形图/直方图/箱图等图像展示。对于类别型特征，考虑是否需要合并频数较少的类别，但信息熵、基尼值、信息值IV都提示不合并类别更好。bill_detail表中，有一些无法解释的负值，约占bill_detail记录总数的4.3%，可考虑删除。

* **特征工程**  
&ensp;&ensp;&ensp;&ensp;user_info, loan_time, overdue表 的userID列都唯一且排过序，所以无需处理。bank_detail, browse_history, bill_detail表 因有同一userID对应多行记录的情况，所以需要进行数据整合，使得一个userID只有一条记录。
  * 通过sum、count、max、mean等聚合函数将数据整合为一个用户只对应一条记录的情况。同时，根据browse_time或billtime小于等于或大于loantime的情况，可以把记录分为放款前/后分别统计。另外，对于bank_detail部分来说，因只有14.4%的用户有bank_detail信息，所以会产生大量缺失，个人认为不宜使用特殊值/算法填充。而有85.4%的用户有browse_history信息，96.15%的用户有bill_detail信息，缺失较少，考虑采用算法填充缺失值。
  * 另外，还根据已有特征构建了少量新特征（如bank_detail表中的“支出占收入比”，bill_detail表中的“卡额度”等）并对个别特征进行了描述性统计。

* **汇总表与缺失值填充**  
&ensp;&ensp;&ensp;&ensp;将所有表汇总得到总表，每个用户对应一行记录。删去前边过程中收集到的含异常值的记录。然后，对bank_detail的每个类别特征，添加“缺失类别”，每个连续特征进行MDLP最优分箱离散化处理，将缺失值记为“缺失类别”。browse_history和bill_detail的缺失值使用算法填充（自定义函数RFR_fillna），即使用随机森林模型学习已有数据，然后对缺失数据进行预测，从而完成缺失值填充。最终得到完全无缺失的汇总表。  
  
### **ModelLR1**    
* 数据集没有缺失，将原始训练集划分为训练集和验证集，使用XGBOOST算法学习模型，计算所有特征的特征重要性并图示展示。
* 对数据集特征进行标准化，计算所有特征的经验相关系数矩阵，基于相关系数矩阵和XGB特征重要性进行特征选择，保留特征重要性高且与其它特征相关性低的特征。
* 对完整训练集训练LR模型。使用GridSearchCV交叉验证+ hyperopt求取最佳超参数值，训练模型对未知样本集进行预测，得到KS值为0.21，较低。  
  
### **ModelXGB1**   
* 数据集没有缺失。使用自定义网格搜索函数对XGB模型进行超参数的调参，其中采用AUC作为评价函数。
* 对完整训练集学习得到最终的XGB模型，对未知样本集进行预测，KS值为 0.38.  


## 2. 尝试2：Data clean2 – ModelLR2 – ModelXGB2   
### **Data clean2**    
&ensp;&ensp;&ensp;&ensp;Data clean2 在 Data clean1 的基础上添加了其它创建特征，且Data clean2未进行缺失值填充。  

### **ModelLR2**     
&ensp;&ensp;&ensp;&ensp;ModelLR2使用Data clean2得到的汇总表，含缺失。因XGBOOST模型可以学习有缺失的数据集，所以依然可以获得特征重要性，结果显示，缺失率高的，特征重要性也较低，所以直接删去高缺失的特征。剩下的缺失率低且特征重要性高的特征，采用随机森林算法填充缺失值。填充后的数据集标准化后，计算经验相关系数矩阵，对其进行特征值分解，由特征值结果判断大致需要的降维水平。  
&ensp;&ensp;&ensp;&ensp;然后，训练LR模型，将“标准化”、“PCA降维”、“训练LR模型”组成pipeline，通过GridSearchCV交叉验证+ hyperopt求取最佳超参数值，对完整训练集学习得到最终的LR模型，对未知样本集进行预测，得到KS值为0.19，较低（可能是删掉了一些特征的缘故）。
  
### **ModelXGB2**  
&ensp;&ensp;&ensp;&ensp;ModelXGB2使用Data clean2得到的汇总表，含缺失。因为XGB模型可以学习具有缺失的数据集，所以此处不进行缺失值填充，而且保留所有特征。对完整训练集学习得到XGB模型，对未知样本集进行预测，KS值为 0.40.  
  
三. 结果比较
---------
&ensp;&ensp;&ensp;&ensp;本人的最佳成绩为KS值0.40，排行榜最佳成绩为0.47.  
  
四. 附件
---------
* **Custom_Functions.py**&ensp;&ensp;本文件记录了需要调用的自定义函数。Calnan函数：此函数计算dataframe每一列的缺失比例。bank_detail_change函数：对特征进行MDLP最优分箱和转化。
* **MDLP_discretization.py**&ensp;&ensp;本文件为MDLP最优分箱算法。




