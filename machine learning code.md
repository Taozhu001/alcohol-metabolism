```python
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
from torch import nn
from d2l import torch as d2l
import numpy as np
import seaborn as sns
from imblearn.combine import SMOTEENN
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from matplotlib import pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import class_weight
from sklearn.svm import SVR  # 导入SVR类
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import learning_curve  # 导入learning_curve函数
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
# 使用 RF、SVM 及 ANN 3 种机器学习算法进行模型的构建
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
np.random.seed(42)
data=pd.read_csv("train_test_data.csv",encoding='gbk')
print(data.shape)
print(data.iloc[0:4,[0,1,2,3,4,-2,-1]])
all_features=data.iloc[:,3:]
print(all_features.shape)
print(all_features.iloc[0:4,[0,1,2,-2,-1]])
```

    (216, 211)
      molecule_chembl_id  label                           canonical_smiles  \
    0      CHEMBL2071012      0             CC1=C(C(=NN1C2=CC=CC=C2)C=O)O    
    1      CHEMBL2071013      0               COC(=O)C1=C(C(=NN1)C(=O)OC)O   
    2      CHEMBL4295018      0     C1=CC(=CC=C1C#N)C2=CC=C(C=C2)N3C=CN=C3   
    3      CHEMBL4287095      0  C1=CC(=CC=C1C2=CC=C(C=C2)N3C=CN=C3)C(=O)N   
    
       MaxEStateIndex  MinEStateIndex  fr_unbrch_alkane  fr_urea  
    0       10.588356       -0.061247                 0        0  
    1       10.952933       -0.838979                 0        0  
    2        8.782420        0.679431                 0        0  
    3       11.038312       -0.410180                 0        0  
    (216, 208)
       MaxEStateIndex  MinEStateIndex  MaxAbsEStateIndex  fr_unbrch_alkane  \
    0       10.588356       -0.061247          10.588356                 0   
    1       10.952933       -0.838979          10.952933                 0   
    2        8.782420        0.679431           8.782420                 0   
    3       11.038312       -0.410180          11.038312                 0   
    
       fr_urea  
    0        0  
    1        0  
    2        0  
    3        0  
    


```python
numeric_features=all_features.dtypes[all_features.dtypes!='object'].index
print(numeric_features)
```

    Index(['MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex',
           'MinAbsEStateIndex', 'qed', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt',
           'NumValenceElectrons', 'NumRadicalElectrons',
           ...
           'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene',
           'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene',
           'fr_unbrch_alkane', 'fr_urea'],
          dtype='object', length=208)
    


```python
#数据特征值和标签
data_features=torch.tensor(all_features.values,dtype=torch.float32)
data_labels=torch.tensor(data.label.values.reshape(-1,1),dtype=torch.float32)
data_labels=data_labels.ravel()
print(data_labels)
```

    tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    


```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
model = RandomForestRegressor(n_estimators=100)   # 替换为你想要的模型
rfe = RFE(model, n_features_to_select=60)  # 调整要选择的特征数量
X_rfe = rfe.fit_transform(data_features,data_labels)
```


```python
selected_features_index = rfe.support_
```


```python
X_selected=data_features[:,selected_features_index]
```


```python
X_train,X_test,y_train,y_test=train_test_split(X_selected,data_labels,test_size=0.25,stratify=data_labels,random_state=42)
```


```python
scaler=preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
```


```python
print(X_train.shape)
print(X_test.shape)
```

    (162, 60)
    (54, 60)
    


```python
# 将n_jobs设置为-1，以便使用CPU资源

# 定义要调优的超参数范围
param_dist_rf = {
    'n_estimators': [50,100,125],
    'max_features': [0.25, 0.5, 0.75],
    'max_depth': [10, 15, 20]
}

param_dist_svm = {
    'C': [2.0,3.0, 4.0, 5.0],
    'gamma': [0.01, 0.015, 0.02,0.03]
}

param_dist_ann = {
    'hidden_layer_sizes': [(75,),(100,),(150,),(200,)],
    'alpha': [0.01, 0.02, 0.03],
    'max_iter': [ 1500,2000,2500]
}

# 创建模型对象
rf_model = RandomForestClassifier()
svm_model = SVC(probability=True)
ann_model = MLPClassifier()

# 创建随机搜索对象并进行调参
grid_search_rf= RandomizedSearchCV(rf_model, param_dist_rf, cv=5, n_iter=27,random_state=42)   #n_iter参数是指在随机搜索过程中要尝试的参数组合数量。随机搜索是一种通过从参数空间中随机选择参数组合来进行模型调参的方法。
grid_search_svm = RandomizedSearchCV(svm_model, param_dist_svm, cv=5, n_iter=16,random_state=42)
grid_search_ann  = RandomizedSearchCV(ann_model, param_dist_ann, cv=5, n_iter=36,random_state=42)

# 训练模型并得出最佳超参数组合
grid_search_rf.fit(X_train,y_train)
grid_search_svm.fit(X_train,y_train)
grid_search_ann .fit(X_train,y_train)

# 输出最佳超参数组合
print("Random Forest最佳超参数组合：", grid_search_rf.best_params_)
print("SVM最佳超参数组合：", grid_search_svm.best_params_)
print("ANN最佳超参数组合：", grid_search_ann.best_params_)

```

    Random Forest最佳超参数组合： {'n_estimators': 100, 'max_features': 0.75, 'max_depth': 10}
    SVM最佳超参数组合： {'gamma': 0.015, 'C': 2.0}
    ANN最佳超参数组合： {'max_iter': 1500, 'hidden_layer_sizes': (75,), 'alpha': 0.02}
    


```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

# 使用最佳超参数组合的模型进行预测
y_pred_r = grid_search_rf.best_estimator_.predict_proba(X_test)[:,1]
y_pred_s = grid_search_svm.best_estimator_.predict_proba(X_test)[:,1]
y_pred_a = grid_search_ann.best_estimator_.predict_proba(X_test)[:,1]
y_pred_rf=(y_pred_r>0.5).astype(int)
y_pred_svm=(y_pred_s>0.5).astype(int)
y_pred_ann=(y_pred_a>0.5).astype(int)
# 计算准确率
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_ann = accuracy_score(y_test, y_pred_ann)

# 计算精确度
precision_rf = precision_score(y_test, y_pred_rf)
precision_svm = precision_score(y_test, y_pred_svm)
precision_ann = precision_score(y_test, y_pred_ann)

# 计算召回率
recall_rf = recall_score(y_test, y_pred_rf)
recall_svm = recall_score(y_test, y_pred_svm)
recall_ann = recall_score(y_test, y_pred_ann)

# 计算F1得分
f1_rf = f1_score(y_test, y_pred_rf)
f1_svm = f1_score(y_test, y_pred_svm)
f1_ann = f1_score(y_test, y_pred_ann)

# 计算AUC值
auc_rf = roc_auc_score(y_test, y_pred_rf)
auc_svm = roc_auc_score(y_test, y_pred_svm)
auc_ann = roc_auc_score(y_test, y_pred_ann)

# 打印结果
print("Random Forest准确率：", accuracy_rf)
print("Random Forest精确度：", precision_rf)
print("Random Forest召回率：", recall_rf)
print("Random Forest F1得分：", f1_rf)
print("Random Forest AUC值：", auc_rf)

print("SVM准确率：", accuracy_svm)
print("SVM精确度：", precision_svm)
print("SVM召回率：", recall_svm)
print("SVM F1得分：", f1_svm)
print("SVM AUC值：", auc_svm)

print("ANN准确率：", accuracy_ann)
print("ANN精确度：", precision_ann)
print("ANN召回率：", recall_ann)
print("ANN F1得分：", f1_ann)
print("ANN AUC值：", auc_ann)

# 绘制ROC曲线
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_r)
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, y_pred_s)
fpr_ann, tpr_ann, thresholds_ann = roc_curve(y_test, y_pred_a)

plt.plot(fpr_rf, tpr_rf, label='Random Forest')
plt.plot(fpr_svm, tpr_svm, label='SVM')
plt.plot(fpr_ann, tpr_ann, label='ANN')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# 绘制混淆矩阵图
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_svm = confusion_matrix(y_test, y_pred_svm)
cm_ann = confusion_matrix(y_test, y_pred_ann)

plt.figure(figsize=(10, 4))
plt.subplot(131)
sns.heatmap(cm_rf, annot=True, cmap='Blues', fmt='g')
plt.title('Random Forest')

plt.subplot(132)
sns.heatmap(cm_svm, annot=True, cmap='Blues', fmt='g')
plt.title('SVM')

plt.subplot(133)
sns.heatmap(cm_ann, annot=True, cmap='Blues', fmt='g')
plt.title('ANN')

plt.tight_layout()
plt.show()

```

    Random Forest准确率： 0.8703703703703703
    Random Forest精确度： 0.9166666666666666
    Random Forest召回率： 0.6470588235294118
    Random Forest F1得分： 0.7586206896551724
    Random Forest AUC值： 0.8100158982511924
    SVM准确率： 0.9259259259259259
    SVM精确度： 0.9333333333333333
    SVM召回率： 0.8235294117647058
    SVM F1得分： 0.8749999999999999
    SVM AUC值： 0.8982511923688394
    ANN准确率： 0.8518518518518519
    ANN精确度： 0.8
    ANN召回率： 0.7058823529411765
    ANN F1得分： 0.7500000000000001
    ANN AUC值： 0.8124006359300477
    


    
![png](output_10_1.png)
    



    
![png](output_10_2.png)
    



```python
data=pd.read_csv("predata.csv",encoding='gbk')
print(data.shape)
print(data.iloc[0:4,[0,1,2,3,4,-2,-1]])
all_features=data.iloc[:,2:]
print(all_features.shape)
print(all_features.iloc[0:4,[0,1,2,-2,-1]])
```

    (1358, 210)
          compound names                                             SMILES  \
    0           aluminum                                               [Al]   
    1         mucic acid  [C@@H]([C@@H]([C@H](C(=O)O)O)O)([C@@H](C(=O)O)O)O   
    2    glucuronic acid      [C@@H]1([C@@H]([C@H](OC([C@@H]1O)O)C(=O)O)O)O   
    3  D-Mannuronic Acid       [C@@H]1([C@@H]([C@H](OC([C@H]1O)O)C(=O)O)O)O   
    
       MaxEStateIndex  MinEStateIndex  MaxAbsEStateIndex  fr_unbrch_alkane  \
    0        0.000000        0.000000           0.000000               0.0   
    1       10.087469       -2.362037          10.087469               0.0   
    2       10.373695       -1.806019          10.373695               0.0   
    3       10.373695       -1.806019          10.373695               0.0   
    
       fr_urea  
    0      0.0  
    1      0.0  
    2      0.0  
    3      0.0  
    (1358, 208)
       MaxEStateIndex  MinEStateIndex  MaxAbsEStateIndex  fr_unbrch_alkane  \
    0        0.000000        0.000000           0.000000               0.0   
    1       10.087469       -2.362037          10.087469               0.0   
    2       10.373695       -1.806019          10.373695               0.0   
    3       10.373695       -1.806019          10.373695               0.0   
    
       fr_urea  
    0      0.0  
    1      0.0  
    2      0.0  
    3      0.0  
    


```python
numeric_features=all_features.dtypes[all_features.dtypes!='object'].index
print(numeric_features)
```

    Index(['MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex',
           'MinAbsEStateIndex', 'qed', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt',
           'NumValenceElectrons', 'NumRadicalElectrons',
           ...
           'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene',
           'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene',
           'fr_unbrch_alkane', 'fr_urea'],
          dtype='object', length=208)
    


```python
#数据特征值和标签
X_predict=torch.tensor(all_features.values,dtype=torch.float32)
print(X_predict.shape)
```

    torch.Size([1358, 208])
    


```python
X_pre = X_predict[:, selected_features_index]
```


```python
from sklearn.preprocessing import StandardScaler
# 创建 StandardScaler 对象
scaler = StandardScaler()
# 对训练数据进行拟合和转换
X_pre= scaler.fit_transform(X_train)
```


```python
# 假设你有新的数据集 X_new，可以通过模型的 predict_proba 方法获取预测概率
y_pred_rf_new = grid_search_rf.best_estimator_.predict_proba(X_pre )[:, 1]
y_pred_svm_new = grid_search_svm.best_estimator_.predict_proba(X_pre )[:, 1]
y_pred_ann_new = grid_search_ann.best_estimator_.predict_proba(X_pre )[:, 1]

# 获取前 30 个预测概率最大的样本索引
top_30_indices_rf_new = np.argsort(y_pred_rf_new)[::-1][:30]
top_30_indices_svm_new = np.argsort(y_pred_svm_new)[::-1][:30]
top_30_indices_ann_new = np.argsort(y_pred_ann_new)[::-1][:30]

compound_name_column=data["compound names"]
# 输出前 30 个样本的预测概率
print("Random Forest Top 30 Predictions for New Data:")
for i, index in enumerate(top_30_indices_rf_new):
    compound_name = compound_name_column.iloc[index]
    print(f"Sample {i + 1}:Compound Name {compound_name}, Probability {y_pred_rf_new[index]}")

print("\nSVM Top 30 Predictions for New Data:")
for i, index in enumerate(top_30_indices_svm_new):
    compound_name = compound_name_column.iloc[index]
    print(f"Sample {i + 1}: Compound Name {compound_name}, Probability {y_pred_svm_new[index]}")

print("\nANN Top 30 Predictions for New Data:")
for i, index in enumerate(top_30_indices_ann_new):
    compound_name = compound_name_column.iloc[index]
    print(f"Sample {i + 1}: Compound Name {compound_name}, Probability {y_pred_ann_new[index]}")

```

    Random Forest Top 30 Predictions for New Data:
    Sample 1:Compound Name Hederagenin 3-O-Arabinoside, Probability 1.0
    Sample 2:Compound Name licarin b, Probability 1.0
    Sample 3:Compound Name stachyose, Probability 1.0
    Sample 4:Compound Name Tangshenoside I, Probability 0.99
    Sample 5:Compound Name Licarin A, Probability 0.99
    Sample 6:Compound Name Panose, Probability 0.99
    Sample 7:Compound Name Licoleafol, Probability 0.98
    Sample 8:Compound Name Croomionidine, Probability 0.97
    Sample 9:Compound Name aspartic acid, Probability 0.97
    Sample 10:Compound Name Glyoxal, Probability 0.97
    Sample 11:Compound Name phytol, Probability 0.96
    Sample 12:Compound Name Tr-Saponin A, Probability 0.96
    Sample 13:Compound Name arginine, Probability 0.96
    Sample 14:Compound Name praeruptorin a, Probability 0.96
    Sample 15:Compound Name Isoeugenol, Probability 0.95
    Sample 16:Compound Name cymarin, Probability 0.95
    Sample 17:Compound Name dextrine, Probability 0.95
    Sample 18:Compound Name Sucrose, Probability 0.95
    Sample 19:Compound Name Astragaloside V, Probability 0.95
    Sample 20:Compound Name D-Mannuronic Acid, Probability 0.95
    Sample 21:Compound Name Macrostemonoside D, Probability 0.93
    Sample 22:Compound Name lysine, Probability 0.93
    Sample 23:Compound Name Canavanine, Probability 0.93
    Sample 24:Compound Name succinic acid, Probability 0.93
    Sample 25:Compound Name Platycodigenin, Probability 0.93
    Sample 26:Compound Name calcium sulphate, Probability 0.92
    Sample 27:Compound Name jujuboside a1, Probability 0.92
    Sample 28:Compound Name Glyuranolide, Probability 0.92
    Sample 29:Compound Name niga-ichigoside f1, Probability 0.92
    Sample 30:Compound Name Alpha-Cedrene, Probability 0.92
    
    SVM Top 30 Predictions for New Data:
    Sample 1: Compound Name stachyose, Probability 0.9945700731178494
    Sample 2: Compound Name Hederagenin 3-O-Arabinoside, Probability 0.9938823811669456
    Sample 3: Compound Name arginine, Probability 0.9938556352949578
    Sample 4: Compound Name Panose, Probability 0.9928144372189245
    Sample 5: Compound Name niga-ichigoside f1, Probability 0.9920377766390479
    Sample 6: Compound Name Licarin A, Probability 0.9916415125364878
    Sample 7: Compound Name dextrine, Probability 0.9915917087643027
    Sample 8: Compound Name cymarin, Probability 0.9905401738615899
    Sample 9: Compound Name licarin b, Probability 0.990438083348182
    Sample 10: Compound Name Tangshenoside I, Probability 0.9902269465982172
    Sample 11: Compound Name Platycodigenin, Probability 0.989803992400244
    Sample 12: Compound Name Tr-Saponin A, Probability 0.9874880949127154
    Sample 13: Compound Name D-Mannuronic Acid, Probability 0.9838032151182807
    Sample 14: Compound Name jujuboside a1, Probability 0.9831868809760335
    Sample 15: Compound Name Croomionidine, Probability 0.9815873844068622
    Sample 16: Compound Name calcium sulphate, Probability 0.9799534426006957
    Sample 17: Compound Name phytol, Probability 0.9795098433933558
    Sample 18: Compound Name Macrostemonoside D, Probability 0.978933198013044
    Sample 19: Compound Name Cis-Methyl Isoeugenol, Probability 0.9787758849946061
    Sample 20: Compound Name Licoleafol, Probability 0.9787726927474524
    Sample 21: Compound Name Soyasapogenol B, Probability 0.9787705481515608
    Sample 22: Compound Name citric acid, Probability 0.9787701652230466
    Sample 23: Compound Name lysine, Probability 0.9787633369057119
    Sample 24: Compound Name Astramembrannin I, Probability 0.9787539311057615
    Sample 25: Compound Name Glyoxal, Probability 0.9787536907426073
    Sample 26: Compound Name Echinatine, Probability 0.9787516715819492
    Sample 27: Compound Name canaline, Probability 0.9787494361744791
    Sample 28: Compound Name nootkatone, Probability 0.9787469239520245
    Sample 29: Compound Name 24-Methylenelophenol, Probability 0.9787447373177292
    Sample 30: Compound Name laminarin, Probability 0.9787432834091789
    
    ANN Top 30 Predictions for New Data:
    Sample 1: Compound Name dextrine, Probability 0.9999510370295767
    Sample 2: Compound Name Platycodigenin, Probability 0.9998582321791052
    Sample 3: Compound Name Glyoxal, Probability 0.999832104309334
    Sample 4: Compound Name Tr-Saponin A, Probability 0.9998065480787875
    Sample 5: Compound Name aspartic acid, Probability 0.9997692945861957
    Sample 6: Compound Name stachyose, Probability 0.9996937729553617
    Sample 7: Compound Name Panose, Probability 0.9992951513556686
    Sample 8: Compound Name Licarin A, Probability 0.9992509981934541
    Sample 9: Compound Name jujuboside a1, Probability 0.999248769835219
    Sample 10: Compound Name calcium sulphate, Probability 0.9991632228216785
    Sample 11: Compound Name Hederagenin 3-O-Arabinoside, Probability 0.9990366882314735
    Sample 12: Compound Name niga-ichigoside f1, Probability 0.9987828696425416
    Sample 13: Compound Name Astramembrannin I, Probability 0.998697836566945
    Sample 14: Compound Name Croomionidine, Probability 0.9986773423023344
    Sample 15: Compound Name cymarin, Probability 0.9985727887931578
    Sample 16: Compound Name Licoleafol, Probability 0.9983659190855877
    Sample 17: Compound Name arginine, Probability 0.9980583288315045
    Sample 18: Compound Name licarin b, Probability 0.9979358147600187
    Sample 19: Compound Name D-Mannuronic Acid, Probability 0.9965056553914563
    Sample 20: Compound Name Isoeugenol, Probability 0.9953839971747056
    Sample 21: Compound Name Astragaloside V, Probability 0.9949395083280875
    Sample 22: Compound Name canaline, Probability 0.9939544069400933
    Sample 23: Compound Name ursiniolide a, Probability 0.9932596797322953
    Sample 24: Compound Name Cis-Methyl Isoeugenol, Probability 0.9929508943867874
    Sample 25: Compound Name laminarin, Probability 0.9928259614298487
    Sample 26: Compound Name praeruptorin a, Probability 0.9914126885958824
    Sample 27: Compound Name succinic acid, Probability 0.9898872515992762
    Sample 28: Compound Name Tangshenoside I, Probability 0.9890622374407438
    Sample 29: Compound Name Canavanine, Probability 0.9866883146050023
    Sample 30: Compound Name phytol, Probability 0.9858470888140076
    


```python

```


```python

```
