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
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import learning_curve  
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
model = RandomForestRegressor(n_estimators=100)   
rfe = RFE(model, n_features_to_select=60)  
X_rfe = rfe.fit_transform(data_features,data_labels)
```


```python
selected_features_index = rfe.support_
```


```python
X_selected=data_features[:,selected_features_index]
selected_feature_names = all_features.columns[selected_features_index]
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
svm_model = SVC(kernel='linear',probability=True)
ann_model = MLPClassifier()

# 创建随机搜索对象并进行调参
grid_search_rf= RandomizedSearchCV(rf_model, param_dist_rf, cv=5, n_iter=27,random_state=42)   
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

best_svm = grid_search_svm.best_estimator_
coef = best_svm.coef_.ravel()
features_importance = np.abs(coef)
indices = np.argsort(features_importance)[::-1]

print("Top 20 important features (SVM):")
for f in range(min(20, len(indices))):
    feature_name = selected_feature_names[indices[f]]
    print("%s (%f)" % (feature_name, features_importance[indices[f]]))

print("\n")

best_rf = grid_search_rf.best_estimator_
importance = best_rf.feature_importances_
top_indices = np.argsort(importance)[::-1][:20]


print("Top 20 Feature Importance:")
for index in top_indices:
    feature_name = selected_feature_names[index]
    print(f"{feature_name}: {importance[index]}")
print("\n")
best_ann = grid_search_ann.best_estimator_
weights = best_ann.coefs_


feature_importance = np.abs(weights[0]).sum(axis=1)

top_features_index = np.argsort(feature_importance)[-20:][::-1]


print("Top 20 Important Features:")
for idx in top_features_index:
    feature_name = selected_feature_names[idx]
    print(f"{feature_name}: {feature_importance[idx]}")

    # print(f"Feature {idx + 1}: {feature_importance[idx]}")

print("\n")
```

```python
Top 20 important features (SVM):
qed (2.034821)
PEOE_VSA4 (1.393445)
BCUT2D_CHGHI (1.286525)
NumHAcceptors (1.211815)
EState_VSA9 (1.191918)
VSA_EState10 (1.138230)
SMR_VSA10 (1.129630)
VSA_EState8 (1.088787)
VSA_EState5 (0.947984)
MolLogP (0.927657)
SMR_VSA7 (0.878612)
BCUT2D_MRHI (0.813003)
HallKierAlpha (0.753999)
SlogP_VSA1 (0.728554)
MaxEStateIndex (0.707105)
MaxAbsEStateIndex (0.707105)
SMR_VSA9 (0.694168)
PEOE_VSA9 (0.651155)
PEOE_VSA2 (0.603743)
FpDensityMorgan1 (0.569659)


Top 20 Feature Importance:
MaxPartialCharge: 0.11690350094971952
MinAbsPartialCharge: 0.07589028402183211
MaxAbsEStateIndex: 0.06996427253241876
PEOE_VSA2: 0.059342916991622
MaxEStateIndex: 0.05506900584784368
SMR_VSA10: 0.0485890497980379
BCUT2D_MWLOW: 0.045704500481776475
SMR_VSA9: 0.03908561696559687
FpDensityMorgan1: 0.03805215638000522
EState_VSA3: 0.030494319541411782
PEOE_VSA3: 0.03035566579943683
HallKierAlpha: 0.028261561044375303
SMR_VSA7: 0.02659686381254809
VSA_EState1: 0.020701470425790732
qed: 0.020485822885238116
VSA_EState5: 0.020469303196951564
BCUT2D_LOGPHI: 0.01640820371617181
SMR_VSA6: 0.01578138277327718
NumHAcceptors: 0.0147188581425311
BCUT2D_CHGHI: 0.013425726234159567


Top 20 Important Features:
EState_VSA3: 13.880652116646576
qed: 12.751100837952091
PEOE_VSA2: 12.690568586430885
PEOE_VSA4: 12.652543055640583
HallKierAlpha: 12.440624152572468
SMR_VSA9: 12.42363941072588
MaxAbsEStateIndex: 12.368523859755218
PEOE_VSA3: 12.210422691241352
PEOE_VSA9: 12.081172504581302
MinEStateIndex: 12.036518296510211
SMR_VSA7: 11.86186272752981
EState_VSA2: 11.807125427203623
BCUT2D_CHGHI: 11.770331680408141
MaxAbsPartialCharge: 11.764227046575986
fr_Ndealkylation1: 11.712473625822009
SMR_VSA10: 11.699331223846512
VSA_EState8: 11.671878080540473
VSA_EState6: 11.633089463104303
MolLogP: 11.374083645367746
PEOE_VSA1: 11.29848206074489
```

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
X_predict=torch.tensor(all_features.values,dtype=torch.float32)
print(X_predict.shape)
```

    torch.Size([1358, 208])



```python
X_pre = X_predict[:, selected_features_index]
```


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_pre= scaler.fit_transform(X_train)
```


```python
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

```python

```


```python

```

