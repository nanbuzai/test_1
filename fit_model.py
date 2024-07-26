from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import pandas as pd
def train(df, params):
    params['objective'] = 'binary'
    x_train, x_test , y_train, y_test =  train_test_split(df.drop(columns = ['label']),df['label'],random_state=42)
    lgb =  LGBMClassifier(**params)
    model = lgb.fit(x_train,y_train)
    # 验证模型
    import numpy as np
    y_true = y_test
    y_prob = model.predict_proba(x_test)[:,1]
    result = pd.DataFrame(y_true)
    result['prob'] = y_prob
    final_result = []
    for t in range(99,-1,-1):
        if t % 10 != 0 and t not in (99,98,97,96,95,94,93,92,91,90,0):

            continue
        a = np.percentile(y_prob,t) # 计算阈值
        result['predict'] = result['prob'].apply(lambda x: 1 if x >= a else 0)
        count = result[result['predict'] == 1].shape[0] # 输出人数
        count_true = result[(result['predict'] == 1 ) & (result['label'] == 1)].shape[0] # 输出中再购人数
        recall = count_true/result[result['label']==1].shape[0] # 输出中再购率
        accuracy = count_true/count # 输出中再购率
        up_rate = accuracy/(result[result['label']==1].shape[0]/result.shape[0]) # 输出提升率
        final_result.append(pd.DataFrame([[100-t,count,count_true,result[result['label']==1].shape[0],
                                        round(recall*100,2),a,round(accuracy,2),up_rate]],
                                        columns=['top','输出人数','输出中再购人数','总再购人数','召回率',
                                                    '得分','准确率','提升率'])) 
    return pd.concat(final_result),model    

# 测试