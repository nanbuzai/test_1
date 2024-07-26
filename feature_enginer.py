import pandas as pd

def feature_enginer(df):
    normal_features = list(df.select_dtypes(include='number').columns)
    normal_features.remove('label')
    onehot_features = list(df.select_dtypes(exclude='number').columns)
    # 归一化
    from sklearn.preprocessing import MinMaxScaler
    ms = MinMaxScaler()
    ms.fit(df[normal_features])
    X_ms = ms.transform(df[normal_features])
    df = pd.concat([df.reset_index(drop=True),pd.DataFrame(X_ms,columns = list(map(lambda x: x+'_norm',normal_features)))],axis = 1)


    # onehot 
    
    for f in onehot_features:
        df = pd.concat([df,pd.get_dummies(df[f],prefix=f)],axis=1)

    # 选出特征
    select_features  = list(set(df.columns) - set(onehot_features)-set(normal_features))
    
    # check again
    check_feature = pd.DataFrame(df[select_features].dtypes.astype(str), columns=['数据类型'])
    type_mapping = {'int64':'数值类型','float64':'数值类型','object':'枚举值类型','uint8':'数值类型'} # 其中object为枚举值类型，float和int为数值类型
    check_feature['数据类型'] = check_feature['数据类型'].apply(lambda x: type_mapping[x] if str(x) in type_mapping else '其它类型')
    check_feature = pd.merge(left = check_feature,right = df[select_features].describe(include = 'all',percentiles=[.5]).T[['count','min','max','50%']], left_index = True, right_index = True, how = 'left')
    check_feature['填充率'] = check_feature['count'].apply(lambda x: str(int(x)/df.shape[0]*100)+'%')
    check_feature.columns = ['数据类型','非空个数','最小值','最大值','50%分位数','填充率']
    check_feature = check_feature[['数据类型','非空个数','填充率','最小值','50%分位数','最大值']]
   
    return df[select_features],check_feature