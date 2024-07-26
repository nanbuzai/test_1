import pandas as pd


def feature_processing(df):
    # 根据特征表修改数据
    features_des = pd.read_excel('data/feature_des_fill.xlsx',index_col=0)
    df = pd.read_csv('data/data.csv')



    # 修改类型
    for feature in features_des[features_des['修改后数据类型'].notna()].index:
        df[feature] = df[feature].astype(str)

    # 删除特征
    df = df.drop(columns = features_des[features_des['是否删除该特征'] == '是'].index.to_list())

    # 填充空值
    fillna_features = features_des[features_des['填充值'].notna()].index
    for feature in fillna_features:
        df[feature] = df[feature].fillna(features_des.loc[feature,'填充值'])

    # 删除空值
    dropna_features = features_des[features_des['空值处理方法']=='删除'].index
    for feature in dropna_features:
        df = df.drop(df[df[feature].isna()].index)

    # 替换极值
    # 上边界替换
    replace_max_features = features_des[features_des['极大值处理方法'] == '替换'].index
    for feature in replace_max_features:
        df[feature] = df[feature].astype(float)
        max_value = float(features_des.loc[feature,'极大值替换值'])
        df.loc[df[feature] > max_value,feature] = max_value
    # 下边界替换
    replace_min_features = features_des[features_des['极小值处理方法'] == '替换'].index
    for feature in replace_min_features:
        df[feature] = df[feature].astype(float)
        min_value = float(features_des.loc[feature,'极小值替换值'])
        df.loc[df[feature] < min_value,feature] = min_value

    # 上边界删除
    drop_max_features = features_des[features_des['极大值处理方法'] == '删除'].index
    for feature in drop_max_features:
        df[feature] = df[feature].astype(float)
        max_value = float(features_des.loc[feature,'极大值替换值'])
        df = df.drop(df[df[feature] > max_value].index)

    # 下边界删除
    drop_min_features = features_des[features_des['极小值处理方法'] == '删除'].index
    for feature in drop_min_features:
        df[feature] = df[feature].astype(float)
        min_value = float(features_des.loc[feature,'极小值替换值'])
        df = df.drop(df[df[feature] < min_value].index)


    # 负值替换
    negative_replace_features = features_des[features_des['负值处理方式'] == '替换'].index
    for feature in negative_replace_features:
        df[feature] = df[feature].astype(float)
        df.loc[df[feature] < 0 ,feature] = float(features_des.loc[feature,'负值填充值'])

    # 负值删除
    negative_remove_features = features_des[features_des['负值处理方式'] == '删除'].index
    for feature in negative_remove_features:
        df[feature] = df[feature].astype(float)
        df = df.drop(df.loc[df[feature] < 0].index)

    # check again
    check_feature = pd.DataFrame(df.dtypes.astype(str), columns=['数据类型'])
    type_mapping = {'int64':'数值类型','float64':'数值类型','object':'枚举值类型'} # 其中object为枚举值类型，float和int为数值类型
    check_feature['数据类型'] = check_feature['数据类型'].apply(lambda x: type_mapping[x] if str(x) in type_mapping else '其它类型')
    check_feature = pd.merge(left = check_feature,right = df.describe(include = 'all',percentiles=[]).T[['count','min','max']], left_index = True, right_index = True, how = 'left')
    check_feature['填充率'] = check_feature['count'].apply(lambda x: str(int(x)/df.shape[0]*100)+'%')
    check_feature.columns = ['数据类型','非空个数','最小值','最大值','填充率']
    check_feature = check_feature[['数据类型','非空个数','填充率','最小值','最大值']]
    return df, check_feature