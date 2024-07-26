from scipy import stats
import pandas as pd



# pd.DataFrame.dtypes  # 返回对象类型的 Series
# astype 将类型转成 string 格式

def feature_explore(df):
    # 数据类型
    features_des = pd.DataFrame(df.dtypes.astype(str),columns = ['原始数据类型'])
    type_mapping = {'int64':'数值类型','float64':'数值类型','object':'枚举值类型'} # 其中object为枚举值类型，float和int为数值类型
    features_des['原始数据类型'] = features_des['原始数据类型'].apply(lambda x: type_mapping[x] if x in type_mapping else '其它类型')
    features_des['修改后数据类型'] = ''
    # 计算各个特征的非空值数量
    not_null = pd.DataFrame(df.notna().sum(),columns = ['非空个数'])
    # 更新特征描述表
    features_des = pd.merge(left = features_des,right = not_null, left_index = True, right_index = True, how = 'left')
    # 计算覆盖率
    # shape属性返回一个包含行数和列数的元组，因此df.shape[0]就是获取DataFrame的行数，而df.shape[1]则会返回列数
    features_des['填充率'] =  features_des['非空个数'].apply(lambda x: str(round(x/df.shape[0]*100,2))+'%')
    # 空值填充
    features_des['空值处理方法'] = ''
    features_des['填充值'] = ''

    # 使用pandas自带的describe函数自动计算数值类型变量的统计值如均值，中位数及各分位数，并且更新到特征描述表中
    # .describe()：然后调用describe()方法来生成描述性统计信息，包括计数、平均值、标准差、最小值、四分位数和最大值。
    # 但是，由于数据已经被转换为字符串，所以describe()方法对于字符串数据的统计意义可能有限，通常只返回计数、唯一值、最频繁出现的值及其频率。
    describe_str = df.astype(str).describe().T
    describe_str = describe_str.drop(columns = ['count'])
    describe_str.columns = ['枚举值数量','出现次数最多的枚举值','出现次数']
    features_des = pd.merge(left = features_des,right = describe_str, left_index = True, right_index = True, how = 'left')

    # 统计计数、平均值、标准差、最小值、四分位数和最大值。
    describe = df.describe(percentiles=[.25,.5,.75,.90,.95,.99]).T
    describe = describe.drop(columns = ['count'])
    describe.columns = ['均值','标准差','最小值','25%分位数','50%分位数',
                        '75%分位数','90%分位数','95%分位数','99%分位数','最大值']
    features_des = pd.merge(left = features_des,right = describe, left_index = True, right_index = True, how = 'left')

    # 极值处理
    features_des['上边界-3σ'] = features_des['均值'] + 3 * features_des['标准差']
    features_des['下边界-3σ'] = features_des['均值'] - 3 * features_des['标准差']
    features_des['上边界-iqr'] = features_des['75%分位数'] + 1.5 * (features_des['75%分位数']-features_des['25%分位数'])
    features_des['下边界-iqr'] = features_des['75%分位数'] - 1.5 * (features_des['75%分位数']-features_des['25%分位数'])
    # 正态分布统计
    norm_feature = features_des[features_des['原始数据类型'] == '数值类型'].index  #提取数值类型 字段名称
    for feature in norm_feature:
        # 对每一列中的 非空值, 计算正态分布p-value
        statistic, p_value = stats.normaltest(df[df[feature].notna()][feature])
        # 报存 p-value值
        features_des.loc[feature,'正态分布p-value'] = p_value

    features_des['极大值处理方法'] = ''
    features_des['极大值替换值'] = ''
    features_des['极小值处理方法'] = ''
    features_des['极小值替换值'] = ''
    features_des['是否删除该特征'] = ''
    features_des['删除该特征原因'] = ''



    # 筛选出连续数值型以及枚举值类型特征
    num_features = []
    object_features = []
    for feature in df.columns:
        if features_des.loc[feature,'原始数据类型'] == '数值类型':
            num_features.append(feature)
        elif features_des.loc[feature,'原始数据类型'] == '枚举值类型':
             object_features.append(feature)                 
    num_features.remove('label') # 将label移除

    # 对于连续数值型特征绘制概率密度图
    import math
    import matplotlib.pyplot as plt
    import seaborn as sns 
    n_cols = 3
    n_rows = math.ceil(len(num_features)/n_cols)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*3.5))
    ax = ax.flatten()
    plt.rc('font', family='WenQuanYi Micro Hei')
    for i, column in enumerate(num_features):
        plot_axes = [ax[i]]
        #确保seaborn版本号为0.11，大于此版本则会报错
        sns.kdeplot(df[column],hue=df['label'],ax=ax[i],common_norm=False)
        ax[i].set_title(f'{column} Distribution');
        ax[i].set_xlabel(None)   
    for i in range(i+1, len(ax)):
        ax[i].axis('off')
    kde = plt.tight_layout()

    # 对于枚举值绘制直方图
    import math
    n_cols = 3
    n_rows = math.ceil(len(object_features)/n_cols)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*3.5))
    ax = ax.flatten()
    for i, column in enumerate(object_features):
        plot_axes = [ax[i]]
        hist, bins, _ = ax[i].hist(df[column].fillna('其它'))
        ax[i].set_title(f'{column} hist');
        ax[i].set_xlabel(None)
        ax[i].tick_params(axis = 'x', labelrotation = 90)
        for j in range(len(hist)):
            if hist[j] == 0:
                continue
            ax[i].text(bins[j] + (bins[j+1] - bins[j]) / 2, hist[j], str(int(hist[j])), ha='center', va='bottom')
    for i in range(i+1, len(ax)):
        ax[i].axis('off')
    plt.tight_layout()



    # 绘制箱线图
    n_cols = 3
    n_rows = math.ceil(len(num_features)/n_cols)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(18, n_rows*3.5))
    ax = ax.flatten()
    for i,feature in enumerate(num_features):
        sns.boxplot(y = df[feature].astype(float),ax = ax[i])
    for i in range(i+1, len(ax)):
        ax[i].axis('off')
    plt.tight_layout()
    ###根据判上述数据修改特征处理方法并更新在feature_des.xlsx表格中
    features_des.sort_values('原始数据类型').to_excel('data/feature_des.xlsx')

    