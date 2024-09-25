from featuresselect import *
import joblib

def train_model_H(X_train, X_test, y_train, y_test):

    model = ExtraTreesRegressor(n_estimators=22, max_depth=20,
                                    min_samples_split=2,
                                    min_samples_leaf=1,
                                    random_state=42)
    # 十倍交叉验证测试模型
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
    train_r2 = scores.mean()

    # 训练集和测试集预测
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)

    test_r2 = r2_score(y_test, test_pred)

    return train_r2, test_r2, model

def train_model_M(X_train, X_test, y_train, y_test):

    model = BaggingRegressor(n_estimators=13, max_samples=0.82, max_features=0.59, base_estimator="deprecated",random_state=42)

    # 十倍交叉验证测试模型
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
    train_r2 = scores.mean()

    # 训练集和测试集预测
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)

    test_r2 = r2_score(y_test, test_pred)

    return train_r2, test_r2, model

def follow(org_path, descrip_name, HorM, gen_model_name, sample):

    # 导入原始数据集-计算描述符-归一化
    property, craft, composition, descriptor = data_load(org_path, i=0, j=4)
    # 最优描述符组合
    descriptor = descriptor[descrip_name]
    features = pd.concat([craft, composition, descriptor], axis=1)

    # 划分数据集
    X_train, X_test, y_train, y_test = split_data(feature=features, property=property)

    # 导入生成数据集
    dataset_gen = pd.read_csv(f'Augemented_dataset_{HorM}/{HorM}_{gen_model_name}_generated_data_{sample}.csv')
    property_gen = dataset_gen.iloc[::, 0]
    features_gen = dataset_gen.iloc[::, 1:]

    # 合并数据集
    X_train_add = pd.concat([X_train, features_gen], axis=0)
    y_train_add = pd.concat([y_train, property_gen], axis=0)

    if HorM == 'H':
        train_r2, test_r2, model = train_model_H(X_train=X_train_add, X_test=X_test, y_train=y_train_add, y_test=y_test)
    if HorM == 'M':
        train_r2, test_r2, model = train_model_M(X_train=X_train_add, X_test=X_test, y_train=y_train_add, y_test=y_test)

    # 保存模型
    joblib.dump(model, filename=f'ML_model_{HorM}.pkl')

    # 构建csv文件，用于绘制预测散点图
    org_train_pred = model.predict(X_train)
    gen_train_pred = model.predict(features_gen)
    test_pred = model.predict(X_test)

    org_results = pd.DataFrame({'org_train_truelabel':y_train, 'org_train_pred':org_train_pred})
    gen_results = pd.DataFrame({'gen_train_truelabels':property_gen, 'gen_train_pred':gen_train_pred})
    test_results = pd.DataFrame({'test_truelabels':y_test, 'test_pred':test_pred})
    results = pd.concat([org_results, gen_results, test_results], axis=1)
    results.to_csv(f'prediction_results_{HorM}.csv', index=False)
    return train_r2, test_r2

train_r2, test_r2 = follow(org_path='data_H.csv', descrip_name=['ave:num_f_valence','var:evaporation_heat','var:num_unfilled','var:vdw_radius_alvarez'],
       HorM='H', gen_model_name='dhgcgan', sample=250)

train_r2, test_r2 = follow(org_path='data_M.csv', descrip_name=['ave:thermal_conductivity', 'var:Polarizability', 'var:gs_energy'],
       HorM='M', gen_model_name='dhgcgan', sample=50)