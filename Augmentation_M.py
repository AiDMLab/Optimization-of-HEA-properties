from featuresselect import *


def org_gen_ML_sorce(HorM, X_train, y_train, X_test, y_test, gen_model_name):

    r2_score = []

    trainset_r2, testset_r2 = train_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    r2_score.append([trainset_r2, testset_r2])

    for i in range(50, 301, 50):

        # 导入生成数据集
        dataset_gen = pd.read_csv(f'Augemented_dataset/{HorM}_{gen_model_name}_generated_data_{i}.csv')
        property_gen = dataset_gen.iloc[::,0]
        features_gen = dataset_gen.iloc[::,1:]

        # 合并数据集
        X_train_add = pd.concat([X_train, features_gen], axis=0)
        y_train_add = pd.concat([y_train, property_gen], axis=0)

        trainset_r2, testset_r2 = train_model(X_train=X_train_add, X_test=X_test, y_train=y_train_add, y_test=y_test)

        r2_score.append([trainset_r2, testset_r2])
    return r2_score

def follow(org_path, HorM, descrip_name):

    # 导入原始数据集-计算描述符-归一化
    property, craft, composition, descriptor = data_load(org_path, i=0, j=4)
    # 最优描述符组合
    descriptor = descriptor[descrip_name]
    features = pd.concat([craft, composition, descriptor], axis=1)

    # 划分数据集
    X_train, X_test, y_train, y_test = split_data(feature=features, property=property)

    r2_score_cgan = org_gen_ML_sorce(HorM, X_train, y_train, X_test, y_test, gen_model_name='cgan')
    r2_score_dhgcgan = org_gen_ML_sorce(HorM, X_train, y_train, X_test, y_test, gen_model_name='dhgcgan')

    return r2_score_gan, r2_score_cgan, r2_score_dhgcgan

#r2_score_gan, r2_score_cgan, r2_score_dhgcgan = follow(org_path='data_H.csv', HorM='H', descrip_name=['ave:num_f_valence','var:evaporation_heat','var:num_unfilled','var:vdw_radius_alvarez'])
r2_score_gan, r2_score_cgan, r2_score_dhgcgan = follow(org_path='../data_M.csv', HorM='M', descrip_name=['ave:thermal_conductivity', 'var:Polarizability', 'var:gs_energy'])
print('CGAN生成数据下的R2评分：')
print(r2_score_cgan)
print('DHGCGAN生成数据下的R2评分：')
print(r2_score_dhgcgan)