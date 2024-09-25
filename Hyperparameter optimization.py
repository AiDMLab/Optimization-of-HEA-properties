import random
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor,AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor, \
    BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from deap import base, creator, tools, algorithms
import pandas as pd
from xgboost import XGBRegressor
from featuresselect import *



def nsga_H(X_train, X_test, y_train, y_test):
    def evaluate(params):
        n_estimators, max_depth, min_samples_split, min_samples_leaf = params[0]
        if abs(n_estimators) <= 1:
            n_estimators = 20
        if n_estimators >= 200:
            n_estimators = 200
        if abs(max_depth) == 0:
            max_depth = 3
        if abs(max_depth) >= 20:
            max_depth = 20
        if abs(int(min_samples_split)) <= 1:
            min_samples_split = 2
        if abs(min_samples_split) >= 20:
            min_samples_split = 20
        if abs(int(min_samples_leaf)) == 0:
            min_samples_leaf = 1
        if abs(min_samples_leaf) >= 10:
            min_samples_leaf = 10

        # 使用给定的超参数配置构建模型
        model = ExtraTreesRegressor(n_estimators=abs(int(n_estimators)), max_depth=abs(int(max_depth)),
                                    min_samples_split=abs(int(min_samples_split)),
                                    min_samples_leaf=abs(int(min_samples_leaf)),
                                    random_state=42)
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
        train_r2 = scores.mean()
        # 训练模型
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        test_r2 = r2_score(y_test, test_preds)
        # 返回训练集和测试集上的R^2得分作为多目标优化的目标函数
        return train_r2, test_r2

    def attr_int_with_ranges():
        return np.random.randint(20, 200), np.random.randint(10, 25), np.random.randint(2, 20), np.random.randint(1, 10)

    #创建DEAP遗传算法所需的适应度函数和个体
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))  # 最大化训练集和测试集上的R^2得分
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", attr_int_with_ranges)# 使用新的函数来生成四个整数值
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 注册评价函数
    toolbox.register("evaluate", evaluate)

    # 定义遗传算法操作
    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 交叉操作
    toolbox.register("mutate", tools.mutUniformInt, low=[20, 10, 2, 1],
                 up=[200, 25, 20, 10], indpb=0.2)  # 变异操作，使用参数的最大范围
    toolbox.register("select", tools.selNSGA2)  # 选择操作

    # 创建初始种群
    population = toolbox.population(n=100)

    # 运行遗传算法进行优化
    algorithms.eaMuPlusLambda(population, toolbox, mu=50, lambda_=100, cxpb=0.7, mutpb=0.3, ngen=50, verbose=True)

    # 输出最优个体
    best_individuals = tools.selBest(population, k=100)
    for ind in best_individuals:
        print("Best Individual:", ind)
        print("Best Individual Fitness (Train R^2, Test R^2):", ind.fitness.values)


def nsga_M(X_train, X_test, y_train, y_test):
    def evaluate(params):
        n_estimators, max_sample, max_feature = params[0]
        if abs(n_estimators) <= 1:
            n_estimators = 10
        if n_estimators >= 100:
            n_estimators = 100
        if abs(max_sample) <= 0.5:
            max_sample = 0.5
        if abs(max_sample) > 1:
            max_sample = 1
        if abs(max_feature) == 0:
            max_feature = 0.1
        if abs(max_feature) > 1:
            max_feature = 1

        # 使用给定的超参数配置构建模型
        model = BaggingRegressor(n_estimators=abs(int(n_estimators)), max_samples=abs(max_sample),
                                max_features=abs(max_feature), base_estimator="deprecated", random_state=42)
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
        train_r2 = scores.mean()
        # 训练模型
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        test_r2 = r2_score(y_test, test_preds)
        # 返回训练集和测试集上的R^2得分作为多目标优化的目标函数
        return train_r2, test_r2

    def attr_int_with_ranges():
        return np.random.randint(10, 100), np.random.uniform(0.5, 1), np.random.uniform(0.1, 1)


    # 创建DEAP遗传算法所需的适应度函数和个体
    creator.create("FitnessMulti", base.Fitness, weights=(0.9, 0.1))  # 最大化训练集和测试集上的R^2得分
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", attr_int_with_ranges)  # 使用新的函数来生成四个整数值
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 注册评价函数
    toolbox.register("evaluate", evaluate)

    # 定义遗传算法操作
    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 交叉操作
    toolbox.register("mutate", tools.mutUniformInt, low=[10, 0.5, 0.1],
                     up=[100, 1.0, 1.0], indpb=0.2)  # 变异操作，使用参数的最大范围
    toolbox.register("select", tools.selNSGA2)  # 选择操作

    # 创建初始种群
    population = toolbox.population(n=100)

    # 运行遗传算法进行优化
    algorithms.eaMuPlusLambda(population, toolbox, mu=50, lambda_=100, cxpb=0.7, mutpb=0.3, ngen=500, verbose=True)

    # 输出最优个体
    best_individuals = tools.selBest(population, k=100)
    for ind in best_individuals:
        print("Best Individual:", ind)
        print("Best Individual Fitness (Train R^2, Test R^2):", ind.fitness.values)

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
        nsga_H(X_train=X_train_add, X_test=X_test, y_train=y_train_add, y_test=y_test)
    if HorM == 'M':
        nsga_M(X_train=X_train_add, X_test=X_test, y_train=y_train_add, y_test=y_test)


follow(org_path='data_H.csv', descrip_name=['ave:num_f_valence','var:evaporation_heat','var:num_unfilled','var:vdw_radius_alvarez'],
       HorM='H', gen_model_name='dhgcgan', sample=250)

follow(org_path='data_M.csv', descrip_name=['ave:thermal_conductivity', 'var:Polarizability', 'var:gs_energy'],
       HorM='M', gen_model_name='dhgcgan', sample=50)