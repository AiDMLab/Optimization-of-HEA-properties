import joblib
import numpy as np

def feature_importance_H(model_name, feature_name):

    model = joblib.load(model_name)

    feature_importances = model.feature_importances_

    feature_importances_with_names = list(zip(feature_name, feature_importances))

    feature_importances_sorted = sorted(feature_importances_with_names, key=lambda x: x[1], reverse=True)

    print("Hardness特征重要性排序：")
    for feature, importance in feature_importances_sorted:
        print(f'{feature}:{importance}')

def feature_importance_M(model_name, feature_name):

    model = joblib.load(model_name)

    feature_importances = {name: [] for name in feature_name}


    for estimator, estimator_features in zip(model.estimators_, model.estimators_features_):

        estimator_importances = estimator.feature_importances_

        for idx, feature_idx in enumerate(estimator_features):

            feature_importances[feature_name[feature_idx]].append(estimator_importances[idx])

    average_importances = {name: np.mean(importances) for name, importances in feature_importances.items()}
    average_importances_sorted = dict(sorted(average_importances.items(), key=lambda item: item[1], reverse=True))

    print("Modulus特征重要性排序：")
    for feature, importance in average_importances_sorted.items():
        print(f'{feature}:{importance}')


feature_importance_H(model_name='ML_model_H.pkl', feature_name=['pressure', 'bias', 'flow', 'Cu', 'Al', 'Fe', 'Zr',
                                                              'V', 'Co', 'Ni', 'Nb', 'Ti', 'Cr', 'Mo', 'Mn', 'W', 'Ta',
                                                              'Hf','ave:num_f_valence', 'var:evaporation_heat', 'var:num_unfilled',
                                                              'var:vdw_radius_alvarez'])
print()

feature_importance_M(model_name='ML_model_M.pkl', feature_name=['pressure', 'bias', 'flow', 'Cu', 'Al', 'Fe', 'Zr',
                                         'V', 'Co', 'Ni', 'Nb', 'Ti', 'Cr', 'Mo', 'Mn', 'W', 'Ta', 'Hf',
                                         'ave:thermal_conductivity', 'var:Polarizability', 'var:gs_energy'])