# Optimization-of-HEA-properties

Background:
Although the multi-objective optimization algorithm based on machine learning (ML) prediction model has been applied to the collaborative optimization of HEA (high entropy alloy) properties, the problem of small number and imbalance of HEA coating samples that can be used for modeling still exists. To address this problem, improve the accuracy of ML prediction models and reveal the implicit relationship between process-composition-properties, we design a machine learning strategy driven collaborative optimization algorithm for hardness and toughness of high entropy alloy coatings. A Deheterogeneous conditional Generative Adversarial network (DGH-CGAN) considering the influence of composition and process differences was proposed to improve the prediction ability of the ML model. On this basis, a multi-objective optimization algorithm is used to optimize the hardness and elastic modulus of HEA coating to cope with the problem of performance antagonism.


Optimize the process:
Feature selection →GAN model training → Data augmentation → determining the amount of augmented data and prediction model → Prediction model hyperparameter optimization → Training and saving prediction model → multi-objective optimization → Interpretability analysis.


Note:
1. We provide all the code involved in the feature selection, data augmentation, model building, multi-objective optimization, and interpretability analysis processes, as well as the data augmentation and performance prediction models trained on our dataset. 2. Before running this code, you need to have your own data set, which will be used in place of 'data_H.csv' and 'data_M.csv' in the program.
2. In the project, '_H' and '_M' represent the results for hardness and modulus respectively.
