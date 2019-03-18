# -*- coding:utf-8 -*-
import time
import lightgbm as lgb


train_data = lgb.Dataset(Xtrain, label=ytrain)
test_data = lgb.Dataset(Xtest, label=ytest)

# specify parameters via map
# reference http://lightgbm.apachecn.org/#/docs/2
params = {
    'num_leaves':31,                # Same to max_leaf_nodes in GBDT, but GBDT's default value is None
    'max_depth': -1,                # Same to max_depth of xgboost
    'tree_learner': 'serial',
    'application':'multiclass',     # Same to objective of xgboost
    'num_class':10,                 # Same to num_class of xgboost
    'learning_rate': 0.1,           # Same to eta of xgboost
    'min_split_gain': 0,            # Same to gamma of xgboost
    'lambda_l1': 0,                 # Same to alpha of xgboost
    'lambda_l2': 0,                 # Same to lambda of xgboost
    'min_data_in_leaf': 20,         # Same to min_samples_leaf of GBDT
    'bagging_fraction': 1.0,        # Same to subsample of xgboost
    'bagging_freq': 0,
    'bagging_seed': 0,
    'feature_fraction': 1.0,         # Same to colsample_bytree of xgboost
    'feature_fraction_seed': 2,
    'min_sum_hessian_in_leaf': 1e-3, # Same to min_child_weight of xgboost
    'num_threads': 1
}
num_round = 10

# start training
start_time = time.time()
bst = lgb.train(params, train_data, num_round)
end_time = time.time()
print('The training time = {}'.format(end_time - start_time))

# get prediction and evaluate
ypred_onehot = bst.predict(Xtest)
ypred = []
for i in range(len(ypred_onehot)):
    ypred.append(ypred_onehot[i].argmax())

accuracy = np.sum(ypred == ytest) / len(ypred)
print('Test accuracy = {}'.format(accuracy))