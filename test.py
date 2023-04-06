from sklearn.model_selection import KFold
import GPy 
from GPy.models import GPRegression
from scipy.stats import spearmanr, pearsonr
import numpy as np
from utils.utils_ import *
from plot import plot_CrossVal_avg

def evaluate(parameters):
    evaluation = branin_currin(torch.tensor([parameters.get("x1"), parameters.get("x2")]))
    # In our case, standard error is 0, since we are computing a synthetic function.
    # Set standard error to None if the noise level is unknown.
    return {"a": (evaluation[0].item(), 0.0), "b": (evaluation[1].item(), 0.0)}


def cross_train_validation(X_norm, y, Kfold, num_restarts, ker_lengthscale_upper, ker_var_upper):
    # when use K fold not use train_test split: 
    X_train = X_norm; y_train = y[:, -1]
    # 创建一个用于得到不同训练集和测试集样本的索引的StratifiedKFold实例，折数为5
    strtfdKFold = KFold(n_splits=Kfold)
    #把特征和标签传递给StratifiedKFold实例
    kfold = strtfdKFold.split(X_train, y_train)
    #循环迭代，（K-1）份用于训练，1份用于验证，把每次模型的性能记录下来。
    scores_train, scores_test = [], []
    uncer_train, uncer_test = [], []

    for k, (train, test) in enumerate(kfold):
        X_train_fold, y_train_fold = X_train[train], y_train[train]
        X_test_fold, y_test_fold = X_train[test], y_train[test]

        ker = GPy.kern.Matern52(input_dim = len(X_train_fold[0]), ARD =True)     #Matern52有啥讲究吗？
        ker.lengthscale.constrain_bounded(1e-2, ker_lengthscale_upper)         #超参数？（好像是posterior 得到的）
        ker.variance.constrain_bounded(1e-2, ker_var_upper)

        gpy_regr = GPRegression(X_train_fold, y_train_fold.reshape(-1,1), ker)#
        #gpy_regr.Gaussian_noise.variance = (0.01)**2       #这个一般需要怎么调整呢？（好像是posterior 得到的）
        #gpy_regr.Gaussian_noise.variance.fix()

        y_pred_train, y_uncer_train = gpy_regr.predict(X_train_fold)
        y_pred_test, y_uncer_test = gpy_regr.predict(X_test_fold)

        y_pred_train, y_uncer_train = y_pred_train[:,-1], y_uncer_train[:,-1]
        y_pred_test, y_uncer_test = y_pred_test[:,-1], y_uncer_test[:,-1]

        gpy_regr.randomize()
        gpy_regr.optimize_restarts(num_restarts=num_restarts, verbose=False, messages=False)
        
        score_train = pearsonr(y_train_fold, y_pred_train) [0]; scores_train.append(score_train)
        score_test = pearsonr(y_test_fold, y_pred_test) [0]; scores_test.append(score_test)
        # train_score = spearmanr(y_train_fold, y_pred_train) [0]
        uncer_train.append(y_uncer_train.mean())
        uncer_test.append(y_uncer_test.mean())

    # plot_CrossVal_avg(uncer_train, uncer_test)
    # plot_CrossVal_avg(scores_train, scores_test)
    A, B ,C, D = np.array(scores_train).mean(), np.array(scores_train).std(), np.array(scores_test).mean(), np.array(scores_test).std()
    printc.blue('\n\nTRAIN: Cross-Validation score: %.3f +/- %.3f' %(np.array(scores_train).mean(), 
                                                                np.array(scores_train).std()))
    printc.yellow('\n\nTEST: Cross-Validation score: %.3f +/- %.3f' %(np.array(scores_test).mean(), 
                                                                np.array(scores_test).std()))
    printc.blue('\n\nTRAIN: Cross-Validation uncertainty: %.3f +/- %.3f' %(np.array(uncer_train).mean(), 
                                                                np.array(uncer_train).std()))
    printc.yellow('\n\nTEST: Cross-Validation uncertainty: %.3f +/- %.3f' %(np.array(uncer_test).mean(),
                                                                np.array(uncer_test).std()))
    
    # plot_CrossVal_avg(A, B ,C, D)
    return A, B ,C ,D 
