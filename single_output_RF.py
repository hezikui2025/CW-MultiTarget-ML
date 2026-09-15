import os
import tempfile
import numpy as np
from save_load import load_data, save_data
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from sklearn.metrics import r2_score

temp_dir = r'C:\temp_joblib'
os.environ['JOBLIB_TEMP_FOLDER'] = temp_dir
os.makedirs(temp_dir, exist_ok=True)

# 加载数据
important_features = ['表面积', '厚度/cm', 'Inf.COD', 'Inf.NO3--N', 'Inf.NH4+-N', 'HRT/h', '曝气', '主要基质类型',
                      '植物']
data = load_data('../output/preprocessed_data.joblib')
X_train, y_train = data['X_train'], data['y_train']

# 1. 定义预处理
# 手动指定分类变量和数值变量
important_numeric = [f for f in important_features if
                     f in ['表面积', '厚度/cm', 'Inf.COD', 'Inf.NO3--N', 'Inf.NH4+-N', 'HRT/h', '曝气']]
important_categorical = [f for f in important_features if f in ['主要基质类型', '植物']]

# 使用passthrough直接传递已编码的分类变量
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), important_numeric),
        ('cat', 'passthrough', important_categorical)
    ])

# 2. 训练模型（随机森林）
os.makedirs('output/models', exist_ok=True)

# 定义每个目标的贝叶斯搜索空间
bayes_search_spaces = {
    'Eff.COD': {
        'model__n_estimators': Integer(50, 200),
        'model__max_depth': Integer(3, 7),
        'model__min_samples_split': Integer(3, 6),
        'model__min_samples_leaf': Integer(3, 7),
    },
    'Eff.NO3--N': {
        'model__n_estimators': Integer(50, 200),
        'model__max_depth': Integer(3, 7),
        'model__min_samples_split': Integer(3, 6),
        'model__min_samples_leaf': Integer(1, 4),
    },
    'Eff.NH4+-N': {
        'model__n_estimators': Integer(50, 200),
        'model__max_depth': Integer(3, 7),
        'model__min_samples_split': Integer(4, 8),
        'model__min_samples_leaf': Integer(4, 8),
    }
}

# 预先创建KFold对象，确保数据分割一致
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for target in y_train.columns:
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ])

    # 使用针对每个目标的贝叶斯搜索空间
    search_space = bayes_search_spaces.get(target, {
        'model__n_estimators': Integer(50, 200),
        'model__max_depth': Integer(3, 7),
        'model__min_samples_split': Integer(2, 10),
        'model__min_samples_leaf': Integer(1, 4),
        'model__max_features': Categorical(['sqrt', 0.7, None])
    })

    bayes_search = BayesSearchCV(
        pipeline,
        search_space,
        cv=kfold,  # 使用预先创建的KFold对象
        scoring='neg_mean_squared_error',  # 使用MSE作为优化指标
        n_iter=50,  # 迭代次数，可以根据需要调整
        n_jobs=-1,
        verbose=1,
        random_state=42,
        return_train_score=True  # 确保返回训练分数
    )

    print(f"\n{'=' * 60}")
    print(f"开始训练 {target} 模型...")
    print(f"{'=' * 60}")

    bayes_search.fit(X_train[important_features], y_train[target])

    # 输出最优参数组合的交叉验证结果
    print(f"\n>>> {target} 随机森林模型训练完成 <<<")
    print(f"最佳参数：{bayes_search.best_params_}")
    print(f"最佳分数（负MSE）：{bayes_search.best_score_:.6f}")
    print(f"最佳分数对应MSE：{-bayes_search.best_score_:.6f}")

    # 获取最优参数组合的交叉验证详细结果
    best_index = bayes_search.best_index_
    cv_results = bayes_search.cv_results_

    print(f"\n最优参数组合的5折交叉验证详细结果：")
    print(f"参数组合排名：第 {cv_results['rank_test_score'][best_index]} 名")

    # 重新使用相同的KFold分割来计算R²
    print(f"\n计算各折R²分数（使用相同的数据分割）...")
    r2_scores = []

    best_estimator = bayes_search.best_estimator_

    # 重新使用相同的KFold分割
    fold_indices = list(kfold.split(X_train[important_features]))

    for fold, (train_idx, test_idx) in enumerate(fold_indices):
        # 获取该折的测试数据
        X_fold_test = X_train[important_features].iloc[test_idx]
        y_fold_test = y_train[target].iloc[test_idx]

        # 使用最优模型进行预测
        y_pred = best_estimator.predict(X_fold_test)

        # 计算该折的R²
        fold_r2 = r2_score(y_fold_test, y_pred)
        r2_scores.append(fold_r2)

        # 获取该折的MSE分数（来自贝叶斯优化的结果）
        fold_mse_key = f'split{fold}_test_score'
        fold_mse_score = cv_results[fold_mse_key][best_index]

        print(f"第 {fold + 1} 折 - MSE: {-fold_mse_score:.6f}, R²: {fold_r2:.6f}")

    # 计算整体R²
    y_pred_all = best_estimator.predict(X_train[important_features])
    overall_r2 = r2_score(y_train[target], y_pred_all)

    print(f"\n平均分数 - MSE: {-cv_results['mean_test_score'][best_index]:.6f}, R²: {overall_r2:.6f}")
    print(f"各折R²平均: {np.mean(r2_scores):.6f} ± {np.std(r2_scores):.6f}")
    print(f"标准差 - MSE: {cv_results['std_test_score'][best_index]:.6f}")

    save_data(bayes_search.best_estimator_, f'output/models/rf_model_{target}.joblib')
    print(f"模型已保存到：output/models/rf_model_{target}.joblib")