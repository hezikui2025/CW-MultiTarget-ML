import os
import tempfile
import numpy as np
from save_load import load_data, save_data
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
from sklearn.metrics import r2_score, mean_squared_error

# 设置临时文件夹解决中文路径问题
temp_dir = r'C:\temp_joblib'
os.environ['JOBLIB_TEMP_FOLDER'] = temp_dir
os.makedirs(temp_dir, exist_ok=True)

# 加载数据
important_features = ['表面积', '厚度/cm', 'Inf.COD', 'Inf.NO3--N', 'Inf.NH4+-N', 'HRT/h', '曝气', '主要基质类型',
                      '植物']
data = load_data('../output/preprocessed_data.joblib')
X_train, y_train = data['X_train'], data['y_train']

# 1. 定义预处理（仅重要特征）

important_numeric = [f for f in important_features if
                     f in ['表面积', '厚度/cm', 'Inf.COD', 'Inf.NO3--N', 'Inf.NH4+-N', 'HRT/h', '曝气']]
important_categorical = [f for f in important_features if f in ['主要基质类型', '植物']]

# 修改点：使用passthrough直接传递已编码的分类变量
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), important_numeric),
        ('cat', 'passthrough', important_categorical)
    ])

# 2. 定义贝叶斯搜索空间
bayes_search_spaces = {
    'model__estimator__n_estimators': Integer(50, 150),
    'model__estimator__learning_rate': Real(0.01, 0.04, prior='log-uniform'),
    'model__estimator__max_depth': Integer(3, 7),
    'model__estimator__min_samples_split': Integer(2, 15),
    'model__estimator__min_samples_leaf': Integer(2, 12),
    'model__estimator__subsample': Real(0.7, 0.8),
    'model__estimator__validation_fraction': [0.1],
    'model__estimator__n_iter_no_change': [5],
    'model__estimator__tol': [1e-4]
}

# 预先创建KFold对象，确保数据分割一致
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n{'=' * 60}")
print(f"开始训练多目标GBDT模型...")
print(f"{'=' * 60}")

# 创建多目标模型
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', MultiOutputRegressor(
        GradientBoostingRegressor(
            random_state=42,
            # 早停相关参数
            validation_fraction=0.1,
            n_iter_no_change=5,
            tol=1e-4
        )
    ))
])

# 使用贝叶斯优化
bayes_search = BayesSearchCV(
    pipeline,
    bayes_search_spaces,  # 使用统一的贝叶斯搜索空间
    cv=kfold,  # 使用预先创建的KFold对象
    scoring='neg_mean_squared_error',
    n_iter=50,
    n_jobs=-1,
    verbose=1,
    random_state=42,
    return_train_score=True
)

# 训练模型
bayes_search.fit(X_train[important_features], y_train)

# 输出最优参数组合的交叉验证结果
print(f"\n>>> 多目标GBDT模型训练完成 <<<")
print(f"最佳参数：{bayes_search.best_params_}")
print(f"最佳分数（负MSE）：{bayes_search.best_score_:.6f}")
print(f"最佳分数对应MSE：{-bayes_search.best_score_:.6f}")

# 获取最优参数组合的交叉验证详细结果
best_index = bayes_search.best_index_
cv_results = bayes_search.cv_results_

print(f"\n最优参数组合的5折交叉验证详细结果：")
print(f"参数组合排名：第 {cv_results['rank_test_score'][best_index]} 名")

# 重新使用相同的KFold分割来计算各目标各折的R²和MSE
print(f"\n计算各目标各折R²和MSE分数（使用相同的数据分割）...")

best_estimator = bayes_search.best_estimator_
targets = y_train.columns

# 为每个目标创建存储结果的字典
r2_scores_per_target = {target: [] for target in targets}
mse_scores_per_target = {target: [] for target in targets}

# 重新使用相同的KFold分割
fold_indices = list(kfold.split(X_train[important_features]))

for fold, (train_idx, test_idx) in enumerate(fold_indices):
    print(f"\n--- 第 {fold + 1} 折 ---")

    # 获取该折的测试数据
    X_fold_test = X_train[important_features].iloc[test_idx]
    y_fold_test = y_train.iloc[test_idx]

    # 使用最优模型进行预测
    y_pred = best_estimator.predict(X_fold_test)

    # 计算每个目标的分数
    for i, target in enumerate(targets):
        # 计算该折该目标的R²
        fold_r2 = r2_score(y_fold_test[target], y_pred[:, i])
        r2_scores_per_target[target].append(fold_r2)

        # 计算该折该目标的MSE
        fold_mse = mean_squared_error(y_fold_test[target], y_pred[:, i])
        mse_scores_per_target[target].append(fold_mse)

        print(f"{target} - MSE: {fold_mse:.6f}, R²: {fold_r2:.6f}")

# 计算整体R²和MSE
print(f"\n{'=' * 60}")
print(f"整体性能评估（在整个训练集上）")
print(f"{'=' * 60}")

y_pred_all = best_estimator.predict(X_train[important_features])

for i, target in enumerate(targets):
    # 整体R²
    overall_r2 = r2_score(y_train[target], y_pred_all[:, i])

    # 整体MSE
    overall_mse = mean_squared_error(y_train[target], y_pred_all[:, i])

    # 各折平均R²和标准差
    mean_r2 = np.mean(r2_scores_per_target[target])
    std_r2 = np.std(r2_scores_per_target[target])

    # 各折平均MSE和标准差
    mean_mse = np.mean(mse_scores_per_target[target])
    std_mse = np.std(mse_scores_per_target[target])

    print(f"\n{target}:")
    print(f"  整体 - MSE: {overall_mse:.6f}, R²: {overall_r2:.6f}")
    print(f"  各折平均 - MSE: {mean_mse:.6f} ± {std_mse:.6f}, R²: {mean_r2:.6f} ± {std_r2:.6f}")

# 获取贝叶斯优化的平均MSE和标准差
print(f"\n贝叶斯优化交叉验证结果：")
print(f"平均MSE: {-cv_results['mean_test_score'][best_index]:.6f}")
print(f"标准差MSE: {cv_results['std_test_score'][best_index]:.6f}")

# 获取最佳模型并保存
best_model = bayes_search.best_estimator_
os.makedirs('output/models', exist_ok=True)
save_data(best_model, 'output/models/gbdt_model_multi_output.joblib')

print(f"\n模型已保存到：output/models/gbdt_model_multi_output.joblib")