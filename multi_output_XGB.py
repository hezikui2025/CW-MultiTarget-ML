import os
import tempfile
import numpy as np
import pandas as pd
from save_load import load_data, save_data
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical, Real


temp_dir = r'C:\temp_joblib'
os.environ['JOBLIB_TEMP_FOLDER'] = temp_dir
os.makedirs(temp_dir, exist_ok=True)

# 加载数据
important_features = ['表面积', '厚度/cm', 'Inf.COD', 'Inf.NO3--N', 'Inf.NH4+-N', 'HRT/h', '曝气', '主要基质类型',
                      '植物']
data = load_data('../output/preprocessed_data.joblib')
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

# 1. 定义预处理
important_numeric = [f for f in important_features if
                     f in ['表面积', '厚度/cm', 'Inf.COD', 'Inf.NO3--N', 'Inf.NH4+-N', 'HRT/h', '曝气']]
important_categorical = [f for f in important_features if f in ['主要基质类型', '植物']]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), important_numeric),
        ('cat', 'passthrough', important_categorical)
    ])

# 2. 定义贝叶斯搜索空间
bayes_search_spaces = {
    'model__estimator__n_estimators': Integer(50, 150),
    'model__estimator__learning_rate': Real(0.005, 0.04, prior='log-uniform'),
    'model__estimator__max_depth': Integer(3, 6),
    'model__estimator__min_child_weight': Integer(4, 15),
    'model__estimator__reg_lambda': Real(5, 20)  
}

# 预先创建KFold对象，确保数据分割一致
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n{'=' * 60}")
print(f"开始训练多输出XGBoost模型...")
print(f"{'=' * 60}")

# 3. 创建XGBoost多输出管道
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', MultiOutputRegressor(
        XGBRegressor(
            random_state=42,
            n_jobs=1,  

            eval_metric='rmse'
        )
    ))
])

# 4. 贝叶斯优化
bayes_search = BayesSearchCV(
    pipeline,
    bayes_search_spaces,
    cv=kfold,  #
    scoring='neg_mean_squared_error',
    n_iter=50,  
    n_jobs=-1,
    verbose=1,
    random_state=42,
    return_train_score=True  # 确保返回训练分数
)

# 训练模型
bayes_search.fit(X_train[important_features], y_train)

# 输出最优参数组合的交叉验证结果
print(f"\n>>> 多输出XGBoost模型训练完成 <<<")
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

# 计算整体R²和MSE（在整个训练集上）
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

# 保存最佳模型
os.makedirs('output/models', exist_ok=True)
save_data(best_estimator, 'output/models/multi_output_xgb_model.joblib')
print(f"\n模型已保存到：output/models/multi_output_xgb_model.joblib")

# 5. 评估训练集和测试集性能
print(f"\n{'=' * 60}")
print(f"测试集性能评估")
print(f"{'=' * 60}")

# 测试集预测和评估
y_test_pred = best_estimator.predict(X_test[important_features])

# 评估函数
def evaluate_performance(y_true, y_pred, dataset_name):
    """评估模型性能"""
    results = {}
    mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    r2 = r2_score(y_true, y_pred, multioutput='raw_values')

    for i, target in enumerate(y_train.columns):
        results[target] = {
            'MSE': mse[i],
            'RMSE': rmse[i],
            'MAE': mae[i],
            'R2': r2[i]
        }

    results['Overall'] = {
        'MSE': np.mean(mse),
        'RMSE': np.mean(rmse),
        'MAE': np.mean(mae),
        'R2': np.mean(r2)
    }

    print(f"\n{dataset_name} 性能评估:")
    print("=" * 50)
    for target, metrics in results.items():
        print(f"{target}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()

    return results

# 评估训练集和测试集
train_results = evaluate_performance(y_train, y_pred_all, "训练集")
test_results = evaluate_performance(y_test, y_test_pred, "测试集")

# 保存评估结果
evaluation_results = {
    'train': train_results,
    'test': test_results,
    'best_params': bayes_search.best_params_,
    'best_score': bayes_search.best_score_
}
save_data(evaluation_results, 'output/models/xgb_multioutput_evaluation.joblib')

# 6. 打印模型总结
print("\n" + "=" * 60)
print("多输出XGBoost模型训练总结")
print("=" * 60)
print(f"最佳参数: {bayes_search.best_params_}")
print(f"最佳交叉验证分数 (负MSE): {bayes_search.best_score_:.4f}")
print(f"训练集平均R²: {train_results['Overall']['R2']:.4f}")
print(f"测试集平均R²: {test_results['Overall']['R2']:.4f}")
print(f"训练集平均RMSE: {train_results['Overall']['RMSE']:.4f}")
print(f"测试集平均RMSE: {test_results['Overall']['RMSE']:.4f}")

# 7. 检查过拟合情况
overfitting_ratio = test_results['Overall']['RMSE'] / train_results['Overall']['RMSE']
print(f"过拟合比率 (测试集RMSE/训练集RMSE): {overfitting_ratio:.4f}")

if overfitting_ratio > 1.5:
    print("⚠️ 警告：模型可能存在过拟合")
elif overfitting_ratio < 1.1:
    print("✅ 模型泛化能力良好")
else:
    print("🔶 模型泛化能力正常")

# 8. 输出每个目标变量的详细比较
print("\n" + "=" * 60)
print("各目标变量性能详细比较")
print("=" * 60)
for target in y_train.columns:
    train_r2 = train_results[target]['R2']
    test_r2 = test_results[target]['R2']
    train_rmse = train_results[target]['RMSE']
    test_rmse = test_results[target]['RMSE']
    r2_gap = test_r2 - train_r2
    rmse_ratio = test_rmse / train_rmse

    print(f"{target}:")
    print(f"  训练集 R²: {train_r2:.4f}, 测试集 R²: {test_r2:.4f}, 差异: {r2_gap:.4f}")
    print(f"  训练集 RMSE: {train_rmse:.4f}, 测试集 RMSE: {test_rmse:.4f}, 比率: {rmse_ratio:.4f}")

    if rmse_ratio > 3.0:
        print(f"  🚨 严重过拟合")
    elif rmse_ratio > 2.0:
        print(f"  ⚠️ 明显过拟合")
    elif rmse_ratio > 1.5:
        print(f"  🔶 轻微过拟合")
    else:
        print(f"  ✅ 泛化良好")
    print()