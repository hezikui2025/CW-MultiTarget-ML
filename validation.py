import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from save_load import load_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import os
import seaborn as sns

# 创建输出目录
os.makedirs('output/experiment_validation', exist_ok=True)

# 设置学术风格的绘图参数
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

# 使用Matplotlib内置的数学文本渲染
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# 定义学术颜色方案
academic_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#6B8E23']

# 定义目标变量的数学表达式显示名称
target_display_names = {
    'Eff.COD': r'Eff. COD (mg/L)',
    'Eff.NH4+-N': r'Eff. NH$_4^+$-N (mg/L)',
    'Eff.NO3--N': r'Eff. NO$_3^-$-N (mg/L)',
    'Inf.COD': r'Inf. COD (mg/L)',
    'Inf.NH4+-N': r'Inf. NH$_4^+$-N (mg/L)',
    'Inf.NO3--N': r'Inf. NO$_3^-$-N (mg/L)',
}

# 定义分组标签映射
group_labels = {
    0: 'G1', 1: 'G2', 2: 'G3', 3: 'G4', 4: 'G5',
    5: 'C1', 6: 'C2', 7: 'C3', 8: 'C4', 9: 'C5',
    10: 'FeC1', 11: 'FeC2', 12: 'FeC3', 13: 'FeC4', 14: 'FeC5'
}


def load_and_average_experiment_data(experiment_file_path):

    if experiment_file_path.endswith('.csv'):
        data = pd.read_csv(experiment_file_path)
    elif experiment_file_path.endswith(('.xls', '.xlsx')):
        data = pd.read_excel(experiment_file_path)
    else:
        raise ValueError("不支持的文件格式，请使用CSV或Excel文件")

    # 定义用于识别平行实验组的列（除了实际观测值外的所有特征列）
    feature_columns = ['湿地类型', '主要基质类型', '表面积', '厚度/cm', 'Inf.COD',
                       'Inf.NO3--N', 'Inf.NH4+-N', 'HRT/h', '曝气', '植物']

    # 确保所有特征列都存在
    missing_features = set(feature_columns) - set(data.columns)
    if missing_features:
        raise ValueError(f"实验数据缺少以下必要特征: {missing_features}")

    # 对平行实验组取平均值
    grouped = data.groupby(feature_columns)

    # 计算平均值
    avg_data = grouped.mean().reset_index()

    # 分离特征和标签
    X_experiment_avg = avg_data[feature_columns]

    # 检查是否有实际观测值
    target_columns = ['Eff.COD', 'Eff.NO3--N', 'Eff.NH4+-N']
    available_targets = [col for col in target_columns if col in avg_data.columns]

    if available_targets:
        y_experiment_avg = avg_data[available_targets]
    else:
        y_experiment_avg = None
        print("警告: 实验数据中没有发现实际观测值，将只进行预测")

    print(f"原始数据有 {len(data)} 个样本，平均后有 {len(avg_data)} 个实验组")

    return X_experiment_avg, y_experiment_avg, feature_columns


def plot_scatter_comparison(actual, predicted, target_name, metrics, save_path):

    fig, ax = plt.subplots(figsize=(8, 6))

    # 获取显示名称
    display_name = target_display_names.get(target_name, target_name)

    # 创建散点图
    scatter = ax.scatter(actual, predicted,
                         c=academic_colors[0],
                         alpha=0.8,
                         s=80,
                         edgecolors='white',
                         linewidth=1.0,
                         zorder=3)

    # 绘制1:1参考线
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    margin = (max_val - min_val) * 0.05
    ax.plot([min_val - margin, max_val + margin],
            [min_val - margin, max_val + margin],
            'k--', lw=2, alpha=0.7, zorder=2)

    # 设置坐标轴
    ax.set_xlabel('Actual Values', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted Values', fontsize=14, fontweight='bold')

    # 设置标题
    ax.set_title(f'{display_name}\nPredicted vs Actual', fontsize=16, fontweight='bold', pad=20)

    # 设置网格
    ax.grid(True, alpha=0.3, zorder=1)
    ax.set_axisbelow(True)

    # 设置坐标轴范围
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)

    # 添加评估指标文本框
    textstr = '\n'.join([
        f'R² = {metrics["R2"]:.3f}',
        f'RMSE = {metrics["RMSE"]:.3f}',
        f'MAE = {metrics["MAE"]:.3f}'
    ])
    props = dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    # 设置边框线宽
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_line_comparison(actual, predicted, target_name, save_path):

    fig, ax = plt.subplots(figsize=(12, 6))

    # 获取显示名称
    display_name = target_display_names.get(target_name, target_name)

    # 创建索引并映射到分组标签
    indices = np.arange(len(actual))
    x_labels = [group_labels.get(i, f'G{i}') for i in indices]

    # 绘制折线图
    ax.plot(indices, actual, label='Actual',
            color=academic_colors[0], marker='o', linewidth=2.5,
            markersize=8, markerfacecolor='white', markeredgewidth=2)

    ax.plot(indices, predicted, label='Predicted',
            color=academic_colors[1], marker='s', linewidth=2.5,
            markersize=8, markerfacecolor='white', markeredgewidth=2)

    # 设置坐标轴和标签
    ax.set_xlabel('Sample Group', fontsize=14, fontweight='bold')
    ax.set_ylabel('Concentration (mg/L)', fontsize=14, fontweight='bold')
    ax.set_title(f'{display_name}\nActual vs Predicted Comparison',
                 fontsize=16, fontweight='bold', pad=20)

    # 设置网格
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # 设置图例
    ax.legend(fontsize=12, framealpha=0.9, loc='best')

    # 设置x轴刻度和标签
    ax.set_xticks(indices)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')

    # 设置边框线宽
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_residual_analysis(actual, predicted, target_name, save_path):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 获取显示名称
    display_name = target_display_names.get(target_name, target_name)

    # 计算残差
    residuals = predicted - actual

    # 左图：残差vs预测值
    ax1.scatter(predicted, residuals, c=academic_colors[2], alpha=0.7, s=60, edgecolors='white', linewidth=1)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax1.set_xlabel('Predicted Values', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Residuals', fontsize=14, fontweight='bold')
    ax1.set_title(f'{display_name}\nResiduals vs Predicted', fontsize=15, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)

    # 右图：残差分布直方图
    ax2.hist(residuals, bins=15, color=academic_colors[3], alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax2.set_xlabel('Residuals', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax2.set_title(f'{display_name}\nResidual Distribution', fontsize=15, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)

    # 设置边框
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def add_group_labels_to_results(combined_results):

    results_with_groups = combined_results.copy()

    # 添加分组标签列
    results_with_groups['Group_Label'] = [group_labels.get(i, f'G{i}') for i in range(len(combined_results))]

    # 重新排列列，将分组标签放在前面
    cols = ['Group_Label'] + [col for col in results_with_groups.columns if col != 'Group_Label']
    results_with_groups = results_with_groups[cols]

    return results_with_groups


def validate_multi_output_model(X_experiment, y_experiment=None):

    results_dict = {}
    evaluation_metrics = []


    for feature in X_experiment.columns:
        results_dict[feature] = X_experiment[feature].values

    # 加载训练时的预处理数据以获取目标列顺序
    data = load_data('../output/preprocessed_data.joblib')
    target_columns = data['y_train'].columns

    # 加载多目标输出模型
    model = load_data('../3模型训练/output/models/gbdt_model_multi_output.joblib')


    y_pred = model.predict(X_experiment)

    # 将预测结果存入字典
    for i, target in enumerate(target_columns):
        results_dict[f'Predicted_{target}'] = y_pred[:, i]

        # 如果有实际观测值，计算评估指标
        if y_experiment is not None and target in y_experiment.columns:
            mse = mean_squared_error(y_experiment[target], y_pred[:, i])
            rmse = sqrt(mse)
            mae = mean_absolute_error(y_experiment[target], y_pred[:, i])
            r2 = r2_score(y_experiment[target], y_pred[:, i])

            metrics = {
                'Target': target,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
            evaluation_metrics.append(metrics)

            # 分别保存散点图、折线图和残差分析图
            scatter_path = f'output/experiment_validation/{target}_scatter_comparison.png'
            line_path = f'output/experiment_validation/{target}_line_comparison.png'
            residual_path = f'output/experiment_validation/{target}_residual_analysis.png'

            plot_scatter_comparison(
                y_experiment[target].values,
                y_pred[:, i],
                target,
                metrics,
                scatter_path
            )

            plot_line_comparison(
                y_experiment[target].values,
                y_pred[:, i],
                target,
                line_path
            )

            plot_residual_analysis(
                y_experiment[target].values,
                y_pred[:, i],
                target,
                residual_path
            )

            print(f"{target} 验证完成 - R²: {r2:.4f}, RMSE: {rmse:.4f}")
            print(f"散点图已保存至: {scatter_path}")
            print(f"折线图已保存至: {line_path}")
            print(f"残差分析图已保存至: {residual_path}")
        else:
            print(f"{target} 预测完成 - 无实际观测值可供比较")

    # 合并所有结果到一个DataFrame
    combined_results = pd.DataFrame(results_dict)

    # 如果有实际观测值，添加实际值到结果中
    if y_experiment is not None:
        for target in y_experiment.columns:
            combined_results[f'Actual_{target}'] = y_experiment[target].values

    # 添加分组标签到结果中
    combined_results = add_group_labels_to_results(combined_results)

    # 保存合并后的结果
    combined_results.to_csv('output/experiment_validation/combined_predictions.csv', index=False, float_format='%.4f')

    # 如果有评估指标，保存评估结果
    if evaluation_metrics:
        eval_df = pd.DataFrame(evaluation_metrics)
        eval_df.to_csv('output/experiment_validation/validation_metrics.csv', index=False, float_format='%.4f')

    return combined_results


if __name__ == "__main__":
    # 示例用法
    experiment_file = "../../实验验证数据.csv"  # 替换为您的实验数据路径

    print("=" * 80)
    print("实验数据验证".center(80))
    print("=" * 80)

    try:
        # 加载实验数据并对平行组取平均值
        X_exp, y_exp, grouping_cols = load_and_average_experiment_data(experiment_file)

        # 保存平均后的实验数据
        avg_data = X_exp.copy()
        if y_exp is not None:
            for col in y_exp.columns:
                avg_data[col] = y_exp[col]

        # 添加分组标签到平均数据中
        avg_data_with_groups = add_group_labels_to_results(avg_data)
        avg_data_with_groups.to_csv('output/experiment_validation/averaged_experiment_data.csv',
                                    index=False, float_format='%.4f')
        print(f"平均后的实验数据已保存至: output/experiment_validation/averaged_experiment_data.csv")

        # 进行验证并获取合并后的结果
        results = validate_multi_output_model(X_exp, y_exp)

        print("\n验证完成!")
        print(f"合并预测结果已保存至: output/experiment_validation/combined_predictions.csv")
        if y_exp is not None:
            print(f"评估指标已保存至: output/experiment_validation/validation_metrics.csv")
            print(f"学术风格的可视化图表已分别保存至独立的文件")
        print("=" * 80)
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback

        traceback.print_exc()