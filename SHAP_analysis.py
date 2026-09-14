import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from save_load import load_data
from tqdm import tqdm
import matplotlib.font_manager as fm
import seaborn as sns

# ============================================================
# 出版级绘图参数设置
# ============================================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

# ===== 核心字体大小 =====
BASE_FONT_SIZE = 18
TITLE_FONT_SIZE = 24
AXIS_LABEL_SIZE = 22
AXIS_TICK_SIZE = 18
LEGEND_SIZE = 18
ANNOTATION_SIZE = 16

plt.rcParams['font.size'] = BASE_FONT_SIZE
plt.rcParams['axes.labelsize'] = AXIS_LABEL_SIZE
plt.rcParams['axes.titlesize'] = TITLE_FONT_SIZE
plt.rcParams['xtick.labelsize'] = AXIS_TICK_SIZE
plt.rcParams['ytick.labelsize'] = AXIS_TICK_SIZE
plt.rcParams['legend.fontsize'] = LEGEND_SIZE
plt.rcParams['legend.title_fontsize'] = LEGEND_SIZE

plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 10
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['ytick.major.size'] = 10

plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.05

os.makedirs('output/shap_analysis/test_set/high_res', exist_ok=True)

# 加载数据和模型
print("正在加载数据...")
try:
    data = load_data('../output/preprocessed_data.joblib')
    important_features = ['表面积', '厚度/cm', 'Inf.COD', 'Inf.NO3--N', 'Inf.NH4+-N', 'HRT/h', '曝气', '主要基质类型',
                          '植物']
    X_test = data['X_test'][important_features]
    y_test = data['y_test']
except Exception as e:
    raise Exception(f"数据加载失败: {str(e)}")

print("正在加载多目标GBDT模型...")
try:
    gbdt_model = load_data('../3模型训练/output/models/gbdt_model_multi_output.joblib')
    print("多目标GBDT模型加载成功")
except Exception as e:
    raise Exception(f"模型加载失败: {str(e)}")

# 特征分类
important_numeric = [f for f in important_features if
                     f in ['表面积', '厚度/cm', 'Inf.COD', 'Inf.NO3--N', 'Inf.NH4+-N', 'HRT/h', '曝气']]
important_categorical = [f for f in important_features if f in ['主要基质类型', '植物']]

preprocessor = gbdt_model.named_steps['preprocessor']
try:
    num_features = preprocessor.named_transformers_['num'].get_feature_names_out(important_numeric)
except AttributeError:
    num_features = important_numeric
try:
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(important_categorical)
except AttributeError:
    cat_features = important_categorical
feature_names = np.concatenate([num_features, cat_features])

feature_display_names = {
    '主要基质类型': 'Substrate',
    '植物': 'Plants',
    '表面积': 'Area_m2',
    '厚度/cm': 'Filter_thickness_cm',
    'Inf.COD': 'Inf_COD_mgL',
    'Inf.NO3--N': 'Inf_NO3_N_mgL',
    'Inf.NH4+-N': 'Inf_NH4_N_mgL',
    'HRT/h': 'HRT_h',
    '曝气': 'Aeration_volume_L/d'
}

feature_plot_names = {
    '主要基质类型': 'Substrate',
    '植物': 'Plants',
    '表面积': 'Area (m²)',
    '厚度/cm': 'Filter thickness (cm)',
    'Inf.COD': 'Inf. COD (mg/L)',
    'Inf.NO3--N': 'Inf. NO₃⁻-N (mg/L)',
    'Inf.NH4+-N': 'Inf. NH₄⁺-N (mg/L)',
    'HRT/h': 'HRT (h)',
    '曝气': 'Aeration volume (L/d)'
}

target_display_names = {
    'Eff.COD': 'Eff_COD_mgL',
    'Eff.NH4+-N': 'Eff_NH4_N_mgL',
    'Eff.NO3--N': 'Eff_NO3_N_mgL',
    'Inf.COD': 'Inf_COD_mgL',
    'Inf.NH4+-N': 'Inf_NH4_N_mgL',
    'Inf.NO3--N': 'Inf_NO3_N_mgL',
}

target_plot_names = {
    'Eff.COD': 'Eff. COD (mg/L)',
    'Eff.NH4+-N': 'Eff. NH₄⁺-N (mg/L)',
    'Eff.NO3--N': 'Eff. NO₃⁻-N (mg/L)',
    'Inf.COD': 'Inf. COD (mg/L)',
    'Inf.NH4+-N': 'Inf. NH₄⁺-N (mg/L)',
    'Inf.NO3--N': 'Inf. NO₃⁻-N (mg/L)',
}

safe_feature_names = []
plot_feature_names = []
for feature in feature_names:
    found = False
    for original_name, safe_name in feature_display_names.items():
        if original_name in feature:
            safe_feature_names.append(safe_name)
            plot_feature_names.append(feature_plot_names[original_name])
            found = True
            break
    if not found:
        safe_feature_names.append(feature.replace(' ', '_').replace('/', '_'))
        plot_feature_names.append(feature)

print("安全特征名称:", safe_feature_names)

try:
    X_test_transformed = preprocessor.transform(X_test)
except Exception as e:
    raise Exception(f"数据预处理失败: {str(e)}")

# ============================================================

# ============================================================
print("提取GBDT模型...")
try:
    gb_multi_model = gbdt_model.named_steps['model']

    # ============================================================
    # 从模型中获取目标变量的实际顺序
    # ============================================================
    #  尝试从模型的属性中获取目标变量名称
    model_target_names = None
    if hasattr(gb_multi_model, 'output_names_'):
        model_target_names = gb_multi_model.output_names_
        print(f"模型目标变量 (output_names_): {model_target_names}")
    elif hasattr(gb_multi_model, 'target_names_'):
        model_target_names = gb_multi_model.target_names_
        print(f"模型目标变量 (target_names_): {model_target_names}")

    # 如果模型没有存储目标变量名，使用y_test的列顺序
    if model_target_names is None:
        model_target_names = list(y_test.columns)
        print(f"使用y_test列名作为目标变量顺序: {model_target_names}")

    # 检查估计器数量
    if hasattr(gb_multi_model, 'estimators_'):
        gb_estimators = gb_multi_model.estimators_
        n_estimators = len(gb_estimators)
        print(f"GBDT模型包含 {n_estimators} 个估计器")

        # ============================================================
        # 根据模型实际的目标变量顺序，重新排列target_names
        # ============================================================

        actual_target_names = model_target_names[:n_estimators]
        print(f"模型实际支持的目标变量: {actual_target_names}")

        # 使用模型的实际顺序
        target_names = actual_target_names
    else:
        gb_estimators = [gb_multi_model] * len(model_target_names)
        target_names = model_target_names

except Exception as e:
    raise Exception(f"模型提取失败: {str(e)}")

print(f"\n最终目标变量列表 (按模型顺序): {target_names}")
print("=" * 60)

# ============================================================
# 创建SHAP解释器时，明确建立目标变量到索引的映射
# ============================================================
print("准备SHAP分析...")
gb_explainers = []
target_to_index = {}  # 目标变量名 -> 估计器索引

for i, target_name in enumerate(target_names):
    try:
        if i >= len(gb_estimators):
            print(f"警告: 目标变量 {target_name} 的索引 {i} 超出估计器范围")
            gb_explainers.append(None)
            continue

        gb_explainer = shap.TreeExplainer(gb_estimators[i])
        gb_explainers.append(gb_explainer)
        target_to_index[target_name] = i
        print(f"  索引 {i}: {target_name} -> SHAP解释器创建成功")
    except Exception as e:
        print(f"  索引 {i}: {target_name} -> 创建SHAP解释器失败 - {str(e)}")
        gb_explainers.append(None)


def get_gbdt_shap_values(target_name, X_data):
    """根据目标变量名称获取对应的SHAP值"""
    if target_name not in target_to_index:
        raise Exception(f"目标变量 {target_name} 不在模型支持列表中: {list(target_to_index.keys())}")

    target_idx = target_to_index[target_name]
    explainer = gb_explainers[target_idx]
    if explainer is None:
        raise Exception(f"GBDT解释器对于目标变量 {target_name} (索引 {target_idx}) 不可用")

    shap_values = explainer.shap_values(X_data)
    return shap_values


def get_color_scheme(scheme_name, n_features, reverse=True):
    if scheme_name == 'magenta_to_blue':
        colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, n_features))
    elif scheme_name == 'custom_magenta_blue':
        magenta_blue_colors = [
            '#FF00FF', '#E600E6', '#CC00CC', '#B300B3', '#990099',
            '#800080', '#660066', '#4D004D', '#330033', '#0000FF'
        ]
        if n_features <= len(magenta_blue_colors):
            colors = magenta_blue_colors[:n_features]
        else:
            colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, n_features))
    elif scheme_name == 'viridis':
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_features))
    elif scheme_name == 'plasma':
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, n_features))
    elif scheme_name == 'cool_warm':
        colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, n_features))
    else:
        colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, n_features))
    if reverse:
        colors = colors[::-1]
    return colors


# ================================================================
# SHAP摘要图
# ================================================================
def create_publication_summary_plot(shap_values, features, feature_names, target_name,
                                    color_scheme='magenta_to_blue', figsize=(18, 13)):
    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111)

    feature_importance = np.abs(shap_values).mean(0)
    indices = np.argsort(feature_importance)[::-1]
    n_features = min(10, len(feature_names))
    indices = indices[:n_features]

    total_importance = np.sum(feature_importance[indices])
    percentages = (feature_importance[indices] / total_importance) * 100

    colors = get_color_scheme(color_scheme, n_features, reverse=True)

    y_pos = np.arange(n_features)
    bars = ax.barh(y_pos, feature_importance[indices],
                   color=colors, alpha=0.85, height=0.7,
                   edgecolor='white', linewidth=3)

    feature_fontsize = 22
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices],
                       fontsize=feature_fontsize, fontweight='bold')
    ax.invert_yaxis()

    xlabel_fontsize = 24
    ylabel_fontsize = 24
    ax.set_xlabel('Mean |SHAP Value|', fontsize=xlabel_fontsize, fontweight='bold', labelpad=18)
    ax.set_ylabel('Features', fontsize=ylabel_fontsize, fontweight='bold', labelpad=18)

    title_fontsize = 28
    ax.set_title(f'SHAP Feature Importance for {target_name}',
                 fontsize=title_fontsize, fontweight='bold', pad=35)

    bar_text_fontsize = 18
    x_max = np.max(feature_importance[indices])

    for i, (bar, importance_val) in enumerate(zip(bars, feature_importance[indices])):
        width = bar.get_width()
        text_x = width + x_max * 0.02
        ax.text(text_x, bar.get_y() + bar.get_height() / 2,
                f'{importance_val:.3f}',
                ha='left', va='center',
                fontsize=bar_text_fontsize, fontweight='bold',
                color='black')

    ax.grid(True, alpha=0.4, axis='x', linestyle='--', linewidth=1.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('black')
    ax.set_facecolor('#f5f5f5')

    max_width = np.max(feature_importance[indices])
    ax.set_xlim(0, max_width * 1.30)

    tick_fontsize = 18
    ax.tick_params(axis='y', which='both', left=False)
    ax.tick_params(axis='x', labelsize=tick_fontsize)

    # 圆环图
    donut_left = 0.52
    donut_bottom = 0.12
    donut_width = 0.48
    donut_height = 0.48

    donut_ax = fig.add_axes([donut_left, donut_bottom, donut_width, donut_height])

    wedges, texts, autotexts = donut_ax.pie(percentages,
                                            colors=colors,
                                            autopct='%1.1f%%',
                                            startangle=90,
                                            wedgeprops=dict(width=0.3, edgecolor='w', linewidth=2.5),
                                            textprops={'fontsize': 18, 'fontweight': 'bold'})

    donut_title_fontsize = 22
    donut_ax.set_title('Importance Percentage', fontsize=donut_title_fontsize, fontweight='bold', pad=45)

    donut_percent_fontsize = 18
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(donut_percent_fontsize)

    for i, (wedge, percentage) in enumerate(zip(wedges, percentages)):
        angle = (wedge.theta2 + wedge.theta1) / 2

        start_radius = 1.02
        end_radius = 1.15

        start_x = start_radius * np.cos(np.radians(angle))
        start_y = start_radius * np.sin(np.radians(angle))
        end_x = end_radius * np.cos(np.radians(angle))
        end_y = end_radius * np.sin(np.radians(angle))

        donut_ax.plot([start_x, end_x], [start_y, end_y],
                      color='gray', linewidth=1.5, alpha=0.7)

        label_text = f'{percentage:.1f}%'
        normalized_angle = angle % 360
        if normalized_angle > 180:
            normalized_angle -= 360

        if -90 <= normalized_angle <= 90:
            ha = 'left'
            rotation = normalized_angle
        else:
            ha = 'right'
            rotation = normalized_angle + 180

        text_x = end_x * 1.05
        text_y = end_y * 1.05

        donut_ax.text(text_x, text_y, label_text,
                      ha=ha, va='center',
                      fontsize=donut_percent_fontsize, fontweight='bold',
                      rotation=rotation, rotation_mode='anchor',
                      bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                alpha=0.95, edgecolor='gray', linewidth=1.0))

    for text in texts:
        text.set_visible(False)
    for autotext in autotexts:
        autotext.set_visible(False)

    donut_ax.axis('equal')

    plt.tight_layout()
    return fig


def create_publication_detailed_plot(shap_values, features, feature_names, target_name,
                                     color_scheme='viridis', figsize=(15, 10)):
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')

    feature_importance = np.abs(shap_values).mean(0)
    indices = np.argsort(feature_importance)[::-1]
    n_features = min(10, len(feature_names))
    indices = indices[:n_features]

    shap.summary_plot(shap_values[:, indices], features[:, indices],
                      feature_names=[feature_names[i] for i in indices],
                      show=False, plot_size=(figsize[0], figsize[1]),
                      color=plt.cm.viridis)

    ax = plt.gca()
    ax.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=24, fontweight='bold')
    ax.set_ylabel('Features', fontsize=24, fontweight='bold')
    ax.set_title(f'SHAP Feature Impact for {target_name}',
                 fontsize=28, fontweight='bold', pad=30)

    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    ax.grid(True, alpha=0.3, axis='x', linestyle='--', linewidth=1.2)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    plt.tight_layout()
    return fig


def create_publication_dependence_plot(shap_values, features, feature_names, target_name,
                                       feature_idx, color_scheme='plasma', figsize=(12, 9)):
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')

    feature_name = feature_names[feature_idx]
    feature_values = features[:, feature_idx]
    shap_values_feature = shap_values[:, feature_idx]

    scatter = ax.scatter(feature_values, shap_values_feature,
                         c=feature_values, cmap=color_scheme,
                         alpha=0.7, s=70, edgecolors='white', linewidth=1.2)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f'{feature_name} Value', fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=16)

    if len(feature_values) > 2:
        try:
            z = np.polyfit(feature_values, shap_values_feature, 2)
            p = np.poly1d(z)
            x_sorted = np.sort(feature_values)
            ax.plot(x_sorted, p(x_sorted), "r--", alpha=0.9, linewidth=3.5, label='Trend line')
            ax.legend(fontsize=16, loc='best')
        except:
            try:
                z = np.polyfit(feature_values, shap_values_feature, 1)
                p = np.poly1d(z)
                x_sorted = np.sort(feature_values)
                ax.plot(x_sorted, p(x_sorted), "r--", alpha=0.9, linewidth=3.5, label='Trend line')
                ax.legend(fontsize=16, loc='best')
            except:
                pass

    ax.set_xlabel(feature_name, fontsize=20, fontweight='bold')
    ax.set_ylabel('SHAP Value', fontsize=20, fontweight='bold')
    ax.set_title(f'SHAP Dependence Plot: {feature_name} - {target_name}',
                 fontsize=22, fontweight='bold', pad=28)

    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.2)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    plt.tight_layout()
    return fig


def sanitize_filename(filename):
    invalid_chars = '<>:"/\\|?* '
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename


COLOR_SCHEMES = {
    'summary': 'magenta_to_blue',
    'detailed': 'viridis',
    'dependence': 'plasma'
}

# ============================================================
# 主分析循环
# ============================================================
print("\n" + "=" * 60)
print(f"开始SHAP分析，共 {len(target_names)} 个目标变量")
print("=" * 60 + "\n")

for target_idx, target_name in enumerate(tqdm(target_names, desc="SHAP分析进度")):
    print(f"\n分析目标变量: {target_name} (使用模型: GBDT)")

    try:
        sample_size = min(500, X_test_transformed.shape[0])
        sample_idx = np.random.choice(X_test_transformed.shape[0], sample_size, replace=False)

        # ============================================================
        # 使用目标变量名称获取SHAP值，而不是索引
        # ============================================================
        test_shap_values = get_gbdt_shap_values(
            target_name, X_test_transformed[sample_idx, :]
        )

        target_safe_name = target_display_names.get(target_name, target_name)
        target_display_name = target_plot_names.get(target_name, target_name)

        out_dir = 'output/shap_analysis/test_set/high_res'

        # 1. 摘要图
        fig = create_publication_summary_plot(
            test_shap_values,
            X_test_transformed[sample_idx, :],
            plot_feature_names,
            target_display_name,
            color_scheme=COLOR_SCHEMES['summary'],
            figsize=(18, 13)
        )
        safe_filename = sanitize_filename(f'gbdt_shap_summary_test_{target_safe_name}.png')
        fig.savefig(os.path.join(out_dir, safe_filename),
                    dpi=600, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✓ 摘要图保存: {safe_filename}")

        # 2. 详细图
        fig = create_publication_detailed_plot(
            test_shap_values,
            X_test_transformed[sample_idx, :],
            plot_feature_names,
            target_display_name,
            color_scheme=COLOR_SCHEMES['detailed'],
            figsize=(15, 10)
        )
        safe_filename = sanitize_filename(f'gbdt_shap_detailed_test_{target_safe_name}.png')
        fig.savefig(os.path.join(out_dir, safe_filename),
                    dpi=600, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✓ 详细图保存: {safe_filename}")

        # 3. 依赖图
        shap_df = pd.DataFrame(test_shap_values, columns=safe_feature_names)
        top_features = shap_df.abs().mean().sort_values(ascending=False).index[:3]

        for feature in top_features:
            feature_idx = safe_feature_names.index(feature)
            fig = create_publication_dependence_plot(
                test_shap_values,
                X_test_transformed[sample_idx, :],
                plot_feature_names,
                target_display_name,
                feature_idx,
                color_scheme=COLOR_SCHEMES['dependence'],
                figsize=(12, 9)
            )
            safe_feature_filename = sanitize_filename(feature)
            safe_filename = sanitize_filename(
                f'gbdt_shap_dependence_test_{target_safe_name}_{safe_feature_filename}.png'
            )
            fig.savefig(os.path.join(out_dir, safe_filename),
                        dpi=600, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"  ✓ 依赖图保存: {safe_filename}")

        # 保存数据
        pd.DataFrame(test_shap_values, columns=feature_names).to_csv(
            os.path.join(out_dir, f'gbdt_shap_values_test_{target_safe_name}.csv'),
            index=False
        )

        abs_shap = np.abs(test_shap_values)
        total_shap = np.sum(abs_shap, axis=0)
        percentage_shap = (total_shap / np.sum(total_shap)) * 100

        shap_percentage_df = pd.DataFrame({
            'Feature': plot_feature_names,
            'SHAP_Absolute_Sum': total_shap,
            'SHAP_Percentage(%)': percentage_shap,
            'Used_Model': 'GBDT'
        }).sort_values('SHAP_Percentage(%)', ascending=False)

        shap_percentage_df.to_csv(
            os.path.join(out_dir, f'gbdt_shap_percentage_test_{target_safe_name}.csv'),
            index=False,
            float_format='%.2f'
        )

        print(f"目标变量 {target_name} 分析完成\n")

    except Exception as e:
        print(f"分析目标变量 {target_name} 时出错: {str(e)}")
        import traceback

        traceback.print_exc()
        continue

print("\n" + "=" * 60)
print("SHAP分析完成！")
print("=" * 60)
print(f"结果保存在: output/shap_analysis/test_set/high_res/")
print(f"共分析了 {len(target_names)} 个目标变量")
print("=" * 60)