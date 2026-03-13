import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import sys
from matplotlib.lines import Line2D
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    roc_curve
)

# Ensure Python 3
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})


def find_results_path(target_folder):
    """
    Searches for the results folder in current or parent directories.
    """
    possible_paths = [
        target_folder,
        os.path.join('..', target_folder),
        os.path.join('.', target_folder)
    ]

    for path in possible_paths:
        if os.path.isdir(path):
            if glob.glob(os.path.join(path, "*.csv")):
                return path
    return None


def load_data(scenario_path, model_name="IDM_MOBIL"):
    search_path = os.path.join(scenario_path, f"{model_name}_*.csv")
    all_files = glob.glob(search_path)

    if not all_files:
        print(f"No result files found in {scenario_path} for {model_name}.")
        return None

    df_list = []
    print(f"Found {len(all_files)} files. Loading...")
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            df_list.append(df)
        except Exception as e:
            print(f"Skipping corrupt file {filename}: {e}")

    if not df_list:
        return None

    combined_df = pd.concat(df_list, ignore_index=True)

    # --- DATA CLEANING ---
    combined_df['Crash_Num'] = combined_df['Crash'].astype(int)

    # Fix Y-Axis Values (Round to 1 decimal place)
    if 'lat_vel' in combined_df.columns:
        combined_df['lat_vel'] = combined_df['lat_vel'].round(1)
    if 'long_dist' in combined_df.columns:
        combined_df['long_dist'] = combined_df['long_dist'].round(1)

    # Clean TTC
    if 'minTTC' in combined_df.columns:
        combined_df['minTTC_clean'] = combined_df['minTTC'].replace([np.inf, -np.inf], 50).fillna(50)
        # Inverse TTC (High Value = High Risk) is better for Correlation & ROC
        combined_df['TTC_Inverse'] = 1.0 / (combined_df['minTTC_clean'] + 0.1)

    return combined_df


def print_classification_metrics(df):
    """
    Calculates and prints Precision, Recall, F1, Accuracy.
    """
    print("\n" + "=" * 40)
    print("      CLASSIFICATION METRICS REPORT")
    print("      (Threshold: TTC < 1.5s)")
    print("=" * 40)

    y_true = df['Crash']
    # Prediction: If TTC is low, we predict a crash
    y_pred = df['minTTC_clean'] < 1.5

    # Check if we have both classes to avoid errors
    if len(y_true.unique()) < 2:
        print("NOTE: Data contains only one class (e.g. all Safe). Metrics may be 0.0 or undefined.")

    # Calculate Scores
    # zero_division=0 ensures script doesn't crash if we divide by zero
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f" Accuracy:  {acc:.4f}")
    print(f" Precision: {prec:.4f}")
    print(f" Recall:    {rec:.4f}")
    print(f" F1 Score:  {f1:.4f}")
    print("=" * 40 + "\n")


def plot_correlation_matrix(df, output_dir):
    plt.figure(figsize=(10, 8))
    cols = ['minTTC_clean', 'long_dist', 'lat_vel', 'Crash_Num', 'TTC_Inverse']
    valid_cols = [c for c in cols if c in df.columns]

    rename_map = {'minTTC_clean': 'TTC', 'long_dist': 'Headway (dhw)',
                  'lat_vel': 'Lat. Velocity', 'Crash_Num': 'Crash', 'TTC_Inverse': 'Inv TTC'}

    data_for_corr = df[valid_cols].rename(columns=rename_map)
    data_for_corr = data_for_corr.loc[:, (data_for_corr != data_for_corr.iloc[0]).any()]

    if data_for_corr.empty:
        return

    corr_data = data_for_corr.corr()

    sns.heatmap(corr_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=1, linecolor='white', fmt=".2f",
                cbar_kws={'label': 'Correlation Coefficient'})

    plt.suptitle('Parameter Correlation Matrix', fontweight='bold')
    plt.title('Red = Positive Correlation | Blue = Negative Correlation', fontsize=12, color='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'IDM_Correlation_Matrix.png'), dpi=300)
    plt.close()


def plot_safety_density(df, output_dir):
    plt.figure(figsize=(11, 7))
    plot_df = df[df['minTTC_clean'] < 20].copy()

    sns.kdeplot(
        data=plot_df, x='long_dist', y='lat_vel',
        fill=True, thresh=0.05, levels=10, cmap="viridis", alpha=0.6,
        cbar=True, cbar_kws={'label': 'Simulation Frequency'}
    )

    crashes = df[df['Crash'] == True]
    legend_elements = []

    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='High Density Area (Yellow)',
                                  markerfacecolor='gold', markersize=10))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Low Density Area (Purple)',
                                  markerfacecolor='indigo', markersize=10))

    if not crashes.empty:
        plt.scatter(
            crashes['long_dist'], crashes['lat_vel'],
            marker='x', color='red', s=100, linewidth=2.5, zorder=10
        )
        legend_elements.append(Line2D([0], [0], marker='x', color='w', label='Crash Event (Red X)',
                                      markeredgecolor='red', markersize=10, linestyle='None'))

    plt.legend(handles=legend_elements, loc='upper right', frameon=True, title="Legend")
    plt.suptitle('IDM+MOBIL: Scenario Density', fontweight='bold')
    plt.title('Yellow = Most Common Scenarios | Purple = Rare Scenarios', fontsize=12, color='gray')
    plt.xlabel('Initial Distance (m)')
    plt.ylabel('Lateral Velocity (m/s)')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'IDM_Safety_Density.png'), dpi=300)
    plt.close()


def plot_risk_heatmap(df, output_dir):
    plt.figure(figsize=(12, 8))

    if 'lat_vel' in df.columns and 'long_dist' in df.columns:
        grid_data = df.pivot_table(index='lat_vel', columns='long_dist', values='Crash_Num', aggfunc='max')
        grid_data = grid_data.sort_index(ascending=False)

        ax = sns.heatmap(grid_data, cmap='hot', vmin=0, vmax=1,
                         cbar_kws={'label': 'Crash Indicator'},
                         linewidths=0.0, rasterized=True)

        plt.suptitle('Crash Risk Map', fontweight='bold')
        plt.title('Black = Safe (No Crash) | Red/White = Crash Occurred', fontsize=12, color='gray')
        plt.xlabel('Initial Distance (m)')
        plt.ylabel('Lateral Velocity (m/s)')

        if len(grid_data.index) > 20:
            for label in ax.yaxis.get_ticklabels():
                label.set_visible(False)
            for label in ax.yaxis.get_ticklabels()[::2]:
                label.set_visible(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'IDM_Risk_Heatmap.png'), dpi=300)
        plt.close()


def plot_confusion_matrix(df, output_dir):
    plt.figure(figsize=(8, 7))
    y_true = df['Crash']
    y_pred = df['minTTC_clean'] < 1.5

    labels_list = [False, True]
    cm = confusion_matrix(y_true, y_pred, labels=labels_list)

    group_names = ['True Neg\n(Safe)', 'False Pos\n(Alarm)', 'False Neg\n(Missed)', 'True Pos\n(Hit)']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]

    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=True,
                cbar_kws={'label': 'Number of Cases'},
                xticklabels=['Pred: Safe', 'Pred: Crash'], yticklabels=['Actual: Safe', 'Actual: Crash'])

    plt.suptitle('Confusion Matrix (TTC < 1.5s)', fontweight='bold')
    plt.title('Darker Blue = Higher Number of Cases', fontsize=12, color='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'IDM_Confusion_Matrix.png'), dpi=300)
    plt.close()


def plot_roc_curve(df, output_dir):
    """
    NEW: Plots the Receiver Operating Characteristic (ROC) Curve.
    Metric: TTC Inverse (Higher = More Risk).
    """
    plt.figure(figsize=(8, 6))

    y_true = df['Crash_Num']
    # We use TTC_Inverse because ROC expects High Score = Positive Class (Crash)
    y_score = df['TTC_Inverse']

    # Check if we have crashes, otherwise ROC is undefined
    if len(y_true.unique()) > 1:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)

        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.suptitle('Receiver Operating Characteristic (ROC)', fontweight='bold')
        plt.title('Performance of "Inverse TTC" as a Crash Predictor', fontsize=12, color='gray')
        plt.legend(loc="lower right")
    else:
        plt.text(0.5, 0.5, "ROC Undefined\n(No Crashes in Data)",
                 ha='center', va='center', fontsize=14, color='red')
        plt.title('Receiver Operating Characteristic (ROC)')
        print("\n[INFO] Skipping ROC Score calculation (Only one class present in data).")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'IDM_ROC_Curve.png'), dpi=300)
    plt.close()


if __name__ == "__main__":
    # --- CONFIG ---
    target_folder = 'results/cut_in_high_speed'

    found_path = find_results_path(target_folder)

    if found_path is None:
        print("ERROR: Could not find results folder.")
        print("Run 'python example.py' first.")
        exit()

    print(f"Loading IDM_MOBIL data from: {found_path}")
    output_dir = found_path

    df = load_data(found_path)

    if df is not None:
        print(f"Loaded {len(df)} rows. Generating improved plots...")
        try:
            # 1. Print Text Metrics (Precision, Recall, F1, Accuracy)
            print_classification_metrics(df)

            # 2. Generate Plots
            plot_correlation_matrix(df, output_dir)
            print(" - Correlation Matrix Saved")

            plot_safety_density(df, output_dir)
            print(" - Safety Density Plot Saved")

            plot_risk_heatmap(df, output_dir)
            print(" - Risk Heatmap Saved")

            plot_confusion_matrix(df, output_dir)
            print(" - Confusion Matrix Saved")

            # 3. New ROC Curve
            plot_roc_curve(df, output_dir)
            print(" - ROC Curve Saved (NEW)")

            print(f"\nSUCCESS! Images saved in: {output_dir}")
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("Data empty.")