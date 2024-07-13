import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.lines import Line2D

from scipy import stats
from scipy.stats import wilcoxon, spearmanr
import scipy.interpolate

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

from sklearn.utils import resample

import os

#============================================================================================#
#============================================================================================#

antibody = "IgG"  # Change parameter. e.g., IgG, IgM, IgA

# Define the base path and filenames
base_path = "../data/"
experiment_numbers = ["Ex0001", "Ex0002", "Ex0003", "Ex0004"]
file_types = [" MFI.xlsx", " bead count.xlsx"]

# Generate datasets list dynamically
datasets = [(f"{base_path}{exp} {antibody}{file_type1}", f"{base_path}{exp} {antibody}{file_type2}")
            for exp in experiment_numbers
            for file_type1, file_type2 in zip(file_types, file_types[1:])]

# Define the name of the new folder
result_folder = "../"+antibody+"_result/"

# Create the new folder
os.makedirs(result_folder, exist_ok=True)

#============================================================================================#
#============================================================================================#
# Data processing

# Function to calculate the coefficient of variation (CV)
def calculate_cv(dataframes):
    cv_result = pd.DataFrame(columns=dataframes[0].columns, index=dataframes[0].index)
    for index in dataframes[0].index:
        for column in dataframes[0].columns:
            values = [pd.to_numeric(df.at[index, column], errors='coerce') for df in dataframes]
            mean_value = np.nanmean(values)
            std_value = np.nanstd(values)
            cv = std_value / mean_value * 100 if mean_value != 0 else np.nan
            cv_result.at[index, column] = cv
    return cv_result

# Function to process Luminex data
def process_luminex_data(mfi_file, bead_file):
    data = TrimData1(mfi_file, bead_file)
    trimmed_data = data.trim_data_by_bead_count(30)
    avg_mfi, std_mfi, cv_mfi, avg_mfi_sc, std_mfi_sc, cv_mfi_sc = data.subtract_background_and_calculate_stats(trimmed_data, antibody)
    return avg_mfi_sc, avg_mfi

# Import the necessary class
from Functions import TrimData1, TrimData2

# Process each dataset and store results
avg_mfi_results = []
avg_mfi_sc_results = []

for mfi_file, bead_file in datasets:
    avg_mfi_sc, avg_mfi = process_luminex_data(mfi_file, bead_file)
    avg_mfi_sc_results.append(avg_mfi_sc)
    avg_mfi_results.append(avg_mfi)

# Calculate CV for SC samples
sc_cv_result = calculate_cv(avg_mfi_sc_results)

# Normalize the standard curve factors
norm_factors = {}
for column in sc_cv_result.columns:
    try:
        min_cv = sc_cv_result[column].min()
        min_index = sc_cv_result[column].astype(float).idxmin()
        norm_factors[column] = (min_index, min_cv)
    except Exception as e:
        print(f"Error processing column {column}: {e}")

# Calculate normalization factors
normalization_factors = {}
for bead, (index_name, _) in norm_factors.items():
    norm_values = [avg_mfi_sc_results[i].at[index_name, bead] for i in range(4)]
    reference_value = norm_values[0]
    normalization_factors[bead] = [value / reference_value for value in norm_values]

# Apply normalization factors to each dataset
for i in range(4):
    for column in avg_mfi_results[i].columns:
        avg_mfi_results[i][column] /= normalization_factors[column][i]

# Function to process and arrange data
def process_arranged_data(avg_mfi, sample_info_file):
    data = TrimData2(avg_mfi, sample_info_file)
    return data.data_arrangement()

# Process data for visualization
ex1_grouped_data = process_arranged_data(avg_mfi_results[0], base_path+"Sample info.xlsx")
ex2_grouped_data = process_arranged_data(avg_mfi_results[1], base_path+"Sample info.xlsx")
ex3_grouped_data = process_arranged_data(avg_mfi_results[2], base_path+"Sample info.xlsx")
ex4_grouped_data = process_arranged_data(avg_mfi_results[3], base_path+"Sample info.xlsx")

# Combine all datasets
combined_data = pd.concat([ex1_grouped_data, ex2_grouped_data, ex3_grouped_data, ex4_grouped_data])

# Function to define outlier detection using IQR method
def calculate_upper_fence(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    upper_fence = Q3 + (1.5 * IQR)
    return upper_fence

# Detect and remove outliers
hc_epa_data = combined_data["EPA"][[index for index in combined_data.index if "HC" in index]]
upper_fence = calculate_upper_fence(hc_epa_data)

kpn_remove_list = []
bc_remove_list = []
hc_remove_list = []

for index in combined_data.index:
    if "KPN" in index:
        if combined_data.at[index, "EPA"] > upper_fence:
            kpn_name = index.split('_')[0]
            if kpn_name not in kpn_remove_list:
                kpn_remove_list.append(kpn_name)
    elif "BC" in index:
        if combined_data.at[index, "EPA"] > upper_fence:
            bc_remove_list.append(index)
    elif "HC" in index:
        if combined_data.at[index, "EPA"] > upper_fence:
            hc_remove_list.append(index)

# Apply removal criteria
columns_to_nan = ["O1v1-EPA", "O2v2-EPA", "O3b-EPA", "EPA"]
for index in combined_data.index:
    if "KPN" in index:
        kpn_name = index.split("_")[0]
        if kpn_name in kpn_remove_list:
            combined_data.loc[index, columns_to_nan] = np.nan
    elif "BC" in index:
        if index in bc_remove_list:
            combined_data.loc[index, columns_to_nan] = np.nan
    elif "HC" in index:
        if index in hc_remove_list:
            combined_data.loc[index, columns_to_nan] = np.nan

# Further adjustments to specific samples
for index in combined_data.index:
    if "KPN6" == index.split("_")[0]:
        combined_data.at[index, "MrkA"] = np.nan
    if "KPN55" == index.split("_")[0]:
        combined_data.at[index, "MrkA"] = np.nan

# Read sample information from Sample info2.xlsx
sample_info = pd.read_excel(base_path+"Sample info2.xlsx")
sample_info.set_index("KPN", inplace=True)

# Initialize new columns for Condition and Infection site
combined_data["Condition"] = combined_data.index.map(lambda x: sample_info.at[x.split('_')[0], "Condition"] if x.split('_')[0] in sample_info.index else np.nan)
combined_data["Infection site"] = combined_data.index.map(lambda x: sample_info.at[x.split('_')[0], "Infection site"] if x.split('_')[0] in sample_info.index else np.nan)

# Find highest value samples
combined_data_highest_value = combined_data.copy(deep=True)

for key, df in combined_data_highest_value.groupby("Group"):
    if "KPN" in key:
        for column in ["O1v1-EPA", "O2v2-EPA", "O3b-EPA", "O1v2-HSA", "O2v1-HSA", "O3/O3a-HSA", "O5-HSA", "EPA", "HSA", "MrkA"]:
            highest_value = df[column].max()
            if pd.notna(highest_value):
                highest_value_sample = df[df[column] == highest_value].index[0]
                combined_data_highest_value.loc[df.index.difference([highest_value_sample]), column] = np.nan

# Calculate the fold changes of the samples with the highest values
combined_data_highest_value_foldchange = combined_data_highest_value.copy(deep=True)
for column in ["O1v1-EPA", "O1v2-HSA", "O2v1-HSA", "O2v2-EPA", "O3/O3a-HSA", "O3b-EPA", "O5-HSA", "MrkA","HSA","EPA"] :
    hc_data = combined_data_highest_value[combined_data_highest_value['Group'] == 'HC'][[column, 'Group']]
    hc_median = hc_data[column].median()
    combined_data_highest_value_foldchange[column] = combined_data_highest_value_foldchange[column]/hc_median

# Calculate the fold changes of all the samples   
combined_data_foldchange = combined_data.copy(deep=True)
for column in ["O1v1-EPA", "O1v2-HSA", "O2v1-HSA", "O2v2-EPA", "O3/O3a-HSA", "O3b-EPA", "O5-HSA", "MrkA","HSA","EPA"] :
    hc_data = combined_data[combined_data['Group'] == 'HC'][[column, 'Group']]
    hc_median = hc_data[column].median()
    combined_data_foldchange[column] = combined_data_foldchange[column]/hc_median
    
combined_data.to_excel(result_folder+"combined_data.xlsx")
combined_data_foldchange.to_excel(result_folder+"combined_data_foldchange.xlsx")

#============================================================================================#
#============================================================================================#
# Define functions
def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def smooth(x, y, xgrid):
    samples = np.random.choice(len(x), 50, replace=True)
    y_s = y[samples]
    x_s = x[samples]
    y_sm = sm_lowess(y_s, x_s, frac=4./5., it=5, return_sorted=False)
    y_grid = scipy.interpolate.interp1d(x_s, y_sm, fill_value='extrapolate')(xgrid)
    return y_grid

def bootstrap_ci(data1, data2, alpha=0.05, n_bootstraps=10000):
    bootstrapped_diffs = []
    
    for _ in range(n_bootstraps):
        sample1 = np.random.choice(data1, size=len(data1), replace=True)
        sample2 = np.random.choice(data2, size=len(data2), replace=True)
        bootstrapped_diffs.append(np.median(sample1) - np.median(sample2))
    
    lower = np.percentile(bootstrapped_diffs, 100 * alpha / 2)
    upper = np.percentile(bootstrapped_diffs, 100 * (1 - alpha / 2))
    
    return lower, upper

def mannwhittest_with_ci(df, column, group_column, alpha=0.05):
    groups = df[group_column].unique()
    results = []
    
    for i, first_group in enumerate(groups):
        for second_group in groups[i + 1:]:
            data1 = df[df[group_column] == first_group][column].dropna()
            data2 = df[df[group_column] == second_group][column].dropna()
            
            u_statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            ci_lower, ci_upper = bootstrap_ci(data1, data2, alpha=alpha)
            
            results.append({
                'Group1': first_group,
                'Group2': second_group,
                'U Statistic': u_statistic,
                'P Value': p_value,
                'CI Lower': ci_lower,
                'CI Upper': ci_upper
            })
    
    return pd.DataFrame(results)

#============================================================================================#
#============================================================================================#
# Figure S1A

# Define parameters
final_antigen_list = ["HSA","EPA"]
Antigen_num = len(final_antigen_list)

combined_raw_data = pd.concat([ex1_grouped_data, ex2_grouped_data, ex3_grouped_data, ex4_grouped_data])

# Detect outliers
hc_epa_data = combined_raw_data["EPA"][[index for index in combined_raw_data.index if "HC" in index]]
upper_fence_epa = calculate_upper_fence(hc_epa_data)

# Identify HC and BC samples
HC_list = [index for index in combined_raw_data.index if "HC" in index]
BC_list = [index for index in combined_raw_data.index if "BC" in index]

# Calculate the maximum y value
y_max_num = max(pd.DataFrame.max(combined_raw_data[final_antigen_list]))

# Create the figure
fig = plt.figure(figsize=(7, 4))
subfigs = fig.subfigures(1, Antigen_num)

for outerind, subfig in enumerate(subfigs.flat):
    HC_BC_data = pd.DataFrame(index=HC_list + BC_list)
    HC_BC_data["HC"] = combined_raw_data[final_antigen_list[outerind]][HC_list]
    HC_BC_data["BC"] = combined_raw_data[final_antigen_list[outerind]][BC_list]
    HC_median = combined_raw_data[final_antigen_list[outerind]][HC_list].median()
    BC_median = combined_raw_data[final_antigen_list[outerind]][BC_list].median()
    
    axs = subfig.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [0.8, 3], 'wspace': 0.02, 'left': -0.08})
    q = sns.boxplot(data=HC_BC_data, color="gray", width=0.4, ax=axs[0])
    q.set_ylim(-2000, y_max_num * 1.05)
    
    HC_number = HC_BC_data["HC"].count()
    BC_number = HC_BC_data["BC"].count()
    
    axs[0].text(0.98, 0.98, f"HC($\it{{N}}$={HC_number})", transform=axs[0].transAxes, ha="right", va="top", fontsize=9, color="black")
    axs[0].text(0.98, 0.92, f"BC($\it{{N}}$={BC_number})", transform=axs[0].transAxes, ha="right", va="top", fontsize=9, color="black")
    
    new_df = pd.DataFrame(columns=[final_antigen_list[outerind], "Time point"])
    for index in combined_raw_data.index:
        if "KPN" in index :
            MFI_value = combined_raw_data[final_antigen_list[outerind]][index]
            Time_point = combined_raw_data["Time point"][index]
            if MFI_value >= 0:
                new_df.loc[index] = [MFI_value, Time_point]
    
    KPN_number = 0
    
    for key, df in combined_raw_data.groupby("Group"):
        if "KPN" in key:
            p = sns.lineplot(df["Time point"], df[final_antigen_list[outerind]], color=lighten_color('gray', 1), ax=axs[1], linewidth=0.7, zorder=8, marker="o", markersize=3)
            if df[final_antigen_list[outerind]].count() > 0:
                KPN_number += 1
    
    plt.axhline(y=HC_median, color="black", linestyle='-', linewidth=1, zorder = 8)
    plt.axhline(y=BC_median, color="black", linestyle='--', linewidth=1, zorder = 8)
    if final_antigen_list[outerind] == "EPA" :
        axs[0].axhline(y=upper_fence_epa, color="red", linestyle='dashdot', linewidth=1, zorder = 8)
        axs[1].axhline(y=upper_fence_epa, color="red", linestyle='dashdot', linewidth=1, zorder = 8)

    axs[1].text(0.98, 0.98, f"KPN($\it{{N}}$={KPN_number})", transform=axs[1].transAxes, ha="right", va="top", fontsize=10, color="blue")

    if outerind > 0 :
        q.tick_params(left=False)
        q.set_yticklabels([])
    if outerind == 0 :
        q.set_ylabel(antibody+" MFI", fontsize = 14, weight = 'bold')
    if outerind >= 0 :
        p.set_xlabel(None)
        axs[0].tick_params(labelsize=12)
        axs[1].tick_params(labelsize=12)
        p.tick_params(left=False)

fig.tight_layout()
plt.savefig(result_folder+"Figure S1A.png", dpi=500, transparent=True, bbox_inches='tight')

#============================================================================================#
#============================================================================================#
# Figure S1B

# Define parameters
final_antigen_list = ["HSA","EPA"]
Antigen_num = len(final_antigen_list)

combined_raw_data = pd.concat([ex1_grouped_data, ex2_grouped_data, ex3_grouped_data, ex4_grouped_data])

# Detect outliers
hc_epa_data = combined_raw_data["EPA"][[index for index in combined_raw_data.index if "HC" in index]]
upper_fence_epa = calculate_upper_fence(hc_epa_data)

# Identify HC and BC samples
HC_list = [index for index in combined_data.index if "HC" in index]
BC_list = [index for index in combined_data.index if "BC" in index]

# Calculate the maximum y value
y_max_num = max(pd.DataFrame.max(combined_data[final_antigen_list]))

# Create the figure
fig = plt.figure(figsize=(7, 4))
subfigs = fig.subfigures(1, Antigen_num)

for outerind, subfig in enumerate(subfigs.flat):
    HC_BC_data = pd.DataFrame(index=HC_list + BC_list)
    HC_BC_data["HC"] = combined_data[final_antigen_list[outerind]][HC_list]
    HC_BC_data["BC"] = combined_data[final_antigen_list[outerind]][BC_list]
    HC_median = combined_data[final_antigen_list[outerind]][HC_list].median()
    BC_median = combined_data[final_antigen_list[outerind]][BC_list].median()
    
    axs = subfig.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [0.8, 3], 'wspace': 0.02, 'left': -0.08})
    q = sns.boxplot(data=HC_BC_data, color="gray", width=0.4, ax=axs[0])
    q.set_ylim(-200, y_max_num * 1.3)
    
    HC_number = HC_BC_data["HC"].count()
    BC_number = HC_BC_data["BC"].count()
    
    axs[0].text(0.98, 0.98, f"HC($\it{{N}}$={HC_number})", transform=axs[0].transAxes, ha="right", va="top", fontsize=9, color="black")
    axs[0].text(0.98, 0.92, f"BC($\it{{N}}$={BC_number})", transform=axs[0].transAxes, ha="right", va="top", fontsize=9, color="black")
    
    new_df = pd.DataFrame(columns=[final_antigen_list[outerind], "Time point"])
    for index in combined_data.index:
        if "KPN" in index :
            MFI_value = combined_data[final_antigen_list[outerind]][index]
            Time_point = combined_data["Time point"][index]
            if MFI_value >= 0:
                new_df.loc[index] = [MFI_value, Time_point]
    
    KPN_number = 0
    
    for key, df in combined_data.groupby("Group"):
        if "KPN" in key:
            p = sns.lineplot(df["Time point"], df[final_antigen_list[outerind]], color=lighten_color('gray', 1), ax=axs[1], linewidth=0.7, zorder=8, marker="o", markersize=3)
            if df[final_antigen_list[outerind]].count() > 0:
                KPN_number += 1
    
    plt.axhline(y=HC_median, color="black", linestyle='-', linewidth=1, zorder = 8)
    plt.axhline(y=BC_median, color="black", linestyle='--', linewidth=1, zorder = 8)
    if final_antigen_list[outerind] == "EPA" :
        axs[0].axhline(y=upper_fence_epa, color="red", linestyle='dashdot', linewidth=1, zorder = 8)
        axs[1].axhline(y=upper_fence_epa, color="red", linestyle='dashdot', linewidth=1, zorder = 8)

    axs[1].text(0.98, 0.98, f"KPN($\it{{N}}$={KPN_number})", transform=axs[1].transAxes, ha="right", va="top", fontsize=10, color="blue")

    if outerind > 0 :
        q.tick_params(left=False)
        q.set_yticklabels([])
    if outerind == 0 :
        q.set_ylabel(antibody+" MFI", fontsize = 14, weight = 'bold')
    if outerind >= 0 :
        p.set_xlabel(None)
        axs[0].tick_params(labelsize=12)
        axs[1].tick_params(labelsize=12)
        p.tick_params(left=False)

fig.tight_layout()
plt.savefig(result_folder+"Figure S1B.png", dpi=500, transparent=True, bbox_inches='tight')

#============================================================================================#
#============================================================================================#
# Figure 2C

# Define parameters
final_antigen_list = ["O1v1-EPA", "O1v2-HSA", "O3b-EPA"]
OPS_subtypes = {"O1v1": "O1v2", "O1v2": "O1v1", "O2a": "O2afg", "O2afg": "O2a", "O3/O3a": "O3b", "O3b": "O3/O3a"}
Antigen_num = len(final_antigen_list)

# Identify HC and BC samples
HC_list = [index for index in combined_data.index if "HC" in index]
BC_list = [index for index in combined_data.index if "BC" in index]

# Debugging: Check if final_antigen_list columns exist in combined_data
print("Columns to be used:", final_antigen_list)
print("Existing columns:", combined_data.columns.intersection(final_antigen_list))

# Calculate the maximum y value
y_max_num = max(pd.DataFrame.max(combined_data[final_antigen_list]))

# Create the figure
fig = plt.figure(figsize=(10, 4))
subfigs = fig.subfigures(1, Antigen_num)

for outerind, subfig in enumerate(subfigs.flat):
    HC_BC_data = pd.DataFrame(index=HC_list + BC_list)
    HC_BC_data["HC"] = combined_data[final_antigen_list[outerind]][HC_list]
    HC_BC_data["BC"] = combined_data[final_antigen_list[outerind]][BC_list]
    HC_median = combined_data[final_antigen_list[outerind]][HC_list].median()
    BC_median = combined_data[final_antigen_list[outerind]][BC_list].median()
    O_antigen_name = final_antigen_list[outerind].split("-")[0]
    
    axs = subfig.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [0.8, 3], 'wspace': 0.02, 'left': -0.08})
    q = sns.boxplot(data=HC_BC_data, color="gray", width=0.4, ax=axs[0])
    q.set_ylim(-2000, y_max_num * 1.05)
    
    HC_number = HC_BC_data["HC"].count()
    BC_number = HC_BC_data["BC"].count()
    
    axs[0].text(0.98, 0.98, f"HC($\it{{N}}$={HC_number})", transform=axs[0].transAxes, ha="right", va="top", fontsize=9, color="black")
    axs[0].text(0.98, 0.92, f"BC($\it{{N}}$={BC_number})", transform=axs[0].transAxes, ha="right", va="top", fontsize=9, color="black")
    
    new_df = pd.DataFrame(columns=[final_antigen_list[outerind], "Time point"])
    for index in combined_data.index:
        if "KPN" in index and combined_data["O-type"][index] == O_antigen_name:
            MFI_value = combined_data[final_antigen_list[outerind]][index]
            Time_point = combined_data["Time point"][index]
            if MFI_value >= 0:
                new_df.loc[index] = [MFI_value, Time_point]
    
    subtype_df = pd.DataFrame(columns=[OPS_subtypes[final_antigen_list[outerind].split("-")[0]], "Time point"])
    for index in combined_data.index:
        if "KPN" in index and combined_data["O-type"][index] == OPS_subtypes[O_antigen_name]:
            MFI_value = combined_data[final_antigen_list[outerind]][index]
            Time_point = combined_data["Time point"][index]
            if MFI_value >= 0:
                subtype_df.loc[index] = [MFI_value, Time_point]
    
    KPN_number = 0
    subtype_number = 0
    Non_KPN_number = 0
    for key, df in combined_data.groupby("Group"):
        if "KPN" in key:
            exact_O_antigen = df["O-type"].iloc[0].split("_")[0]
            if exact_O_antigen == final_antigen_list[outerind].split("-")[0]:
                p = sns.lineplot(df["Time point"], df[final_antigen_list[outerind]], color=lighten_color('blue', 1), ax=axs[1], linewidth=0.7, zorder=8, marker="o", markersize=3)
                if df[final_antigen_list[outerind]].count() > 0:
                    KPN_number += 1
            elif exact_O_antigen == OPS_subtypes[final_antigen_list[outerind].split("-")[0]]:
                p = sns.lineplot(df["Time point"], df[final_antigen_list[outerind]], color=lighten_color('orange', 0.7), ax=axs[1], linewidth=0.7, zorder=8, marker="o", markersize=3)
                if df[final_antigen_list[outerind]].count() > 0:
                    subtype_number += 1
            else:
                p = sns.lineplot(df["Time point"], df[final_antigen_list[outerind]], color=lighten_color('gray', 0.7), ax=axs[1], linewidth=0.7, zorder=0)
                if df[final_antigen_list[outerind]].count() > 0:
                    Non_KPN_number += 1

    x = new_df["Time point"]
    y = new_df[final_antigen_list[outerind]]
    xgrid = np.linspace(x.min(), x.max())
    K = 100
    smooths = np.stack([smooth(x, y, xgrid) for k in range(K)]).T
    mean = np.nanmean(smooths, axis=1)
    stderr = np.nanstd(smooths, axis=1, ddof=0)
    axs[1].fill_between(xgrid, mean - 1.96 * stderr, mean + 1.96 * stderr, alpha=0.25)
    axs[1].plot(xgrid, mean, color='#FF0000', zorder=10)
    axs[1].plot(x, y, 'k.')
    
    plt.axhline(y=HC_median, color="black", linestyle='-', linewidth=1)
    plt.axhline(y=BC_median, color="black", linestyle='--', linewidth=1)
    
    axs[1].text(0.98, 0.92, f"{O_antigen_name} KPN($\it{{N}}$={KPN_number})", transform=axs[1].transAxes, ha="right", va="top", fontsize=10, color="blue")
    axs[1].text(0.98, 0.86, f"{OPS_subtypes[final_antigen_list[outerind].split('-')[0]]} KPN($\it{{N}}$={subtype_number})", transform=axs[1].transAxes, ha="right", va="top", fontsize=10, color="orange")
    axs[1].text(0.98, 0.98, f"non-{O_antigen_name},{OPS_subtypes[final_antigen_list[outerind].split('-')[0]]} KPN($\it{{N}}$={Non_KPN_number})", transform=axs[1].transAxes, ha="right", va="top", fontsize=10, color="gray")

    if outerind > 0 :
        q.tick_params(left=False)
        q.set_yticklabels([])
    if outerind == 0 :
        q.set_ylabel(antibody+" MFI", fontsize = 14, weight = 'bold')
    if outerind >= 0 :
        p.set_xlabel(None)
        axs[0].tick_params(labelsize=12)
        axs[1].tick_params(labelsize=12)
        p.tick_params(left=False)

fig.tight_layout()
plt.savefig(result_folder+"Figure 2C.png", dpi=500, transparent=True, bbox_inches='tight')

#============================================================================================#
#============================================================================================#
# Figure S3

# Define parameters
final_antigen_list = ["O2v1-HSA", "O2v2-EPA", "O3/O3a-HSA","O5-HSA"]
Antigen_num = len(final_antigen_list)

# Identify HC and BC samples
HC_list = [index for index in combined_data.index if "HC" in index]
BC_list = [index for index in combined_data.index if "BC" in index]

# Debugging: Check if final_antigen_list columns exist in combined_data
print("Columns to be used:", final_antigen_list)
print("Existing columns:", combined_data.columns.intersection(final_antigen_list))

# Calculate the maximum y value
y_max_num = max(pd.DataFrame.max(combined_data[final_antigen_list]))

# Create the figure
fig = plt.figure(figsize=(13.5, 4))
subfigs = fig.subfigures(1, Antigen_num)

for outerind, subfig in enumerate(subfigs.flat):
    HC_BC_data = pd.DataFrame(index=HC_list + BC_list)
    HC_BC_data["HC"] = combined_data[final_antigen_list[outerind]][HC_list]
    HC_BC_data["BC"] = combined_data[final_antigen_list[outerind]][BC_list]
    HC_median = combined_data[final_antigen_list[outerind]][HC_list].median()
    BC_median = combined_data[final_antigen_list[outerind]][BC_list].median()
    O_antigen_name = final_antigen_list[outerind].split("-")[0]
    
    axs = subfig.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [0.8, 3], 'wspace': 0.02, 'left': -0.08})
    q = sns.boxplot(data=HC_BC_data, color="gray", width=0.4, ax=axs[0])
    q.set_ylim(-2000, y_max_num * 1.05)
    
    HC_number = HC_BC_data["HC"].count()
    BC_number = HC_BC_data["BC"].count()
    
    axs[0].text(0.98, 0.98, f"HC($\it{{N}}$={HC_number})", transform=axs[0].transAxes, ha="right", va="top", fontsize=9, color="black")
    axs[0].text(0.98, 0.92, f"BC($\it{{N}}$={BC_number})", transform=axs[0].transAxes, ha="right", va="top", fontsize=9, color="black")
    
    new_df = pd.DataFrame(columns=[final_antigen_list[outerind], "Time point"])
    for index in combined_data.index:
        if "KPN" in index and combined_data["O-type"][index] == O_antigen_name:
            MFI_value = combined_data[final_antigen_list[outerind]][index]
            Time_point = combined_data["Time point"][index]
            if MFI_value >= 0:
                new_df.loc[index] = [MFI_value, Time_point]
    
    KPN_number = 0
    Non_KPN_number = 0
    for key, df in combined_data.groupby("Group"):
        if "KPN" in key:
            exact_O_antigen = df["O-type"].iloc[0].split("_")[0]
            if exact_O_antigen == final_antigen_list[outerind].split("-")[0]:
                p = sns.lineplot(df["Time point"], df[final_antigen_list[outerind]], color=lighten_color('blue', 1), ax=axs[1], linewidth=0.7, zorder=8, marker="o", markersize=3)
                if df[final_antigen_list[outerind]].count() > 0:
                    KPN_number += 1
            else:
                p = sns.lineplot(df["Time point"], df[final_antigen_list[outerind]], color=lighten_color('gray', 0.7), ax=axs[1], linewidth=0.7, zorder=0)
                if df[final_antigen_list[outerind]].count() > 0:
                    Non_KPN_number += 1

    x = new_df["Time point"]
    y = new_df[final_antigen_list[outerind]]
    xgrid = np.linspace(x.min(), x.max())
    K = 100
    smooths = np.stack([smooth(x, y, xgrid) for k in range(K)]).T
    mean = np.nanmean(smooths, axis=1)
    stderr = np.nanstd(smooths, axis=1, ddof=0)
    axs[1].fill_between(xgrid, mean - 1.96 * stderr, mean + 1.96 * stderr, alpha=0.25)
    axs[1].plot(xgrid, mean, color='#FF0000', zorder=10)
    axs[1].plot(x, y, 'k.')
    
    plt.axhline(y=HC_median, color="black", linestyle='-', linewidth=1)
    plt.axhline(y=BC_median, color="black", linestyle='--', linewidth=1)
    
    axs[1].text(0.98, 0.92, f"{O_antigen_name} KPN($\it{{N}}$={KPN_number})", transform=axs[1].transAxes, ha="right", va="top", fontsize=10, color="blue")
    axs[1].text(0.98, 0.98, f"non-{O_antigen_name} KPN($\it{{N}}$={Non_KPN_number})", transform=axs[1].transAxes, ha="right", va="top", fontsize=10, color="gray")

    if outerind > 0 :
        q.tick_params(left=False)
        q.set_yticklabels([])
    if outerind == 0 :
        q.set_ylabel(antibody+" MFI", fontsize = 14, weight = 'bold')
    if outerind >= 0 :
        p.set_xlabel(None)
        axs[0].tick_params(labelsize=12)
        axs[1].tick_params(labelsize=12)
        p.tick_params(left=False)

fig.tight_layout()
plt.savefig(result_folder+"Figure S3.png", dpi=500, transparent=True, bbox_inches='tight')

#============================================================================================#
#============================================================================================#
# Figure S7

# Create lists for HC and BC data
HC_list = [index for index in combined_data.index if "HC" in index]
BC_list = [index for index in combined_data.index if "BC" in index]

# Define the final antigen list
final_antigen_list = ["O1v1-EPA", "O1v2-HSA", "O2v1-HSA", "O2v2-EPA", "O3/O3a-HSA", "O3b-EPA", "O5-HSA"]

# Plot data for each antigen
for antigen in final_antigen_list:
    fig, axs = plt.subplots(1, len(final_antigen_list), figsize=(18, 3.5), sharey=True)
    
    y_max_num = combined_data[final_antigen_list].max().max()

    # Iterate through each OPS in the final antigen list
    for outerind, ax in enumerate(axs.flat):
        HC_BC_data = pd.DataFrame({
            "Group": ["HC"] * len(HC_list) + ["BC"] * len(BC_list),
            final_antigen_list[outerind]: combined_data.loc[HC_list + BC_list, final_antigen_list[outerind]].values
        })
        
        HC_median = combined_data[final_antigen_list[outerind]][HC_list].median()
        BC_median = combined_data[final_antigen_list[outerind]][BC_list].median()
        
        # Plot the boxplot
        ax.axhline(y=HC_median, color="black", linestyle='-', linewidth=1)
        ax.axhline(y=BC_median, color="black", linestyle='--', linewidth=1)
        ax.set_ylim(-2000, y_max_num * 1.05)
        
        # Prepare data for line plots
        new_df = pd.DataFrame(columns=[final_antigen_list[outerind], "Time point"])
        for index in combined_data.index:
            if "KPN" in index and combined_data["O-type"][index] in antigen:
                MFI_value = combined_data[final_antigen_list[outerind]][index]
                Time_point = combined_data["Time point"][index]
                if MFI_value >= 0:
                    new_df.loc[index] = [MFI_value, Time_point]

        # Plot line plots
        KPNpla_list = list(set(index.split("_")[0] for index in new_df.index))
        KPNpla_number = len(KPNpla_list)
        for key, df in combined_data.groupby("Group"):
            if "KPN" in key and df["O-type"].iloc[0] in antigen:
                sns.lineplot(df["Time point"], y=df[final_antigen_list[outerind]], data=df, color=lighten_color('blue', 1), linewidth=0.7, marker="o", markersize=3, ax=ax)
            else:
                sns.lineplot(df["Time point"], y=df[final_antigen_list[outerind]], data=df, color=lighten_color('gray', 0), linewidth=0.7, zorder = 0, ax=ax)

        # Smooth and plot data
        x = new_df["Time point"]
        y = new_df[final_antigen_list[outerind]]
        if KPNpla_number != 1:
            xgrid = np.linspace(x.min(), x.max())
            smooths = np.stack([smooth(x, y, xgrid) for k in range(100)]).T
            mean = np.nanmean(smooths, axis=1)
            stderr = np.nanstd(smooths, axis=1, ddof=0)
            ax.fill_between(xgrid, mean - 1.96 * stderr, mean + 1.96 * stderr, alpha=0.25)
            ax.plot(xgrid, mean, color='#FF0000', zorder=10)

        # Annotate the number of KPN isolates
        ax.text(0.02, 0.98, f'$\it{{N}}$={KPNpla_number}', transform=ax.transAxes, ha="left", va="top", fontsize=14, color="black")
        ax.tick_params(labelsize=12)
        ax.set_xlabel('')
        if outerind == 0:
            ax.set_ylabel(antibody+" MFI", fontsize=14, weight='bold')
        else :
            ax.tick_params(left=False)
        
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    
    # Save the figure
    antigen_name = antigen.replace("/", "")
    plt.savefig(result_folder+f"Figure S7_{antigen_name}_Kpn.png", dpi=500, transparent=True, bbox_inches='tight')

#============================================================================================#
#============================================================================================#
# Figure S6A

# Filter the data for O1v1 and O1v2 O-types
filtered_data = combined_data[combined_data['O-type'].isin(['O1v1', 'O1v2'])]

# Define the final antigen list
final_antigen_list = ["O1v1-EPA", "O1v2-HSA", "MrkA"]

# Calculate the maximum y value for setting y-axis limits
y_max_num = filtered_data[final_antigen_list].max().max()

# Create figure and subfigures
fig = plt.figure(figsize=(10, 4))
subfigs = fig.subfigures(1, len(final_antigen_list))

# Plot data for each antigen
for outerind, subfig in enumerate(subfigs.flat):
    
    hc_data = combined_data[combined_data['Group'] == 'HC'][[final_antigen_list[outerind], 'Group']]
    bc_data = combined_data[combined_data['Group'] == 'BC'][[final_antigen_list[outerind], 'Group']]
    combined_hc_bc = pd.concat([hc_data, bc_data])

    hc_median = hc_data.median()
    bc_median = bc_data.median()
    
    axs = subfig.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [0.8, 3], 'wspace': 0.02,'left':-0.08})
    sns.boxplot(data=combined_hc_bc, x='Group', y=final_antigen_list[outerind], ax=axs[0], color="gray", width=0.4)
    axs[0].set_ylim(-2000, y_max_num * 1.05)
    axs[0].set_ylabel("IgG MFI", fontsize=14, weight='bold')
    axs[0].tick_params(labelsize=12)
    
    hc_number = hc_data.count()[0]
    bc_number = bc_data.count()[0]
    
    axs[0].text(0.98, 0.98, f"HC($\\it{{N}}$={hc_number})", transform=axs[0].transAxes, ha="right", va="top", fontsize=9, color="black")
    axs[0].text(0.98, 0.92, f"BC($\\it{{N}}$={bc_number})", transform=axs[0].transAxes, ha="right", va="top", fontsize=9, color="black")
    axs[0].set_xlabel(None)
    
    condition_groups = filtered_data.groupby('Condition')
    
    I_count, NonI_count = 0, 0
    for condition, group in condition_groups:
        color = 'red' if condition == "I" else 'blue'
        new_df = group[group[final_antigen_list[outerind]] >= 0][['Time point', final_antigen_list[outerind]]]
        
        for key, df in group.groupby("Group") :
            sns.lineplot(x='Time point', y=final_antigen_list[outerind], data=df, ax=axs[1], color=lighten_color(color, 0.3), linewidth=0.7, marker='o', markersize=3, zorder=0)
            if df[final_antigen_list[outerind]].count()>0 and condition == "I" :
                I_count += 1
            elif df[final_antigen_list[outerind]].count()>0 and condition == "Non-I" :
                NonI_count += 1
        
        x = new_df['Time point']
        y = new_df[final_antigen_list[outerind]]
        xgrid = np.linspace(x.min(), x.max())
        smooths = np.stack([smooth(x, y, xgrid) for k in range(100)]).T
        mean = np.nanmean(smooths, axis=1)
        stderr = np.nanstd(smooths, axis=1, ddof=0)
        axs[1].fill_between(xgrid, mean - 1.96 * stderr, mean + 1.96 * stderr, color=color, alpha=0.25)
        axs[1].plot(xgrid, mean, color=color, zorder=10)
    
    axs[1].axhline(y=hc_median.item(), color="black", linestyle='-', linewidth=1)
    axs[1].axhline(y=bc_median.item(), color="black", linestyle='--', linewidth=1)
    axs[1].tick_params(labelsize=12)
    axs[1].set_xlabel(None)
    
    axs[1].text(0.98, 0.98, f"I ($\\it{{N}}$={I_count})", transform=axs[1].transAxes, ha="right", va="top", fontsize=10, color="red")
    axs[1].text(0.98, 0.92, f"Non-I ($\\it{{N}}$={NonI_count})", transform=axs[1].transAxes, ha="right", va="top", fontsize=10, color="blue")

    if outerind > 0:
        axs[0].tick_params(left=False)
        axs[0].set_ylabel('')
        axs[1].set_ylabel('')
        axs[1].tick_params(left=False)
        axs[1].set_yticklabels([])
        
    if outerind == 0:
        axs[0].set_ylabel(antibody+" MFI", fontsize=14, weight='bold')
        axs[1].tick_params(left=False)

fig.tight_layout()
plt.savefig(result_folder+"Figure S6A.png", dpi=300, transparent=True, bbox_inches='tight')

#============================================================================================#
#============================================================================================#
# Figure S6B

# Filter the data for O1v1 and O1v2 O-types
filtered_data2 = combined_data_highest_value_foldchange[combined_data_highest_value_foldchange['O-type'].isin(['O1v1', 'O1v2'])]

# Define the final antigen list
final_antigen_list = ["O1v1-EPA", "O1v2-HSA", "MrkA"]

# Calculate the maximum y value for setting y-axis limits
y_max_num = filtered_data2[final_antigen_list].max().max()

# Define colors for the box plots
box_colors = ['tab:orange', 'tab:blue', 'tab:olive']

# Create figure and subfigures
fig, axs = plt.subplots(1, len(final_antigen_list), figsize=(4, 3), sharey=True, gridspec_kw={'wspace': 0.05, 'left': 0.05})

# Plot data for each antigen
for outerind, ax in enumerate(axs.flat):
    # Filter data and create a combined column 'Group_Condition' on the fly
    hc_data = combined_data_highest_value_foldchange[combined_data_highest_value_foldchange['Group'] == 'HC'][[final_antigen_list[outerind]]]
    hc_data['Group_Condition'] = 'HC'

    bc_data = combined_data_highest_value_foldchange[combined_data_highest_value_foldchange['Group'] == 'BC'][[final_antigen_list[outerind]]]
    bc_data['Group_Condition'] = 'BC'

    condition_data = combined_data_highest_value_foldchange[combined_data_highest_value_foldchange['O-type'].isin(['O1v1', 'O1v2'])][[final_antigen_list[outerind], 'Condition']]
    condition_data['Group_Condition'] = condition_data['Condition']

    # Drop the original 'Condition' column as it's now redundant
    condition_data = condition_data.drop(columns='Condition')

    # Concatenate the dataframes
    combined_hc_bc_kpn = pd.concat([hc_data, bc_data, condition_data])

    order = ['HC', 'BC', 'I', 'Non-I']
    
    sns.boxplot(data=combined_hc_bc_kpn, x='Group_Condition', y=final_antigen_list[outerind], order=order, ax=ax, color=box_colors[outerind] , width=0.55, showfliers=False)
    sns.swarmplot(data=combined_hc_bc_kpn, x='Group_Condition', y=final_antigen_list[outerind], order=order, color="black", size=2, ax=ax)
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,fontsize = 10,ha="right",rotation_mode="anchor")
    ax.set_ylim(-1, y_max_num * 1.05)
    
    # Perform Mann-Whitney U test and calculate 95% confidence intervals
    result_df = mannwhittest_with_ci(combined_hc_bc_kpn, column=final_antigen_list[outerind], group_column='Group_Condition')

    # Display the results
    print(result_df)
    
    if outerind > 0:
        ax.tick_params(left=False)
        ax.set_ylabel('')
    else:
        ax.set_ylabel(antibody+" fold change", fontsize=14, weight='bold')
    
fig.tight_layout()
plt.savefig(result_folder+"Figure S6B.png", dpi=300, transparent=True, bbox_inches='tight')

#============================================================================================#
#============================================================================================#
# Figure 4B

# Define the final antigen list
final_antigen_list = ["O1v1-EPA", "O1v2-HSA", "O3/O3a-HSA","O3b-EPA"]

# Define colors for the box plots
box_colors = ['tab:orange', 'tab:blue', 'tab:purple','tab:brown']

# Create figure and subfigures
fig, axs = plt.subplots(1, len(final_antigen_list), figsize=(11, 4), sharey=False, gridspec_kw={'wspace': 0.24, 'left': -0.08})

# Plot data for each antigen
for outerind, ax in enumerate(axs.flat):
    # Filter data and create a combined column 'Classification' on the fly
    hc_data = combined_data_highest_value_foldchange[combined_data_highest_value_foldchange['Group'] == 'HC'][[final_antigen_list[outerind]]]
    hc_data['Classification'] = 'HC'

    bc_data = combined_data_highest_value_foldchange[combined_data_highest_value_foldchange['Group'] == 'BC'][[final_antigen_list[outerind]]]
    bc_data['Classification'] = 'BC'

    kpn_data = combined_data_highest_value_foldchange[combined_data_highest_value_foldchange.index.str.contains("KPN")][[final_antigen_list[outerind], 'O-type']]
    kpn_data['Classification'] = kpn_data['O-type']

    # Drop the original 'O-type' column as it's now redundant
    kpn_data = kpn_data.drop(columns='O-type')

    # Concatenate the dataframes
    combined_hc_bc_kpn = pd.concat([hc_data, bc_data, kpn_data])

    # Classify as "Other" if not in the specified list
    valid_classifications = ["HC","BC","O1v1", "O1v2", "O2v1", "O2v2", "O3/O3a", "O3b", "O5"]
    combined_hc_bc_kpn['Classification'] = combined_hc_bc_kpn['Classification'].apply(
        lambda x: x if x in valid_classifications else "Other"
    )
    
    # Order for the boxplot
    order = valid_classifications + ['Other']
    
    # Plot the data
    sns.boxplot(data=combined_hc_bc_kpn, x='Classification', y=final_antigen_list[outerind], order=order, ax=ax, color=box_colors[outerind], width=0.6, showfliers=False)
    sns.swarmplot(data=combined_hc_bc_kpn, x='Classification', y=final_antigen_list[outerind], order=order, color="black", size=2, ax=ax)
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10, ha="right", rotation_mode="anchor")
    
    # Perform Mann-Whitney U test and calculate 95% confidence intervals
    result_df = mannwhittest_with_ci(combined_hc_bc_kpn, column=final_antigen_list[outerind], group_column='Classification')

    # Display the results
    print(result_df)
    
    if outerind > 0:
        ax.set_ylabel('')
    else:
        ax.set_ylabel(antibody+" fold change", fontsize=14, weight='bold')

fig.tight_layout()
plt.savefig(result_folder+"Figure 4B.png", dpi=300, transparent=True, bbox_inches='tight')

#============================================================================================#
#============================================================================================#
# Figure S8

# Define the final antigen list
final_antigen_list = ["O2v1-HSA", "O2v2-EPA", "O5-HSA","MrkA"]

# Define colors for the box plots
box_colors = ['tab:green', 'tab:red', 'tab:pink','tab:olive']

# Create figure and subfigures
fig, axs = plt.subplots(1, len(final_antigen_list), figsize=(11, 4), sharey=False, gridspec_kw={'wspace': 0.24, 'left': -0.08})

# Plot data for each antigen
for outerind, ax in enumerate(axs.flat):
    # Filter data and create a combined column 'Classification' on the fly
    hc_data = combined_data_highest_value_foldchange[combined_data_highest_value_foldchange['Group'] == 'HC'][[final_antigen_list[outerind]]]
    hc_data['Classification'] = 'HC'

    bc_data = combined_data_highest_value_foldchange[combined_data_highest_value_foldchange['Group'] == 'BC'][[final_antigen_list[outerind]]]
    bc_data['Classification'] = 'BC'

    kpn_data = combined_data_highest_value_foldchange[combined_data_highest_value_foldchange.index.str.contains("KPN")][[final_antigen_list[outerind], 'O-type']]
    kpn_data['Classification'] = kpn_data['O-type']

    # Drop the original 'O-type' column as it's now redundant
    kpn_data = kpn_data.drop(columns='O-type')

    # Concatenate the dataframes
    combined_hc_bc_kpn = pd.concat([hc_data, bc_data, kpn_data])

    # Classify as "Other" if not in the specified list
    valid_classifications = ["HC","BC","O1v1", "O1v2", "O2v1", "O2v2", "O3/O3a", "O3b", "O5"]
    combined_hc_bc_kpn['Classification'] = combined_hc_bc_kpn['Classification'].apply(
        lambda x: x if x in valid_classifications else "Other"
    )
    
    # Order for the boxplot
    order = valid_classifications + ['Other']
    
    # Plot the data
    sns.boxplot(data=combined_hc_bc_kpn, x='Classification', y=final_antigen_list[outerind], order=order, ax=ax, color=box_colors[outerind], width=0.6, showfliers=False)
    sns.swarmplot(data=combined_hc_bc_kpn, x='Classification', y=final_antigen_list[outerind], order=order, color="black", size=2, ax=ax)
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10, ha="right", rotation_mode="anchor")
    
    # Perform Mann-Whitney U test and calculate 95% confidence intervals
    result_df = mannwhittest_with_ci(combined_hc_bc_kpn, column=final_antigen_list[outerind], group_column='Classification')

    # Display the results
    print(result_df)
    
    if outerind > 0:
        ax.set_ylabel('')
    else:
        ax.set_ylabel(antibody+" fold change", fontsize=14, weight='bold')

fig.tight_layout()
plt.savefig(result_folder+"Figure S8.png", dpi=300, transparent=True, bbox_inches='tight')

#============================================================================================#
#============================================================================================#
# Figure 3

# Define the final antigen list
final_antigen_list = ["O1v1-EPA", "O1v2-HSA", "O2v1-HSA", "O2v2-EPA", "O3/O3a-HSA", "O3b-EPA", "O5-HSA", "MrkA"]

# Define colors for the box plots
box_colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']

# Create figure and subfigures
fig, axs = plt.subplots(1, len(final_antigen_list), figsize=(8, 2.5), sharey=False, gridspec_kw={'wspace': 0.7, 'left': -0.08})

# Plot data for each antigen
for outerind, ax in enumerate(axs.flat):
    if final_antigen_list[outerind] == "MrkA":
        bc_data = combined_data_highest_value_foldchange[combined_data_highest_value_foldchange['Group'] == 'BC'][[final_antigen_list[outerind]]]
        bc_data['Classification'] = 'BC'
        bc_data['Condition'] = 'Non-I'
        
        kpn_data = combined_data_highest_value_foldchange[combined_data_highest_value_foldchange.index.str.contains("KPN")][[final_antigen_list[outerind]]]
        kpn_data['Classification'] = "KPN"
        kpn_data['Condition'] = combined_data_highest_value_foldchange[combined_data_highest_value_foldchange.index.str.contains("KPN")]['Condition']
    else:
        bc_data = combined_data_highest_value_foldchange[combined_data_highest_value_foldchange['Group'] == 'BC'][[final_antigen_list[outerind]]]
        bc_data['Classification'] = 'BC'
        bc_data['Condition'] = 'Non-I'
    
        OPS_name = final_antigen_list[outerind].split("-")[0]
    
        kpn_data = combined_data_highest_value_foldchange[combined_data_highest_value_foldchange['O-type'] == OPS_name][[final_antigen_list[outerind], 'O-type']]
        kpn_data['Classification'] = kpn_data['O-type']
        kpn_data['Condition'] = combined_data_highest_value_foldchange[combined_data_highest_value_foldchange['O-type'] == OPS_name]['Condition']
        kpn_data = kpn_data.drop(columns='O-type')
    
    # Concatenate the dataframes
    combined_bc_kpn = pd.concat([bc_data, kpn_data])
    
    sns.boxplot(data=combined_bc_kpn, x='Classification', y=final_antigen_list[outerind], width=0.6, showfliers=False, ax=ax, color=box_colors[outerind])
    sns.swarmplot(data=combined_bc_kpn, x='Classification', hue="Condition", y=final_antigen_list[outerind], palette={"I": 'red', "Non-I": 'black'}, size=2, ax=ax)
    ax.get_legend().set_visible(False)
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(labelsize=12)

    if outerind > 0:
        ax.set_ylabel('')
    else:
        ax.set_ylabel(antibody+" fold change", fontsize=12, weight='bold')

fig.tight_layout()
plt.savefig(result_folder+"Figure 3.png", dpi=300, transparent=True, bbox_inches='tight')

#============================================================================================#
#============================================================================================#
# Figure 4A

# Prepare the data for regression analysis
df_regression = combined_data.copy(deep=True)
df_regression.dropna(subset=["EPA"], inplace=True)

# Filter data to include only HC and BC groups, and the lowest EPA for other groups
filtered_df_regression = pd.DataFrame(columns=df_regression.columns)
for group, df in df_regression.groupby("Group"):
    if group in ["HC", "BC"]:
        filtered_df_regression = pd.concat([filtered_df_regression, df])
    else:
        lowest_EPA_idx = df["EPA"].idxmin()
        filtered_df_regression.loc[lowest_EPA_idx] = df.loc[lowest_EPA_idx]

# Further filter the data
filtered_df_regression_EPA = filtered_df_regression[filtered_df_regression["EPA"] < 200]
filtered_df_regression_EPA_HSA = filtered_df_regression_EPA[filtered_df_regression_EPA["HSA"] < 200]
filtered_df_regression_EPA_HSA = filtered_df_regression_EPA_HSA[["O1v1-EPA", "O1v2-HSA", "O2v1-HSA", "O2v2-EPA", "O3/O3a-HSA", "O3b-EPA", "O5-HSA"]].astype(float)

# Define antigen list
antigen_list = ["O1v1-EPA", "O1v2-HSA", "O2v1-HSA", "O2v2-EPA", "O3/O3a-HSA", "O3b-EPA", "O5-HSA"]

# Initialize dataframes for correlation coefficients and p-values
correlation_df = pd.DataFrame(index=antigen_list, columns=antigen_list, dtype="float")
pvalue_df = pd.DataFrame(index=antigen_list, columns=antigen_list, dtype="float")
color_df = pd.DataFrame(index=range(len(antigen_list)*len(antigen_list)), columns=["Antigen1", "Antigen2", "p-value"], dtype="float")

# Compute correlation coefficients and p-values for each pair of antigens
num = 0
for antigen1 in antigen_list:
    for antigen2 in antigen_list:
        corr, p = spearmanr(filtered_df_regression_EPA_HSA[antigen1], filtered_df_regression_EPA_HSA[antigen2])
        correlation_df[antigen1][antigen2] = corr
        pvalue_df[antigen1][antigen2] = p
        color_df.loc[num] = [antigen1, antigen2, p]
        
        # Bootstrap confidence intervals for the Spearman correlation
        n_bootstraps = 1000
        bootstrap_correlations = []
        for _ in range(n_bootstraps):
            indices = np.random.choice(len(filtered_df_regression_EPA_HSA[antigen1]), len(filtered_df_regression_EPA_HSA[antigen1]), replace=True)
            x_bootstrap = filtered_df_regression_EPA_HSA[antigen1].iloc[indices]
            y_bootstrap = filtered_df_regression_EPA_HSA[antigen2].iloc[indices]
            bootstrap_corr, _ = spearmanr(x_bootstrap, y_bootstrap)
            bootstrap_correlations.append(bootstrap_corr)
        ci_lower = np.percentile(bootstrap_correlations, 2.5)
        ci_upper = np.percentile(bootstrap_correlations, 97.5)
        print(f'{antigen1.split("-")[0]} - {antigen2.split("-")[0]}\n95% Confidence Interval: [{ci_lower}, {ci_upper}]\nCorrelation value and P-value : [{corr}, {p}]\n{"="*60}')
        num += 1

# Rename columns for better readability
antigen_rename_dict = {
    "O1v1-EPA": "O1v1", "O1v2-HSA": "O1v2", "O2v1-HSA": "O2v1",
    "O2v2-EPA": "O2v2", "O3/O3a-HSA": "O3/O3a", "O3b-EPA": "O3b", "O5-HSA": "O5"
}
color_df.replace(antigen_rename_dict, inplace=True)
correlation_df.rename(columns=antigen_rename_dict, index=antigen_rename_dict, inplace=True)
pvalue_df.rename(columns=antigen_rename_dict, index=antigen_rename_dict, inplace=True)

# Annotate significant correlations
annotation_data = pvalue_df.copy()
for col in annotation_data.columns:
    for idx in annotation_data.index:
        p = pvalue_df.loc[idx, col]
        corr_value = correlation_df.loc[idx, col]
        if p < 0.05:
            if corr_value > 0.999:
                annotation_data.loc[idx, col] = f"{corr_value:.0f}"
            else:
                annotation_data.loc[idx, col] = f"{corr_value:.2f}"
        else:
            annotation_data.loc[idx, col] = ""

# Plot the heatmap
sns.set(font_scale=1.8)
plt.figure(figsize=(7, 5))

mask = np.triu(np.ones_like(correlation_df), k=1)
colormap = sns.color_palette("vlag", as_cmap=True)

g = sns.heatmap(correlation_df, annot=annotation_data, annot_kws={"size": 15, "color":"black"}, vmax=1, vmin=-1, cmap=colormap, mask=mask, fmt='s')
g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=15, weight='bold')
g.set_xticklabels(g.get_xticklabels(), rotation=45, fontsize=15, weight='bold')

plt.tight_layout()
plt.savefig(result_folder+"Figure 4A.png", dpi=300, transparent=True, bbox_inches='tight')

#============================================================================================#
#============================================================================================#
# Figure S5

# Load the O1Kpn_capsule data
O1Kpn_capsule = pd.read_excel(base_path+"O1Kpn_capsule.xlsx")
O1Kpn_capsule.set_index("KPN", inplace=True)

# Initialize dataframes for O1v1-EPA and O1v2-HSA
O1Kpn_O1v1EPA_df = pd.DataFrame(index=O1Kpn_capsule.index, columns=["O1v1-EPA", "Glucuronic_acid"], dtype=float)
O1Kpn_O1v2HSA_df = pd.DataFrame(index=O1Kpn_capsule.index, columns=["O1v2-HSA", "Glucuronic_acid"], dtype=float)

# Populate the dataframes with maximum values from combined_data
for key, df in combined_data.groupby("Group"):
    if key in ["HC", "BC"]:
        continue
    if key in O1Kpn_capsule.index:
        O1Kpn_O1v1EPA_df.loc[key, "O1v1-EPA"] = df["O1v1-EPA"].max()
        O1Kpn_O1v1EPA_df.loc[key, "Glucuronic_acid"] = O1Kpn_capsule.loc[key, "Glucuronic_acid"]
        O1Kpn_O1v2HSA_df.loc[key, "O1v2-HSA"] = df["O1v2-HSA"].max()
        O1Kpn_O1v2HSA_df.loc[key, "Glucuronic_acid"] = O1Kpn_capsule.loc[key, "Glucuronic_acid"]

# Calculate the upper fences for HC data
O1v1EPA_hc_data = combined_data.loc[combined_data.index.str.contains("HC"), "O1v1-EPA"]
O1v2HSA_hc_data = combined_data.loc[combined_data.index.str.contains("HC"), "O1v2-HSA"]
O1v1EPA_upper_fence = calculate_upper_fence(O1v1EPA_hc_data)
O1v2HSA_upper_fence = calculate_upper_fence(O1v2HSA_hc_data)

# Drop rows with values below the upper fence
O1Kpn_O1v1EPA_df = O1Kpn_O1v1EPA_df[O1Kpn_O1v1EPA_df["O1v1-EPA"] > O1v1EPA_upper_fence]
O1Kpn_O1v2HSA_df = O1Kpn_O1v2HSA_df[O1Kpn_O1v2HSA_df["O1v2-HSA"] > O1v2HSA_upper_fence]

# Drop any remaining NaN values
O1Kpn_O1v1EPA_df.dropna(inplace=True)
O1Kpn_O1v2HSA_df.dropna(inplace=True)

# Function to plot Spearman correlation and bootstrap confidence intervals
def plot_spearman_correlation(df, x_col, y_col, title, filename):
    sns.set_style('ticks')
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    correlation, p_value = spearmanr(df[x_col], df[y_col])
    print(f'Spearman Correlation: {correlation}, P-value: {p_value}')

    n_bootstraps = 1000
    bootstrap_correlations = []
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(len(df[x_col]), len(df[x_col]), replace=True)
        x_bootstrap = df[x_col].iloc[indices]
        y_bootstrap = df[y_col].iloc[indices]
        bootstrap_corr, _ = spearmanr(x_bootstrap, y_bootstrap)
        bootstrap_correlations.append(bootstrap_corr)

    bootstrap_correlations = np.array(bootstrap_correlations)
    ci_lower = np.percentile(bootstrap_correlations, 2.5)
    ci_upper = np.percentile(bootstrap_correlations, 97.5)
    print(f'95% Confidence Interval: [{ci_lower}, {ci_upper}]')

    sns.scatterplot(x=x_col, y=y_col, data=df, color='orange', zorder=10, s=20, marker='o', ax=ax)
    ax.set_title(f'Correlation: {correlation:.2f} \nP-value: {p_value:.2f}\n95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]')
    ax.text(0.02, 0.98, f'$\\it{{N}}$ = {len(df)}', transform=ax.transAxes, ha="left", va="top", fontsize=14, color="black")

    ax.set_xlabel("Glucuronic acid (μg/ml)", size=14, weight='bold')
    ax.set_ylabel(antibody+" MFI", size=14, weight='bold')
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.tick_params(labelsize = 12)    

    plt.tight_layout()
    plt.savefig(result_folder+filename, transparent=True, dpi=300, bbox_inches='tight')

# Plot for O1v1-EPA
plot_spearman_correlation(O1Kpn_O1v1EPA_df, "Glucuronic_acid", "O1v1-EPA", "O1v1-EPA Correlation", "Figure S5_O1v1-EPA.png")

# Plot for O1v2-HSA
plot_spearman_correlation(O1Kpn_O1v2HSA_df, "Glucuronic_acid", "O1v2-HSA", "O1v2-HSA Correlation", "Figure S5_O1v2-HSA.png")

#============================================================================================#
#============================================================================================#
# Figure S4

# Mapping dictionaries
o_antigen_to_epa_hsa = {
    "O1v1": "O1v1-EPA",
    "O1v2": "O1v2-HSA",
    "O2v1": "O2v1-HSA",
    "O2v2": "O2v2-EPA",
    "O3/O3a": "O3/O3a-HSA",
    "O3b": "O3b-EPA",
    "O5": "O5-HSA"
}

# Initialize DataFrame
infection_site_MFI_df = pd.DataFrame(columns=["KPN", "MFI to homologous OPS", "O-type", "Infection site"])

# Populate the DataFrame
for key, df in combined_data_foldchange.groupby("Group"):
    if df["Infection site"].iloc[0] == "Unknown" :
        continue
    elif df["O-type"].iloc[0] in  o_antigen_to_epa_hsa:
        sample_name = key
        o_type = df["O-type"].iloc[0]
        mfi = df[o_antigen_to_epa_hsa[o_type]].max()
        infection_site = df["Infection site"].iloc[0]
        
        # Append row to the DataFrame
        infection_site_MFI_df = infection_site_MFI_df.append({
            "KPN": sample_name,
            "MFI to homologous OPS": mfi,
            "O-type": o_type,
            "Infection site": infection_site
        }, ignore_index=True)

infection_site_MFI_df.set_index("KPN",inplace=True)
infection_site_MFI_df.dropna(inplace=True)

# Reset to default Matplotlib style
mpl.rc_file_defaults()

# Plot for each infection site
fig, axs = plt.subplots(1,1,figsize=(5.5,5.5))
hue_order = ["O1v1", "O1v2", "O2v1", "O2v2", "O3/O3a", "O3b", "O5"] 
p=sns.boxplot(x = "Infection site", y= "MFI to homologous OPS", data=infection_site_MFI_df,color="gray", width=0.6,showfliers=False)
q = sns.stripplot(x="Infection site", y="MFI to homologous OPS", hue="O-type", data=infection_site_MFI_df,
                  palette={"O1v1": "tab:orange", "O1v2": "tab:blue", "O2v1": "tab:green", "O2v2": "tab:red",
                           "O3/O3a": "tab:purple", "O3b": "tab:brown", "O5": "tab:pink"},
                  size=4, dodge=True, alpha=0.8, edgecolor="black", linewidth=0.5, hue_order=hue_order, jitter=0.4)

custom_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:orange', markeredgecolor='black', markersize=8, label='O1v1'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markeredgecolor='black', markersize=8, label='O1v2'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:green', markeredgecolor='black', markersize=8, label='O2v1'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:red', markeredgecolor='black', markersize=8, label='O2v2'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:purple', markeredgecolor='black', markersize=8, label='O3/O3a'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:brown', markeredgecolor='black', markersize=8, label='O3b'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:pink', markeredgecolor='black', markersize=8, label='O5')
]

axs.legend(handles=custom_handles, title="Kpn infection", loc='upper left', bbox_to_anchor=(1.05, 1))

# Get current x-tick labels
current_labels = [tick.get_text() for tick in axs.get_xticklabels()]

# Define new labels based on a condition
new_labels = []
for label in current_labels:
    if label == "Urinary":
        new_labels.append("U")
    elif label == "Respiratory":
        new_labels.append("R")
    elif label == "Gastrointestinal/Abdominal/Biliary": 
        new_labels.append("G")

# Apply new labels
axs.set_xticklabels(new_labels)

p.set_ylabel(antibody+" fold change",size = 14)
p.set_xlabel("")
axs.tick_params(axis='y', labelsize=12)
axs.tick_params(axis='x', labelsize=14)
plt.tight_layout()
plt.savefig(result_folder+"Figure S4.png",transparent=True, dpi=300,bbox_inches='tight')
#============================================================================================#
#============================================================================================#