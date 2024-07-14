import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

class TrimData1:
    def __init__(self, mfi_data_file, bead_data_file):
        self.mfi_data_file = mfi_data_file
        self.bead_data_file = bead_data_file
        self.mfi_data = pd.read_excel(self.mfi_data_file)
        self.bead_data = pd.read_excel(self.bead_data_file)
        
        # Ensure each duplicated sample has a unique identifier
        self.mfi_data['Sample'] = self.mfi_data.groupby('Sample').cumcount().astype(str).radd(self.mfi_data['Sample'] + '_')
        self.bead_data['Sample'] = self.bead_data.groupby('Sample').cumcount().astype(str).radd(self.bead_data['Sample'] + '_')

        self.mfi_data.set_index('Sample', inplace=True)
        self.bead_data.set_index('Sample', inplace=True)

    def trim_data_by_bead_count(self, threshold):
        trimmed_mfi_data = self.mfi_data.copy()

        for sample in self.bead_data.index:
            bead_count_epa = self.bead_data.at[sample, 'EPA'] if 'EPA' in self.bead_data.columns else threshold + 1
            bead_count_hsa = self.bead_data.at[sample, 'HSA'] if 'HSA' in self.bead_data.columns else threshold + 1

            if bead_count_epa < threshold and bead_count_hsa < threshold:
                # Remove all values except "MrkA"
                for column in trimmed_mfi_data.columns:
                    if column != "MrkA":
                        trimmed_mfi_data.at[sample, column] = np.nan

            elif bead_count_epa < threshold:
                # Remove all columns with "EPA"
                for column in trimmed_mfi_data.columns:
                    if "EPA" in column:
                        trimmed_mfi_data.at[sample, column] = np.nan

            elif bead_count_hsa < threshold:
                # Remove all columns with "HSA"
                for column in trimmed_mfi_data.columns:
                    if "HSA" in column:
                        trimmed_mfi_data.at[sample, column] = np.nan

            # For other columns, if the bead count is below the threshold, remove only the specific value
            for column in self.bead_data.columns:
                if self.bead_data.at[sample, column] < threshold:
                    trimmed_mfi_data.at[sample, column] = np.nan

        return trimmed_mfi_data

    def subtract_background_and_calculate_stats(self, data, antibody):
        blank_rows = data.loc[data.index.str.startswith(f"Blank_{antibody}")]
        
        if blank_rows.empty:
            print(f"Warning: No rows starting with 'Blank_{antibody}' found in data index. Background subtraction cannot be performed.")
            return data, None, None, None, None, None, None
        
        blank_mean = blank_rows.mean()
        net_mfi = data - blank_mean
        net_mfi.drop(blank_rows.index, inplace=True)
        
        net_mfi[net_mfi < 1] = 1
        
        # Separate data into SC samples and non-SC samples
        sc_samples = net_mfi[net_mfi.index.str.contains("SC")]
        non_sc_samples = net_mfi[~net_mfi.index.str.contains("SC")]

        # Extract the base sample names by removing the unique identifiers
        non_sc_samples['Base_Sample'] = non_sc_samples.index.str.replace(r'_\d+$', '', regex=True)
        sc_samples['Base_Sample'] = sc_samples.index.str.replace(r'_\d+$', '', regex=True)
        
        # Store the order of the base samples
        non_sc_base_sample_order = non_sc_samples['Base_Sample'].unique()
        sc_base_sample_order = sc_samples['Base_Sample'].unique()
        
        # Calculate the average, std, and CV for each base sample without SC
        avg_net_mfi = non_sc_samples.groupby('Base_Sample').mean()
        std_net_mfi = non_sc_samples.groupby('Base_Sample').std()
        cv_net_mfi = 100 * (std_net_mfi / avg_net_mfi)
        
        # Calculate the average, std, and CV for each base sample with SC
        avg_net_mfi_SC = sc_samples.groupby('Base_Sample').mean()
        std_net_mfi_SC = sc_samples.groupby('Base_Sample').std()
        cv_net_mfi_SC = 100 * (std_net_mfi_SC / avg_net_mfi_SC)
        
        # Restore the order of the samples
        avg_net_mfi = avg_net_mfi.reindex(non_sc_base_sample_order)
        std_net_mfi = std_net_mfi.reindex(non_sc_base_sample_order)
        cv_net_mfi = cv_net_mfi.reindex(non_sc_base_sample_order)
        
        avg_net_mfi_SC = avg_net_mfi_SC.reindex(sc_base_sample_order)
        std_net_mfi_SC = std_net_mfi_SC.reindex(sc_base_sample_order)
        cv_net_mfi_SC = cv_net_mfi_SC.reindex(sc_base_sample_order)
        
        # Drop the auxiliary columns
        non_sc_samples.drop(columns=['Base_Sample'], inplace=True)
        sc_samples.drop(columns=['Base_Sample'], inplace=True)
        
        return avg_net_mfi, std_net_mfi, cv_net_mfi, avg_net_mfi_SC, std_net_mfi_SC, cv_net_mfi_SC

    
    
class TrimData2:
    def __init__(self, trimmed_avg_mfi_data, sample_info_file):
        self.trimmed_avg_mfi_data = trimmed_avg_mfi_data
        self.sample_info_file = sample_info_file
        self.sample_info = pd.read_excel(self.sample_info_file)
        self.sample_info.set_index("Study ID", inplace=True)
    
    def data_arrangement(self):
        # Initialize lists for grouping data
        groups = []
        time_points = []
        o_types = []

        # Iterate over the index to extract group, time point, and O-type information
        for index in self.trimmed_avg_mfi_data.index:
            if "HC" in index:
                groups.append("HC")
                time_points.append(np.nan)
                o_types.append(np.nan)
            elif "BC" in index:
                groups.append("BC")
                time_points.append(np.nan)
                o_types.append(np.nan)
            else:
                group, time_point = index.split("_")[:2]
                groups.append(group)
                time_points.append(int(time_point[1:]))
                trim_index = "_".join(index.split('_')[:2])
                o_type = self.sample_info.at[trim_index, "O-type"] if trim_index in self.sample_info.index else np.nan
                o_types.append(o_type)

        # Add new columns to the DataFrame
        trimmed_avg_mfi_data2 = self.trimmed_avg_mfi_data.copy(deep=True)
        trimmed_avg_mfi_data2["Group"] = groups
        trimmed_avg_mfi_data2["Time point"] = time_points
        trimmed_avg_mfi_data2["O-type"] = o_types

        return trimmed_avg_mfi_data2





