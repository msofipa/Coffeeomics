import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

sample_info = pd.read_csv(r"C:\Users\papan\Desktop\Coffeomics\sample_information.csv", header=0, index_col="Sample_ID")

def analyzes_df_entries (df):
    # Count NaNs
    nan_count = df.isna().sum().sum()

    # Count zeros
    zero_count = (df == 0).sum().sum()

    # Count negative values
    negative_count = (df < 0).sum().sum()

    # Count positive values
    positive_count = (df > 0).sum().sum()

    print("NaNs:", nan_count)
    print("Zeros:", zero_count)
    print("Negative Entires:", negative_count)
    print("Positive Entries:", positive_count)
    print("Shape:", df.shape)
    

def to_table_counts_per_sample(data, column, attribute=False):

    total_counts = data.sum(axis=0)

    df_table = pd.DataFrame({
        "Sample": total_counts.index,
        f"{column}": total_counts.values
    })

    #df_table["Condition"] = df_table["Sample"].str[:-1]

    if attribute:
        df_table["Sample_num"] = df_table["Sample"].str.extract(r"(\d+)").astype(int)
        df_table["Sample_alpha"] = df_table["Sample"].str.extract(r"([A-Za-z]+)")
        df_table = df_table.sort_values(by=["Sample_num", "Sample_alpha"])
        df_table = df_table.drop(columns=["Sample_num", "Sample_alpha"])

    # Summary row
    summary_row = pd.DataFrame({
        "Sample": ["TOTAL"],
        #"Condition": [""],
        f"{column}": [df_table[f"{column}"].sum()]
        
    })
    df_table = pd.concat([df_table, summary_row], ignore_index=True)

    return df_table

def to_bar_graph_counts_per_sample(data, titulo, sorted=False, average=False):

    total_counts = data.sum(axis=0)

    df_plot = pd.DataFrame({
        "Sample": total_counts.index,
        "Total Reads": total_counts.values
    })

    df_plot["Condition"] = df_plot["Sample"].str[:-1]

    if sorted:
        # Extract numeric and alphabetical parts (e.g., 12A -> 12, A)
        df_plot["Sample_num"] = df_plot["Sample"].str.extract(r"(\d+)").astype(int)
        df_plot["Sample_alpha"] = df_plot["Sample"].str.extract(r"([A-Za-z]+)")

        # Sort by number first, then alphabetically
        df_plot = df_plot.sort_values(by=["Sample_num", "Sample_alpha"])

    plt.figure(figsize=(19, 6))
    ax = sns.barplot(
        x="Sample",
        y="Total Reads",
        hue="Condition",
        data=df_plot,
        palette="colorblind"
    )

    # Add average line if requested
    if average:
        avg_value = df_plot["Total Reads"].mean()
        plt.axhline(
            y=avg_value,
            color='grey',
            linestyle='--',
            linewidth=2,
            label=f'Average ({avg_value:,.0f})'
        )
        plt.legend(title="Condition", loc="upper center",
                   bbox_to_anchor=(0.5, -0.15), ncol=10)

    plt.xticks(rotation=90)
    plt.xlabel("Samples")
    plt.ylabel("Total Counts")
    plt.title(titulo)

    plt.show()

def sample_info_characterization(data):
    print(f"Total number of samples: {len(data)}")
    unique_combinations = data[['Species', 'Temperature', 'CO2_ppm', 'Water_Stress']].drop_duplicates().shape[0]
    print(f'Number of unique combinations: {unique_combinations}')
    print(f"{data['Species'].value_counts()}", "\n")
    print(f"{data['Temperature'].value_counts()}", "\n")
    print(f"{data['CO2_ppm'].value_counts()}", "\n")
    print(f"{data['Water_Stress'].value_counts()}", "\n")

def load_clean_data(sheet):
    count_data = pd.read_excel(r'C:\Users\papan\Desktop\Coffeomics\data_to_normalize.xlsx', sheet_name=sheet, index_col=0)
    print(f'Number of individual samples for {sheet}: {len(count_data.columns)}/{len(sample_info)}')
    return count_data


def count_attribute (df, attribute=None, value=None):
    '''
    Count of hou many times 
    '''
    if attribute is not None and value is not None:
        filtered = sample_info.index[sample_info[attribute] == value]
    else:
        filtered = sample_info.index
    common_samples = df.columns.intersection(filtered)
    print(f"{len(common_samples)}/{len(filtered)} existing samples") 
    return common_samples

