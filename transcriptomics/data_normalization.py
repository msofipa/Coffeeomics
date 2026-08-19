import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def counts_distribution_per_gene (count_data):
    gene_totals = count_data.sum(axis=1) # Sum of the read counts across all samples for each gene

    plt.figure(figsize=(19, 6))
    sns.histplot(gene_totals, bins=100, kde=False, log_scale=True)

    plt.xlabel("log (Total raw counts per gene)")
    plt.ylabel("Number of genes with that expression level")
    plt.title("Raw Gene Abundance Distribution")
    plt.tight_layout()
    plt.show()

def gene_distribution (count_data, sample, line_1=None, line_2=None, line_3=None, label_1=None, label_2=None, label_3=None):
    column = count_data[sample]
    plt.figure(figsize=(19, 6))
    sns.histplot(column, bins=100, kde=False, log_scale=True)

    if line_1 is not None and label_1 is not None:
        plt.axvline(x=line_1, color='red', linestyle='--', linewidth=2, label=label_1)
    if line_2 is not None and label_2 is not None:
        plt.axvline(x=line_2, color='orange', linestyle='--', linewidth=2, label=label_2)
    if line_3 is not None and label_3 is not None:
        plt.axvline(x=line_3, color='purple', linestyle='--', linewidth=2, label=label_3)

    plt.xlabel(f"log (counts)")
    plt.ylabel("Frequency")
    plt.title(f"Raw Abundance Distribution for Sample {sample}")
    plt.legend()
    plt.show()

def  manual_normalization (data, normalization_factors):
    normalized_data = data.divide(normalization_factors, axis=1)  # Gene count in each sample/normalization factor for each sample
    return normalized_data

import pandas as pd
from scipy.stats import lognorm

def our_normalization ():
    
    norm_factor = 'Most common expression level for a gene g in a sample i ir relation to the average expression for a gene in all samples'
    mean_peaks = mean_peaks = np.mean(list(peaks_histogram_rollingavg5.values())) # Get the mean of all the peaks
    ratio = ratio = {k: v / mean_peaks for k, v in peaks_histogram_rollingavg5.items()} # How much the most common expression level for a gene g in a sample i is above the average most common expression
    normalization=manual_normalization(count_data, ratio)
    to_bar_graph_counts_per_sample (normalization, f"Counts per Sample (Normalized) - {norm_factor} - Peaks by Rollin Avg Peak" )
    #plt.savefig(f"Counts_per_Sample_(Normalized)_{norm_factor}_Peaks_by_RollingAvg.png")


from pydeseq2.dds import DeseqDataSet

def pydeseq2_normalization(count_data, file_name):

    # Load metadata (samples and conditions)
    meta_data = pd.DataFrame({
        "sample": count_data.columns,
        "condition": [s[:-1] for s in count_data.columns] 
    })
    meta_data.set_index("sample", inplace=True)

    # Create dds object
    dds = DeseqDataSet(
        counts=count_data.T,  # PyDESeq2 requires samples as rows, so we transposed it
        metadata=meta_data,
        design_factors="condition"
    )

    dds.deseq2()

    print(dds)

    # Get normalized counts from AnnData layer
    normalized_counts = pd.DataFrame(
        dds.layers["normed_counts"],  # numpy array
        index=dds.obs_names,          # sample names
        columns=dds.var_names         # gene names
    )
    normalized_counts.to_csv(f"f{file_name}.csv", index=True)

    return normalized_counts.T

'''
    sample_conditions = [s[:-1] for s in count_data.columns]
    # Total normalized reads per sample
    normalized_totals = normalized_counts.sum(axis=1)

    # Prepare for plotting
    df_norm = pd.DataFrame({
        "Sample": normalized_totals.index,
        "Total Normalized Reads": normalized_totals.values,
        "Condition": sample_conditions
    })

    # Plot
    plt.figure(figsize=(19, 6))
    sns.barplot(x="Sample", y="Total Normalized Reads", hue="Condition", data=df_norm, palette="colorblind")

    # Formatting
    plt.xticks(rotation=90)
    plt.xlabel("Samples")
    plt.ylabel("Total Normalized Read Counts")
    plt.title(f"Counts per Sample - PyDESeq2 Normalized")

    # Legend at bottom
    plt.legend(title="Condition", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=10)
    plt.savefig(f"{file_name}", dpi=300, bbox_inches="tight")
    plt.show()
'''



'''
norm_factor = 'Manual interpertation of dseq'
from scipy.stats import gmean
#geo_means = gmean(count_data + 0.00001, axis=1) #for each gene g #Can not divide by 0 "axis=0 meaning by the row of a column"
geo_means = gmean(count_data[count_data > 0], axis=1)  # dseq drop all-zero genes from the mean
ratio = count_data.divide(geo_means, axis=0)
size_factor=ratio.median(axis=0)
normalization=manual_normalization(count_data, size_factor)
to_bar_graph_counts_per_sample (normalization, f"Counts per Sample (Normalized) - {norm_factor}")
guardar(normalization, 'norm_trans_manual_dseq', 'csv')
#plt.savefig(f"Counts per Sample (Normalized) {norm_factor}.png")
'''
from scipy.stats import gmean

def manual_deseq2_normalization (count_data):
    geo_means = gmean(count_data[count_data > 0], axis=1)  # dseq drop all-zero genes from the mean
    ratios = count_data.divide(geo_means, axis=0)
    size_factors=ratios.median(axis=0)

def gene_distribution (count_data, sample, line_1=None, line_2=None, line_3=None, label_1=None, label_2=None, label_3=None):
    column = count_data[sample]
    plt.figure(figsize=(19, 6))
    sns.histplot(column, bins=100, kde=False, log_scale=True)

    if line_1 is not None and label_1 is not None:
        plt.axvline(x=line_1, color='red', linestyle='--', linewidth=2, label=label_1)
    if line_2 is not None and label_2 is not None:
        plt.axvline(x=line_2, color='orange', linestyle='--', linewidth=2, label=label_2)
    if line_3 is not None and label_3 is not None:
        plt.axvline(x=line_3, color='purple', linestyle='--', linewidth=2, label=label_3)

    plt.xlabel(f"log (total raw counts per gene)")
    plt.ylabel("Frequency (Number of genes)")
    plt.title(f"Raw Gene Abundance Distribution for Sample {sample}")
    plt.legend()
    plt.show()




def peaks_samplewise (count_data, sample, nbins, window=None):

    array = [i for i in count_data[sample] if i >= 10] #Pre-filtering: Only keep counts equal or above 10 
    log_dist = np.log(array) #Make the counts logarithmic
    counts, bins = np.histogram(log_dist, bins=nbins)    #Generate histogram of pre filtered logarithmic counts

    rolling_counts = np.convolve(counts, np.ones(window)/window, mode='same') #Reparar
    plt.figure(figsize=(19, 6))
    plt.bar(bins[:-1], rolling_counts, width=np.diff(bins), edgecolor='black', alpha=0.7, label="Rolling Counts Histogram")

    # Find peak of the histogram
    bin_centers = (bins[:-1] + bins[1:]) / 2
    peak_bin_index = np.argmax(rolling_counts)
    peak = (bins[peak_bin_index] + bins[peak_bin_index + 1]) / 2
    peaks_histogram_rollingavg5 = np.exp(peak)
    plt.axvline(peak, color='orange', linestyle='--', label=f"Histogram Peak Rolling Average = {np.exp(peak):.1f}")

    plt.xlabel("log(counts)")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of log(counts)Rolling Avg 5- {sample}")
    plt.legend()
    plt.show()

    return {sample, peaks_histogram_rollingavg5}

'''
from data_normalization import manual_normalization
norm_factor = 'Peaks Determined by Rolling Average of 5'
# Most common expression level for a gene g in a sample i ir relation to the average expression for a gene in all samples
mean_peaks = mean_peaks = np.mean(list(peaks.values())) # Get the mean of all the peaks
ratio = ratio = {k: v / mean_peaks for k, v in peaks.items()} # How much the most common expression level for a gene g in a sample i is above the average most common expression
normalization= manual_normalization(trans_raw_data, ratio)
to_bar_graph_counts_per_sample (normalization, f"Counts per Sample (Normalized) - {norm_factor} - Peaks by Rollin Avg Peak", True)
'''

def our_normalization (data, nbins, window):
    peaks = our_normalization_factors (data, nbins=nbins, window=window, show=False, single_figure=False)
    norm_factor = f'Peaks Determined by Rolling Average of {window} for '
    # Most common expression level for a gene g in a sample i ir relation to the average expression for a gene in all samples
    mean_peaks = mean_peaks = np.mean(list(peaks.values())) # Get the mean of all the peaks
    ratio = ratio = {k: v / mean_peaks for k, v in peaks.items()} # How much the most common expression level for a gene g in a sample i is above the average most common expression
    return manual_normalization(data, ratio)

def our_normalization_factors(count_data, nbins, window, show=False, single_figure=False):

    count_data = count_data[
        sorted(count_data.columns, key=lambda x: (int(x[:-1]), x[-1]))
    ]

    peaks_histogram_rollingavg5 = {}
    n_samples = len(count_data.columns)

    if single_figure:

        ncols = 4
        nrows = math.ceil(n_samples / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
        axes = axes.flatten()

        for idx, sample in enumerate(count_data.columns):
            ax = axes[idx]

            array = [i for i in count_data[sample] if i >= 10]
            log_dist = np.log(array)

            counts, bins = np.histogram(log_dist, bins=nbins)

            rolling_counts = np.convolve(
                counts, np.ones(window)/window, mode='same'
            )

            peak_bin_index = np.argmax(rolling_counts)
            peak = (bins[peak_bin_index] + bins[peak_bin_index + 1]) / 2
            peaks_histogram_rollingavg5[sample] = np.exp(peak)

            ax.bar(
                bins[:-1],
                rolling_counts,
                width=np.diff(bins),
                edgecolor='black',
                alpha=0.7
            )

            ax.axvline(peak, color='orange', linestyle='--')
            ax.set_title(sample)
            ax.set_xlabel("log(counts)")
            ax.set_ylabel("Frequency")

        for i in range(idx + 1, len(axes)):
            fig.delaxes(axes[i])

        fig.suptitle(
            f"Histogram of log(counts) — nbins={nbins}, rolling window={window}",
            fontsize=18
        )
        if show:
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
            plt.show()

    else:

        for sample in count_data.columns:

            array = [i for i in count_data[sample] if i >= 10]
            log_dist = np.log(array)

            counts, bins = np.histogram(log_dist, bins=nbins)

            rolling_counts = np.convolve(
                counts, np.ones(window)/window, mode='same'
            )

            peak_bin_index = np.argmax(rolling_counts)
            peak = (bins[peak_bin_index] + bins[peak_bin_index + 1]) / 2
            peaks_histogram_rollingavg5[sample] = np.exp(peak)

            if show:
                plt.figure(figsize=(19, 6))
                plt.bar(
                    bins[:-1],
                    rolling_counts,
                    width=np.diff(bins),
                    edgecolor='black',
                    alpha=0.7
                )

                plt.axvline(peak, color='orange', linestyle='--')
                plt.xlabel("log(counts)")
                plt.ylabel("Frequency")
                plt.title(
                    f"{sample} — nbins={nbins}, rolling window={window}"
                )
                plt.show()

    return peaks_histogram_rollingavg5

import math
import numpy as np
import matplotlib.pyplot as plt

def our_normalization_factors(count_data, nbins, window, show=False, single_figure=False):

    count_data = count_data[
        sorted(count_data.columns, key=lambda x: (int(x[:-1]), x[-1]))
    ]

    peaks_histogram_rollingavg5 = {}
    n_samples = len(count_data.columns)

    if single_figure:

        ncols = 4
        nrows = math.ceil(n_samples / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
        axes = axes.flatten()

        for idx, sample in enumerate(count_data.columns):
            ax = axes[idx]

            array = [i for i in count_data[sample] if i >= 10]
            log_dist = np.log(array)

            counts, bins = np.histogram(log_dist, bins=nbins)

            rolling_counts = np.convolve(
                counts, np.ones(window)/window, mode='same'
            )

            peak_bin_index = np.argmax(rolling_counts)
            peak = (bins[peak_bin_index] + bins[peak_bin_index + 1]) / 2
            peaks_histogram_rollingavg5[sample] = np.exp(peak)

            ax.bar(
                bins[:-1],
                rolling_counts,
                width=np.diff(bins),
                edgecolor='black',
                alpha=0.7
            )

            ax.axvline(peak, color='orange', linestyle='--')
            ax.set_title(sample)
            ax.set_xlabel("log(counts)")
            ax.set_ylabel("Frequency")

        for i in range(idx + 1, len(axes)):
            fig.delaxes(axes[i])

        fig.suptitle(
            f"Histogram of log(counts) — nbins={nbins}, rolling window={window}",
            fontsize=18
        )
        if show:
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
            plt.show()

    else:

        for sample in count_data.columns:

            array = [i for i in count_data[sample] if i >= 10]
            log_dist = np.log(array)

            counts, bins = np.histogram(log_dist, bins=nbins)

            rolling_counts = np.convolve(
                counts, np.ones(window)/window, mode='same'
            )

            peak_bin_index = np.argmax(rolling_counts)
            peak = (bins[peak_bin_index] + bins[peak_bin_index + 1]) / 2
            peaks_histogram_rollingavg5[sample] = np.exp(peak)

            if show:
                plt.figure(figsize=(19, 6))
                plt.bar(
                    bins[:-1],
                    rolling_counts,
                    width=np.diff(bins),
                    edgecolor='black',
                    alpha=0.7
                )

                plt.axvline(peak, color='orange', linestyle='--')
                plt.xlabel("log(counts)")
                plt.ylabel("Frequency")
                plt.title(
                    f"{sample} — nbins={nbins}, rolling window={window}"
                )
                plt.show()

    return peaks_histogram_rollingavg5

import numpy as np
import matplotlib.pyplot as plt

def sort_key(k):
    num = int(''.join(filter(str.isdigit, k)))
    letter = ''.join(filter(str.isalpha, k))
    return (num, letter)

def distribution_cohesion (peaks):
    
    sorted_items = sorted(peaks.items(), key=lambda x: sort_key(x[0]))
    labels = [k for k, v in sorted_items]
    values = np.array([v for k, v in sorted_items])
    x = np.arange(len(values))
    avg = values.mean()
    mad = np.mean(np.abs(values - avg))

    plt.figure(figsize=(17, 5))
    plt.scatter(x, values)

    # Average line
    plt.axhline(avg, linestyle='--', linewidth=1)
    plt.text(len(values)-1, avg, f' Avg = {avg:.1f}',
            va='bottom', ha='right')

    # Add labels directly on points
    for i, label in enumerate(labels):
        plt.text(x[i], values[i], label,
                fontsize=8,
                ha='center',
                va='bottom')

    plt.xticks(x, labels, rotation=90)
    plt.ylabel("Value")
    plt.xlabel("Sample")
    plt.tight_layout()
    plt.show()
    return mad


def our_normalization (data, nbins, window):
    peaks = our_normalization_factors (data, nbins=nbins, window=window, show=False, single_figure=False)
    norm_factor = f'Peaks Determined by Rolling Average of {window} for '
    # Most common expression level for a gene g in a sample i ir relation to the average expression for a gene in all samples
    mean_peaks = mean_peaks = np.mean(list(peaks.values())) # Get the mean of all the peaks
    ratio = ratio = {k: v / mean_peaks for k, v in peaks.items()} # How much the most common expression level for a gene g in a sample i is above the average most common expression
    return manual_normalization(data, ratio)




