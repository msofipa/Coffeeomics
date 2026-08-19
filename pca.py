import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


def run_pca_analysis(df, arabica_ids, canephora_ids, name="Dataset", top=30, log_transform=True):

    '''
    Generate individual PCA Plots for each species
    '''

    species_groups = {
        "Arabica": df.loc[:, df.columns.intersection(arabica_ids)],
        "Canephora": df.loc[:, df.columns.intersection(canephora_ids)]
    }

    for species, species_df in species_groups.items():

        if species_df.shape[1] < 2:
            print(f"Skipping {species} (not enough samples)")
            continue

        # transpose so samples are rows
        X = species_df.T
        X.index = X.index.get_level_values(-1)  # add this line

        # log transform
        if log_transform:
            X = X.apply(lambda x: np.log1p(x))

        # scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame(
            pcs,
            columns=["PC1", "PC2"],
            index=X.index
        )

        # -------------------------
        # PCA scatter plot
        plt.figure(figsize=(6,5))
        plt.scatter(pca_df["PC1"], pca_df["PC2"])

        for sample in pca_df.index:
            plt.text(
                pca_df.loc[sample, "PC1"],
                pca_df.loc[sample, "PC2"],
                sample,
                fontsize=9
            )

        plt.title(f"PCA - {name} ({species})")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
        plt.tight_layout()
        plt.show()

        # -------------------------
        # Loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            index=X.columns,
            columns=["PC1", "PC2"]
        )

        squared_loadings = loadings ** 2
        gene_contrib_pct = 100 * squared_loadings.div(squared_loadings.sum())

        # top genes
        top_pc1 = gene_contrib_pct["PC1"].sort_values(ascending=False).head(top)
        top_pc2 = gene_contrib_pct["PC2"].sort_values(ascending=False).head(top)

        # -------------------------
        # Plot contributions
        plt.figure(figsize=(6,6))
        top_pc1.sort_values().plot(kind="barh")
        plt.title(f"Top {top} PC1 Contributors (%) - {species}")
        plt.xlabel("Contribution (%)")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6,6))
        top_pc2.sort_values().plot(kind="barh")
        plt.title(f"Top {top} PC2 Contributors (%) - {species}")
        plt.xlabel("Contribution (%)")
        plt.tight_layout()
        plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


def run_pca_analysis_colured(df, species_series, name="Dataset", top=30, log_transform=True):

    'Generate a single ploth for both species'

    # transpose so samples are rows
    X = df.T

    # log transform
    if log_transform:
        X = X.apply(lambda x: np.log1p(x))

    # scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        pcs,
        columns=["PC1", "PC2"],
        index=X.index
    )

    # add species annotation
    pca_df["Species"] = species_series.loc[pca_df.index]

    # -------------------------
    # PCA PLOT
    plt.figure(figsize=(10,6))

    colors = {
        "C. arabica": "darkgreen",
        "C. canephora": "darkorange"
    }

    for species, group in pca_df.groupby("Species"):
        plt.scatter(
            group["PC1"],
            group["PC2"],
            label=species,
            color=colors.get(species, "grey"),
            s=70
        )

    # sample labels
    for sample in pca_df.index:
        plt.text(
            pca_df.loc[sample, "PC1"],
            pca_df.loc[sample, "PC2"],
            sample,
            fontsize=8
        )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
    plt.title(f"PCA - {name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=["PC1", "PC2"]
    )

    squared_loadings = loadings ** 2
    gene_contrib_pct = 100 * squared_loadings.div(squared_loadings.sum())

    # top contributors
    top_pc1 = gene_contrib_pct["PC1"].sort_values(ascending=False).head(top)
    top_pc2 = gene_contrib_pct["PC2"].sort_values(ascending=False).head(top)

    return pca_df, loadings