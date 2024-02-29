#!/usr/bin/env python3
import argparse
import os

import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
import pandas as pd
from skbio.diversity.alpha import chao1, shannon, simpson
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FixedLocator

# from tqdm.auto import tqdm

from get_abundance import get_otu_count, _agg_along_axis, _taxa_qc


def _get_dendrogram(df: pd.DataFrame, rows_to_ignore=None):
    """Perform hierarchical clustering on the dataframe (treating rows as features and
    columns as samples), reorder the rows and get the dendrogram, ignoring specified rows.

    Args:
    df (pd.DataFrame): The input dataframe.
    rows_to_ignore (list, optional): Rows to ignore during clustering. Defaults to None.

    Returns:
    tuple: A tuple containing the reordered dataframe and the linkage matrix.
    """
    if rows_to_ignore is None:
        rows_to_ignore = []
    not_in_df = set(rows_to_ignore) - set(df.index)
    if not_in_df:
        print(
            f"WANRING: `rows_to_ignore` contains rows not in the dataframe: {not_in_df}"
        )
        rows_to_ignore = [i for i in rows_to_ignore if i not in not_in_df]

    # Separate the dataframe into rows to cluster and rows to ignore
    rows_to_cluster = df.index.difference(rows_to_ignore)
    df_to_cluster = df.loc[rows_to_cluster]

    # Perform hierarchical clustering on the rows to be clustered
    linked = linkage(df_to_cluster, "single", optimal_ordering=False)
    order = leaves_list(linked)

    # Reorder the dataframe based on the clustering result, appending ignored rows at the end
    reordered_df = pd.concat(
        [df_to_cluster.iloc[order], df.loc[rows_to_ignore]], axis=0
    )

    return reordered_df, linked


def _stacked_bar(
    dfs: list[pd.DataFrame],
    names: list[str],
    title: str,
    palette: list,
    axs: list[plt.Axes],
) -> None:
    for i, (ax_dend, ax, df, name) in enumerate(zip(axs[0], axs[1], dfs, names)):
        # df = df.copy()
        # index = []
        # for j in df.index:
        #     if j.endswith("-Swab-combined"):
        #         index.append(j[1] + "-Bulk")
        #     elif j.endswith("-Scrape-R2A"):
        #         index.append(j[1] + "-Plate-R2A")
        #     elif j.endswith("-Scrape-TSA"):
        #         index.append(j[1] + "-Plate-TSA")
        #     else:
        #         raise ValueError(f"Unknown column: {j}")
        # df.index = index

        df.plot(kind="bar", stacked=True, color=palette, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Relative abundance")
        ax.set_title(name)

        if ax_dend is not None:
            ax_dend.axis("off")

        # set xticklabels size
        # ax.tick_params(axis="x", labelsize=6)
        # remove legend unless the last plot
        if i != len(axs[0]) - 1:
            ax.get_legend().remove()
        else:
            ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
            ax.get_legend().set_title(title)
        ax.set_ylabel(ax.yaxis.get_label().get_text(), fontsize=14)
        ax.set_xlabel(ax.xaxis.get_label().get_text(), fontsize=14)


def _black_color_func(*args, **kwargs):
    return "black"


def _heatmap(
    dfs: list[pd.DataFrame],
    names: list[str],
    title: str,
    fig: plt.Figure,
    axs_heatmap: list[plt.Axes],
    axs_dend: list[plt.Axes],
    cmap: str,
    linkage_row: np.ndarray = None,
    cbar_label: str = None,
    vmax: float | None = None,
    vmin: float | None = None,
) -> None:
    for i, (ax, ax_dend, df, name) in enumerate(
        zip(axs_heatmap, axs_dend, dfs, names, strict=True)
    ):
        # df = df.copy()
        # columns = []
        # for j in df.columns:
        #     if j.endswith("-Swab-combined"):
        #         columns.append(j[1] + "-Bulk")
        #     elif j.endswith("-Scrape-R2A"):
        #         columns.append(j[1] + "-Plate-R2A")
        #     elif j.endswith("-Scrape-TSA"):
        #         columns.append(j[1] + "-Plate-TSA")
        #     else:
        #         raise ValueError(f"Unknown column: {j}")
        # df.columns = columns

        if cbar_label is not None and i == len(axs_heatmap) - 1:
            cbar = True
            height = axs_heatmap[i - 1].get_position().height * 0.4
            cbar_ax = fig.add_axes(
                [
                    ax.get_position().x1 + (0.03 if linkage_row is None else 0.12),
                    axs_heatmap[i - 1].get_position().y1 - height,
                    0.03,
                    height,
                ]
            )
            cbar_ax.yaxis.label.set_size(12)
            cbar_kws = {"label": cbar_label}
        else:
            cbar = False
            cbar_ax = None
            cbar_kws = None

        if ax_dend is not None:
            df, linked = _get_dendrogram(df.transpose())
            df = df.transpose()
        sns.heatmap(
            df,
            ax=ax,
            cmap=cmap,
            lw=0.7,
            cbar=cbar,
            cbar_ax=cbar_ax,
            cbar_kws=cbar_kws,
            vmax=vmax,
            vmin=vmin,
            square=True,
        )
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=90)

        if ax_dend is not None:
            # plot dendrogram
            dendrogram(
                linked, ax=ax_dend, leaf_rotation=90, link_color_func=_black_color_func
            )
            ax_dend.axis("off")
            ax_dend.set_title(name)
        else:
            ax.set_title(name)

        if linkage_row is not None and i == len(axs_heatmap) - 1:
            dend_row_ax = fig.add_axes(
                [
                    ax.get_position().x1 + 0.01,
                    axs_heatmap[i - 1].get_position().y0,
                    0.1,
                    axs_heatmap[i - 1].get_position().height,
                ]
            )
            dend_row_ax.axis("off")
            dendrogram(
                linkage_row,
                ax=dend_row_ax,
                orientation="right",
                link_color_func=_black_color_func,
            )
            num_elements_in_linkage = linkage_row.shape[0] + 1
            num_elements_not_in_linkage = df.shape[0] - num_elements_in_linkage
            ylow = -10 * num_elements_not_in_linkage
            yhigh = df.shape[0] * 10 + ylow
            dend_row_ax.set_ylim(ylow, yhigh)

        # turn off y axis ticks and tick labels (set invisible) except for the first panel
        if i == 0:
            ax.set_ylabel(title, fontsize=16)
        else:
            ax.yaxis.set_visible(False)
        # ax.tick_params(axis="x", labelsize=7)


def _barplot_with_whisker_strip(
    dfs: pd.DataFrame,
    names: list[str],
    title: str,
    group_key: str,
    column_of_interest: str,
    plot_strip: bool,
    axs: list[plt.Axes],
    ylog: bool = False,
) -> None:
    axis_tick_fs = 12
    values = []
    for i, (ax, df, name) in enumerate(zip(axs, dfs, names)):
        df = df.copy()
        # index = []
        # for j in df[group_key]:
        #     if j.endswith("-Swab-combined"):
        #         index.append(j[1] + "-Bulk")
        #     elif j.endswith("-Scrape-R2A"):
        #         index.append(j[1] + "-Plate-R2A")
        #     elif j.endswith("-Scrape-TSA"):
        #         index.append(j[1] + "-Plate-TSA")
        #     else:
        #         raise ValueError(f"Unknown column: {j}")
        # df[group_key] = index

        group = df.groupby(group_key, sort=False)
        # Determine which groups have more than one sample
        groups_w_multi_samples = group.filter(lambda x: len(x) > 1)[group_key].unique()

        # assign different colors depending on if group.groups.keys() have "old", "reamplify", or "Scrape"
        # colors = []
        # for key in group.groups.keys():
        #     if key.endswith("-Bulk"):
        #         colors.append("tab:blue")
        #     elif key.endswith("-Plate-R2A"):
        #         colors.append("tab:orange")
        #     elif key.endswith("-Plate-TSA"):
        #         colors.append("tab:green")
        #     else:
        #         raise ValueError(f"Unknown group: {key}")
        colors = "gray"

        # Plot bars with error bars for groups with multiple samples
        value = group[column_of_interest].mean()
        values.extend(value)
        if len(groups_w_multi_samples):
            ax.bar(
                range(len(group)),
                value,
                yerr=group[column_of_interest].std()
                * [
                    1 if group in groups_w_multi_samples else 0
                    for group in group.groups.keys()
                ],
                capsize=2,
                error_kw={"elinewidth": 1.5, "capthick": 1.5},
                width=0.6,
                # fill=False,
                color=colors,
                log=ylog,
            )

            # Remove edgecolor from bars to get rid of the faint gray line
            # for bar in bars:
            #     bar.set_edgecolor("none")

            # Use stripplot for individual points for groups with multiple samples
            if plot_strip:
                df_filtered = df[df[group_key].isin(groups_w_multi_samples)]
                sns.stripplot(
                    x=group_key,
                    y=column_of_interest,
                    data=df_filtered,
                    color="gray",
                    order=group.groups.keys(),
                    ax=ax,
                    size=2,
                )
        else:
            ax.bar(
                range(len(group)),
                value,
                width=0.6,
                # fill=False,
                color=colors,
                log=ylog,
            )
        ax.xaxis.set_major_locator(FixedLocator(range(len(group.groups.keys()))))
        ax.set_xticklabels(
            labels=group.groups.keys(), rotation=90, fontsize=axis_tick_fs
        )
        ax.tick_params(axis="y", labelsize=axis_tick_fs)
        # leave some space on the left and right
        ax.set_xlim(-0.5, len(group.groups.keys()) - 0.5)
        ax.set_title(name)

    axs[0].set_ylabel(title, fontsize=14)
    if ylog:
        # set range to lower than the minimum value and higher than the maximum value
        ax.set_ylim(
            10 ** (np.floor(np.log10(min(values)))),
            10 ** (np.ceil(np.log10(max(values)))),
        )


def _calc_alpha_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate alpha diversity metrics for each sample.
    Note that you should ignore chao1 if your input data is not integer count.
    """
    return pd.DataFrame(
        {
            "chao1": df.apply(chao1, axis=1),
            "richness": df.apply(lambda x: (x > 0).sum(), axis=1),
            "shannon": df.apply(shannon, axis=1),
            "simpson": df.apply(simpson, axis=1),
        },
        index=df.index,
    )


def _get_subplots(
    num_cols: int,
    fig_size: tuple[float, float],
    width_ratios: list[float],
    height_ratios: tuple[float, float] | None = (0.07, 1),
    wspace: float = 0.1,
    hspace: float = 0.01,
) -> tuple[plt.Figure, list[list[plt.Axes]]]:
    """A helper function to get matplotlib figure and axes for plotting stacked bar plot
    or heatmap with multiple panels.
    """
    if height_ratios is None:  # no hierarchical clustering
        fig, axs = plt.subplots(
            1,
            num_cols,
            sharey="row",
            figsize=fig_size,
            width_ratios=width_ratios,
        )
        if num_cols == 1:
            axs = [axs]
        axs = [[None] * num_cols, axs]
        fig.subplots_adjust(wspace=wspace)
    else:
        if not len(height_ratios) == 2:
            raise ValueError("height_ratios should be a list of length 2.")
        fig, axs = plt.subplots(
            2,
            num_cols,
            figsize=fig_size,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
        )
        if num_cols == 1:
            axs = [[axs[0]], [axs[1]]]
        # share y for the heatmap, i.e., the second row
        for i in range(1, num_cols):
            axs[1, i].sharey(axs[1, 0])
        fig.subplots_adjust(wspace=wspace, hspace=hspace)
    return fig, axs


def _rarefying(
    df: pd.DataFrame, ref: list[int], repeat_num: int = 20
) -> list[pd.DataFrame | list]:
    """Rarefy the dataframe to the reference list or integer.

    The input dataframe should be sample x features, with sample name as index.

    In the returned output, each sample in the input dataframe will be rarefied
    `repeat_num` times, resulting in a dataframe with shape (sample x features x
    repeat_num), except for samples whose reference is themselves, which will be
    repeated only once. Sample name in the returned output will be like
    <original_sample_name>_rarefied_<repeat_num>.
    """
    # make sure all columns in df are positive integers and are in ref.
    if not len(ref) == len(df):
        raise ValueError(
            "The length of ref should be the same as the number of rows in df."
        )

    # get a list of 2d numpy array using joblib parallisim
    res = Parallel(n_jobs=4)(
        delayed(rarefy_array)(df.iloc[idx].to_numpy(), n, repeat_num)
        for idx, n in enumerate(ref)
    )
    sample_names_new_orig = [
        (f"{sample_name}_rarefied_{j}", sample_name)
        for idx, sample_name in enumerate(df.index)
        for j in range(res[idx].shape[0])
    ]
    res = np.concatenate(res, axis=0)
    idx_new, idx_orig = zip(*sample_names_new_orig)
    return pd.DataFrame(res, index=idx_new, columns=df.columns), idx_orig


def rarefy_array(arr: np.ndarray, n: int, k: int, seed: int = 42) -> np.ndarray:
    """Rarefy one row k times so that each row sum up to n."""
    depth = arr.sum()
    rng = np.random.default_rng(seed=seed)
    if n >= depth:  # don't rarefy if expected depth is larger than the actual depth.
        return arr.reshape(1, -1)
    all_elements = np.repeat(np.arange(arr.size), arr)
    return np.stack(
        [
            np.bincount(
                rng.choice(all_elements, size=n, replace=False), minlength=arr.size
            )
            for _ in range(k)
        ],
        axis=0,
    )


if __name__ == "__main__":

    matplotlib.use("TkAgg")
    plt.rcParams["pdf.fonttype"] = 42

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    parser_ab = subparsers.add_parser("abundance_group")
    parser_ab.add_argument(
        "plot_type",
        type=str,
        choices=["stacked_bar", "heatmap", "heatmap_log10", "heatmap_binary", "all"],
    )
    parser_ab.add_argument("-i", "--otu_count_tsv", type=str, required=True)
    parser_ab.add_argument("-m", "--metadata", type=str, required=True)
    parser_ab.add_argument("-t", "--otu_taxonomy_tsv", type=str, required=True)
    parser_ab.add_argument("-f", "--fig_dir", type=str, required=True)
    parser_ab.add_argument("-r", "--rep_group_key", type=str, default="rep_group")
    parser_ab.add_argument("-s", "--sample_group_key", type=str, default="sample_group")
    parser_ab.add_argument("-sp", "--spikein_taxa_key", type=str, default="spike_in")
    parser_ab.add_argument(
        "-l", "--tax_levels", nargs="+", default=["order", "family", "genus", "otu"]
    )
    parser_ab.add_argument(
        "-a",
        "--rel_ab_thresholds",
        nargs="+",
        type=float,
        default=[0.0],
        help="Genus relative abundance threshold.",
    )
    parser_ab.add_argument(
        "-sc",
        "--sample_hierarchical_clustering",
        action="store_true",
        default=False,
        help="Perform hierarchical clustering on samples in each group, order them "
        "accordingly and add a dendrogram at the top. Only affects heatmap, not "
        "stacked bar plot.",
    )
    parser_ab.add_argument(
        "-fc",
        "--feature_hierarchy_clustering",
        action="store_true",
        default=False,
        help="Perform hierarchical clustering on features in each group, order them "
        "accordingly and add a dendrogram at the left. Only affects heatmap, not "
        "stacked bar plot.",
    )

    parser_stats = subparsers.add_parser("stats_sample_count")
    parser_stats.add_argument("-i", "--otu_count_tsv", type=str)
    parser_stats.add_argument("-m", "--metadata", type=str)
    parser_stats.add_argument("-t", "--otu_taxonomy_tsv", type=str)
    parser_stats.add_argument("-f", "--fig_dir", type=str)
    parser_stats.add_argument(
        "-s", "--sample_group_key", type=str, default="sample_group"
    )
    parser_stats.add_argument("-r", "--rep_group_key", type=str, default="rep_group")
    parser_stats.add_argument("-sp", "--spikein_taxa_key", type=str, default="spike_in")
    parser_stats.add_argument(
        "-l", "--tax_levels", nargs="+", default=["order", "family", "genus", "otu"]
    )
    parser_stats.add_argument("-rk", "--rarefying_key", type=str, default=None)
    parser_stats.add_argument("-rv", "--rarefying_value", type=int, default=None)
    parser_stats.add_argument("-rn", "--rarefying_repeat", type=int, default=0)
    parser_stats.add_argument("-p", "--plot_strip", action="store_true", default=False)

    args = parser.parse_args()

    otu_count_tsv = args.otu_count_tsv
    metadata_tsv = args.metadata
    otu_taxonomy_tsv = args.otu_taxonomy_tsv
    fig_dir = args.fig_dir
    rep_group_key = args.rep_group_key
    sample_group_key = args.sample_group_key
    spikein_taxa_key = args.spikein_taxa_key
    tax_levels = args.tax_levels

    # ====== read tables and preprocess ======
    df_otu_count, df_meta, df_tax = get_otu_count(
        otu_count_tsv,
        metadata_tsv,
        otu_taxonomy_tsv,
        sample_weight_key=None,
        spikein_taxa_key=spikein_taxa_key,
    )

    names, groups = zip(*[i for i in df_meta.groupby(sample_group_key, sort=False)])
    ratio = [i[rep_group_key].nunique() for i in groups]
    os.makedirs(fig_dir, exist_ok=True)

    if args.command == "abundance_group":
        rel_ab_thresholds = args.rel_ab_thresholds
        plot_type = args.plot_type
        sample_hierarchical_clustering = args.sample_hierarchical_clustering
        feature_hierarchy_clustering = args.feature_hierarchy_clustering

        # process into relative abundance and aggregate at replication group level
        df_otu_rel_ab_g = _agg_along_axis(
            df_otu_count.div(df_otu_count.sum(axis=1), axis=0),
            df_meta[sample_group_key] + "\t" + df_meta[rep_group_key],
            axis=0,
        )
        if len(rel_ab_thresholds) == 1:
            rel_ab_thresholds = rel_ab_thresholds * len(tax_levels)

        # an empirical way to determine width for pretty figure
        width = df_otu_rel_ab_g.shape[0] / 4 + 1.5
        for level, rel_ab_thres in zip(tax_levels, rel_ab_thresholds):
            # aggregate at taxonomic level
            res = _taxa_qc(
                _agg_along_axis(df_otu_rel_ab_g, df_tax[level], axis=1),
                rel_ab_thres,
                keep_rare=True,
                keep_unknown=False,
            )
            res = res[sorted(res.columns)]
            res_group_list = [
                res.loc[name + "\t" + group[rep_group_key].unique()]
                for name, group in zip(names, groups)
            ]
            # fix index
            for res_group in res_group_list:
                res_group.index = res_group.index.map(lambda x: x.split("\t")[1])

            num_cols = len(res_group_list)
            # wspace, hspace = 0.1, 0.01

            if plot_type in ["stacked_bar", "all"]:
                # stacked bar plot
                fig_size = (width, 4)
                custom_palette = (
                    sns.color_palette("tab20", 20)
                    + sns.color_palette("tab20b", 20)
                    + sns.color_palette("tab20c", 20)
                )
                fig, axs = _get_subplots(num_cols, fig_size, ratio, height_ratios=None)
                _stacked_bar(
                    res_group_list,
                    names,
                    f"Taxonomy at {level if level != 'otu' else level.upper()} level",
                    custom_palette,
                    axs,
                )
                fig.savefig(
                    f"{fig_dir}/rel_ab_group_{level}_sb.png",
                    bbox_inches="tight",
                    dpi=300,
                )
                fig.savefig(
                    f"{fig_dir}/rel_ab_group_{level}_sb.pdf",
                    bbox_inches="tight",
                    dpi=300,
                )
            if plot_type in ["heatmap", "heatmap_log10", "heatmap_binary", "all"]:
                size = (
                    res.shape[0] // (res.shape[0] / width),
                    res.shape[1] // (res.shape[0] / width),
                )
                if plot_type in ["heatmap_log10", "all"]:
                    if sample_hierarchical_clustering:
                        fig, (axs_dend, axs) = _get_subplots(num_cols, size, ratio)
                    else:
                        fig, (axs_dend, axs) = _get_subplots(
                            num_cols, size, ratio, height_ratios=None
                        )
                    # pesudo_abundance = 1e-4
                    vmax, vmin = 1, 1e-4

                    def _get_log10(arr: pd.DataFrame) -> pd.DataFrame:
                        # take log10, change -inf to log10(vmin), take transpose
                        arr = arr.copy()
                        arr[arr == 0] = vmin
                        return np.log10(arr)

                    dfs = [
                        _get_log10(res_group.transpose())
                        for res_group in res_group_list
                    ]
                    df = pd.concat(dfs, axis=1)
                    if feature_hierarchy_clustering:
                        df, linkage_row = _get_dendrogram(
                            df, rows_to_ignore=["unknown", "others"]
                        )
                        dfs_temp = []
                        i = 0
                        for df_temp in dfs:
                            dfs_temp.append(df.iloc[:, i : i + df_temp.shape[1]])
                            i += df_temp.shape[1]
                        dfs = dfs_temp
                    else:
                        linkage_row = None
                    _heatmap(
                        dfs,
                        names,
                        title=f"Taxonomy at {level} level",
                        cbar_label=f"log10(relative abundance)\n(range: [{vmin:.0e}, {vmax}])",
                        fig=fig,
                        axs_heatmap=axs,
                        axs_dend=axs_dend,
                        cmap="rocket_r",
                        linkage_row=linkage_row,
                        vmax=np.log10(vmax),
                        vmin=np.log10(vmin),
                    )
                    fig.savefig(
                        f"{fig_dir}/rel_ab_group_{level}_hml.png",
                        bbox_inches="tight",
                        dpi=300,
                    )
                    fig.savefig(
                        f"{fig_dir}/rel_ab_group_{level}_hml.pdf",
                        bbox_inches="tight",
                        dpi=300,
                    )
                if plot_type in ["heatmap"]:
                    if sample_hierarchical_clustering:
                        fig, (axs_dend, axs) = _get_subplots(num_cols, size, ratio)
                    else:
                        fig, (axs_dend, axs) = _get_subplots(
                            num_cols, size, ratio, height_ratios=None
                        )
                    dfs = ([res_group.T for res_group in res_group_list],)
                    df = pd.concat(dfs, axis=1)
                    if feature_hierarchy_clustering:
                        df, linkage_row = _get_dendrogram(
                            df, rows_to_ignore=["unknown", "others"]
                        )
                        dfs_temp = []
                        i = 0
                        for df_temp in dfs:
                            dfs_temp.append(df.iloc[:, i : i + df_temp.shape[1]])
                            i += df_temp.shape[1]
                        dfs = dfs_temp
                    else:
                        linkage_row = None
                    _heatmap(
                        dfs,
                        names,
                        title=f"Taxonomy at {level} level",
                        cbar_label="relative abundance",
                        fig=fig,
                        axs_heatmap=axs_heatmap,
                        axs_dend=axs_dend,
                        cmap=None,
                        linkage_row=linkage_row,
                    )
                    fig.savefig(
                        f"{fig_dir}/rel_ab_group_{level}_hm.png",
                        bbox_inches="tight",
                        dpi=300,
                    )
                    fig.savefig(
                        f"{fig_dir}/rel_ab_group_{level}_hm.pdf",
                        bbox_inches="tight",
                        dpi=300,
                    )
                if plot_type in ["heatmap_binary"]:
                    size = size[0], size[1] / 2
                    if sample_hierarchical_clustering:
                        fig, (axs_dend, axs) = _get_subplots(num_cols, size, ratio)
                        axs_heatmap, axs_dend = axs
                    else:
                        fig, (axs_dend, axs) = _get_subplots(
                            num_cols, size, ratio, height_ratios=None
                        )
                        axs_heatmap = axs
                        axs_dend = [None] * num_cols
                    dfs = [
                        (res_group > 0).astype(int).T for res_group in res_group_list
                    ]
                    df = pd.concat(dfs, axis=1)
                    if feature_hierarchy_clustering:
                        df, linkage_row = _get_dendrogram(
                            df, rows_to_ignore=["unknown", "others"]
                        )
                        dfs_temp = []
                        i = 0
                        for df_temp in dfs:
                            dfs_temp.append(df.iloc[:, i : i + df_temp.shape[1]])
                            i += df_temp.shape[1]
                        dfs = dfs_temp
                    else:
                        linkage_row = None
                    _heatmap(
                        dfs,
                        names,
                        title=f"Taxonomy at {level} level",
                        fig=fig,
                        axs_heatmap=axs_heatmap,
                        axs_dend=axs_dend,
                        cmap="BuPu",
                        linkage_row=linkage_row,
                    )
                    fig.savefig(
                        f"{fig_dir}/rel_ab_group_{level}_hmb.png",
                        bbox_inches="tight",
                        dpi=300,
                    )
                    fig.savefig(
                        f"{fig_dir}/rel_ab_group_{level}_hmb.pdf",
                        bbox_inches="tight",
                        dpi=300,
                    )
    elif args.command == "stats_sample_count":
        plot_strip = args.plot_strip
        rarefying_key = args.rarefying_key
        rarefying_value = args.rarefying_value
        rarefying_repeat = args.rarefying_repeat

        width = df_meta.groupby([sample_group_key, rep_group_key]).ngroups / 3 + 1.5

        if rarefying_repeat > 1:
            # Rarefying takes place by the following order:
            # - When rarefying_key is specified:
            #    - When this key is a string, rarefy to the depth of the sample corresponding to the key.
            #    - When this key is a int, rarefy to this value.
            #    - When this key is None, rarefy to `rarefying_value` if it is
            #        specified, otherwise rarefy to the minimum depth of all samples.
            # - When rarefying_key isn't specified, treat as if rarefying_key is None
            #     for all samples.
            depth = df_otu_count.sum(axis=1)
            if rarefying_value is None:
                rarefying_value = int(depth.min()) - 1

            ref = []
            if rarefying_key is None:
                rarefying_keys = [np.nan] * len(df_otu_count)
            else:
                rarefying_keys = df_meta[rarefying_key].tolist()

            for i in rarefying_keys:
                if isinstance(i, str):
                    if not i in df_otu_count.index:
                        raise ValueError(f"Reference sample {i} is not available.")
                    ref.append(depth[i])
                elif isinstance(i, int):
                    ref.append(i)
                elif np.isnan(i):
                    ref.append(rarefying_value)
                else:
                    raise ValueError(
                        f"Unknown data type for rarefying_key: {i} ({type(i)})"
                    )

            # rarefy
            df_otu_count_orig = df_otu_count.copy()
            df_meta_orig = df_meta.copy()
            df_otu_count, names_orig = _rarefying(df_otu_count, ref, rarefying_repeat)
            # NOTE:
            # Must create the dummy dataframe as a column, cannot be empty dataframe
            # with index, otherwise order of merged dataframe index will be slightly
            # wrong.
            df_meta = pd.merge(
                pd.DataFrame({"original_sample_name": names_orig}),
                df_meta,
                left_on="original_sample_name",
                right_index=True,
                validate="many_to_one",
            ).reset_index(names=df_meta.index.name)
            df_meta.index = df_otu_count.index
        else:
            df_otu_count_orig = None
            df_meta_orig = None

        wspace = 0.15
        for level in tax_levels:
            # no need for taxa qc, all taxa count
            res = _agg_along_axis(df_otu_count, df_tax[level], axis=1)
            meta_l = pd.merge(
                df_meta,
                _calc_alpha_metrics(res),
                left_index=True,
                right_index=True,
            )
            _, groups = zip(*[i for i in meta_l.groupby(sample_group_key, sort=False)])
            title_prefix = f"{'ZOTU' if level == 'otu' else level.capitalize()}"
            for column_of_interest, title in zip(
                ["shannon", "simpson", "richness", "chao1"],
                ["Shannon entropy", "Simpson's index", "Richness", "Chao1 index"],
            ):
                title = f"{title_prefix} {title}"
                fig, axs = plt.subplots(
                    1, len(groups), sharey="row", width_ratios=ratio, figsize=(width, 3)
                )
                if len(groups) == 1:
                    axs = [axs]
                _barplot_with_whisker_strip(
                    groups,
                    names=names,
                    title=title,
                    group_key=rep_group_key,
                    column_of_interest=column_of_interest,
                    plot_strip=plot_strip,
                    axs=axs,
                    ylog=False,
                )
                fig.subplots_adjust(wspace=wspace)
                for format_ in ["png", "pdf"]:
                    fig.savefig(
                        f"{fig_dir}/{column_of_interest}_{level}.{format_}",
                        bbox_inches="tight",
                        dpi=300,
                    )
                # close figure to save memory
                plt.close(fig)

        # plot sequencing depth separately
        if df_otu_count_orig is not None:
            df_otu_count = df_otu_count_orig
            df_meta = df_meta_orig
        res = df_otu_count  # no need to aggregate
        meta_l = pd.merge(
            df_meta,
            _calc_alpha_metrics(res),
            left_index=True,
            right_index=True,
        )
        _, groups = zip(*[i for i in meta_l.groupby(sample_group_key, sort=False)])
        fig, axs = plt.subplots(
            1, len(groups), sharey="row", width_ratios=ratio, figsize=(width, 3)
        )
        if len(groups) == 1:
            axs = [axs]
        fig.subplots_adjust(wspace=wspace)
        _barplot_with_whisker_strip(
            groups,
            names=names,
            title="Sequencing depth",
            group_key=rep_group_key,
            column_of_interest="sequencing_depth",
            plot_strip=plot_strip,
            axs=axs,
            ylog=True,
        )
        for format_ in ["png", "pdf"]:
            fig.savefig(
                f"{fig_dir}/sequencing_depth.{format_}",
                bbox_inches="tight",
                dpi=300,
            )
        plt.close(fig)
    else:
        raise ValueError("Unknown command.")
