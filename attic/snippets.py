
#
#  Function from Emily Isko
#
def sorted_heatmap(df, title=None, sort_by=['type'], sort_ascend=True, drop=['type'], 
                   nsample=None, random_state=10, log=False, cmap=orange_cmp, cbar=False,
                   label_neurons=None, col_order=None, rasterized=True):
    """_summary_

    Args:
        df (DataFrame): Dataframe to plot heatmap
        sort_by (list, optional): How to sort neurons. Defaults to ['type'].
        sort_ascend (boolean, optional): whether to sort ascending or descending. Defaults to True.
        drop (list, optional): What columns to drop before plotting. Defaults to ['type'].
        nsample (int, optional): If present, down sample dataframe. Defaults to None.
        random_state (int, optional): If downsample, what random state to use. Defaults to 10.
        log (bool, optional): Whether to plot log scale. Defaults to None.
        cmap (colormap, optional): What colormap to use for plotting. Defaults orange_cmp.
        cbar (boolean, optional): Whether to plot cbar or not. Defaults to False.
        label_neurons (dict, option): Dictionary of label:index of neurons to label
        col_order (list, optional): Order to plot columns. Defaults to None.
        rasterized (boolean, optional): Wheter to rasterize plot. Defaults to True.
    """

    if nsample:
        plot = df.sample(nsample, random_state=random_state)
    else:
        plot = df.copy()

    if log:
        plot = plot.replace({"IT":1, "CT":10, "PT":100})
        norm = LogNorm()
    else:
        plot = plot.replace({"IT":0.25, "CT":0.5, "PT":0.75})
        norm = None

    plot = plot.sort_values(by=sort_by, ascending=sort_ascend)
    plot = plot.reset_index(drop=True)
    out_plot = plot.copy()


    # reorder cols if given
    if col_order:
        plot = plot[col_order]

    fig=plt.subplot()
    sns.heatmap(plot.drop(drop, axis=1), norm=norm, cmap=cmap, cbar=cbar, rasterized=rasterized)
    plt.gca().get_yaxis().set_visible(False)
    plt.title(title)
    if label_neurons:
        for key in label_neurons.keys():
            plt.text(-0.3,label_neurons[key], "-", va="center_baseline", size=15)
            plt.text(-0.75,label_neurons[key], key, va="center_baseline", size=12)
    return(out_plot, fig)



#
#  Function from Emily Isko
#
def sort_by_celltype(proj, it_areas=["OMCc", "AUD", "STR"], ct_areas=["TH"],
                      pt_areas=["AMY","HY","SNr","SCm","PG","PAG","BS"],
                      sort=True):
    """
    Function takes in projection matrix and outputs matrix sorted by the 3 major celltypes:
    - IT = intratelencephalic (projects to cortical and/or Striatum), type = 10
    - CT = corticalthalamic (projects to thalamus w/o projection to brainstem), type = 100
    - PT = pyramidal tract (projects to brainstem += other areas), type = 1000
    Returns single dataframe with cells sorted and labelled by 3 cell types (IT/CT/PT)

    Args:
        proj (DataFrame): pd.DataFrame of BC x area. Entries can be normalized BC or binary.
        it_areas (list, optional): Areas to determine IT cells. Defaults to ["OMCc", "AUD", "STR"].
        ct_areas (list, optional): Areas to determine CT cells. Defaults to ["TH"]. Don't actually use this...
        pt_areas (list, optional): Areas to determine PT cells. Defaults to ["AMY","HY","SNr","SCm","PG","PAG","BS"].
        sort (bool, optional): Whether to sort by cell type or return w/ original index. Defaults to True.
    """
    
    ds=proj.copy()
 
    # Isolate PT cells
    pt_counts = ds[pt_areas].sum(axis=1)
    pt_idx = ds[pt_counts>0].index
    ds_pt = ds.loc[pt_idx,:]
    ds_pt['type'] = "PT"

    # Isolate remaining non-PT cells
    ds_npt = ds.drop(pt_idx)

    # Identify CT cells by thalamus projection
    th_idx = ds_npt['TH'] > 0
    ds_th = ds_npt[th_idx]
    if sort:
        ds_th = ds_th.sort_values('TH', ascending=False)
    ds_th['type'] = "CT"

    # Identify IT cells by the remaining cells (non-PT, non-CT)
    ds_nth = ds_npt[~th_idx]
    if sort:
        ds_nth = ds_nth.sort_values(it_areas,ascending=False)
    ds_nth['type'] = "IT"

    # combine IT and CT cells
    ds_npt = pd.concat([ds_nth, ds_th])

    # combine IT/CT and PT cells
    if sort:
        sorted = pd.concat([ds_npt,ds_pt],ignore_index=True)
        df_out=sorted.reset_index(drop=True)
    else:
        df_out = pd.concat([ds_npt,ds_pt]).sort_index()

    return(df_out)




def plot_viruslib(barcodematrixtsv):
    #
    # frequency plot of 
    #
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    dirpath, base, ext = split_path(barcodematrixtsv)
    
    df = load_df(barcodematrixtsv)
    rowsum = df.sum(axis=1)
    rowsort = rowsum.sort_values(ascending=False)
    df = pd.DataFrame(data=rowsort)
    df.columns = ['counts']
    df.reset_index(drop=True, inplace=True)
    df['sequence'] = df.index +1
    df['log_sequence'] = np.log2(df['sequence'])
    ax = sns.lineplot(data=df, x=df.log_sequence, y=df.counts)
    ax.set_title(f'{base} counts frequency')
    

