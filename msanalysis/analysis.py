import logging
import pandas as pd
import numpy as np
from natsort import natsorted
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
import math
import seaborn as sns
import scipy

from matplotlib.colors import LogNorm

from mapseq.utils import *
from mapseq.stats import *
from mapseq.core import *


def make_heatmaps_combined_sns(config, sampdf, infiles, outfile=None, outdir=None, expid=None, 
                               recursion=200000, combined_pdf=True ):
    # def make_merged_plots_new(config, outdir=None, expid=None, recursion=200000, combined_pdf=True, label_column='region' ):
    '''
    consume barcode matrices and create heatmap plots.
    Will label plots with leading portion of filename, assuming it is a brain id. 
    
    '''
    from matplotlib.backends.backend_pdf import PdfPages as pdfpages
    sys.setrecursionlimit(recursion)  
    
    clustermap_scale = config.get('plots','clustermap_scale') # log10 | log2
    cmap = config.get('plots','heatmap_cmap')


    if outfile is None:
        outfile = 'heatmaps.pdf'
    if outdir is None:
        outdir = os.path.dirname(infiles[0])

    page_dims = (11.7, 8.27)
    with pdfpages(outfile) as pdfpages:
        for filepath in infiles:
            filepath = os.path.abspath(filepath)    
            dirname = os.path.dirname(filepath)
            filename = os.path.basename(filepath)
            (base, ext) = os.path.splitext(filename) 
            brain_id = base.split('.')[0]
            logging.debug(f'handling file base/brain: {brain_id}')
            scbcmdf = load_df(filepath)          
            logging.info(f'plotting file: {filename}')    

            # check to ensure no columns are missing barcodes, since that messes up
            # clustermaps
            droplist = []
            for c in scbcmdf.columns:
                if not scbcmdf[c].sum() > 0:
                    logging.warn(f'columns {c} for brain {brain_id} has no barcodes, dropping...')
                    droplist.append(c)
            logging.debug(f'dropping columns {droplist}')
            scbcmdf.drop(droplist,inplace=True, axis=1 )             
            num_vbcs = len(scbcmdf.index)

            try:
                kws = dict(cbar_kws=dict(orientation='horizontal'))  
                g = sns.clustermap(scbcmdf, cmap=cmap, yticklabels=False, col_cluster=False, standard_scale=1, **kws)
                #g.ax_cbar.set_title('scaled log10(cts)')
                x0, _y0, _w, _h = g.cbar_pos
                #g.ax_cbar.set_position((0.8, .2, .03, .4))
                g.ax_cbar.set_position([x0, 0.9, g.ax_row_dendrogram.get_position().width, 0.05])
                g.fig.suptitle(f'{expid} brain={brain_id}\n{num_vbcs} VBCs total')
                g.ax_heatmap.set_title(f'Scaled {clustermap_scale}(umi_count)')
                plt.savefig(f'{outdir}/{brain_id}.{clustermap_scale}.clustermap.pdf')
                if combined_pdf:
                    logging.info(f'saving plot to {outfile} ...')
                    pdfpages.savefig(g.fig)
            except Exception as ee:
                logging.warning(f'Unable to clustermap plot for {brain_id}. Message: {ee}')


def apply_binarize(val):
    if val == 0.0:
        return 0
    else:
        return 1


def apply_encode(row):
    '''
    normally axis=1 (row)
    '''
    sum = 0
    m = 1
    for i in row:
        sum += i * m
        m += 1
    return sum
        
 
#
#  Code to make plots
#  These accept the normalized barcode matrix. (real normalized by spike-ins). 
#
#  definitions. 
#  projection frequency   number of distinct neurons (VBCs) to target 
#  projection strength    number of unique VBC molecules, irrespective of neurons (# of unique VBCs)
#  projection intensity   number of unique VBC molecules for a given neuron (unique VBC)

def boolean_sort(df):
    '''
    assumes numeric dataframe
    sorts rows by zero/non-zero presense/absence. 
    returns sorted original dataframe 
    
    '''
    # make binarized df
    bdf = pd.DataFrame(columns=list(df.columns), dtype='uint8')
    for col in list(df.columns):
        bdf[col] = df[col].apply(apply_binarize)
    sbdf = bdf.sort_values(by=list(bdf.columns))
    logging.debug(f'sorted binarized DF = {sbdf}')    

    # sort all by presence in target areas. 
    # by mult. value by   1 10 100 1000 ... 
    totals = np.ones(len(bdf))
    for i, colname in enumerate(bdf.columns):
        m = math.pow(10,i) 
        x = bdf[colname] * m 
        totals = totals + x.values
    df['code'] = totals
    sdf = df.sort_values('code')
    sdf.drop('code', axis=1, inplace=True)
    return sdf 


def binarize_sort_matrix(df):
    '''
    return sorted binarized matrix. 
    '''
    bdf = pd.DataFrame(columns=list(df.columns))
    for col in list(df.columns):
        bdf[col] = df[col].apply(apply_binarize).astype('int8')
    sbdf = bdf.sort_values(by=list(bdf.columns), ascending=False)
    logging.debug(f'sorted binarized DF =\n{sbdf}')
    return sbdf


def threshold_binarized(df, min_rows=10):
    '''
    take sorted, binarized matrix
    assumes dtype is int8
    return only rows where rows in the pattern are greater than min_rows
    '''
    ### using index
    #stbdf = df.astype('string')
    #clist = list( stbdf.columns )
    #clist.reverse()
    #stbdf.index = stbdf[ clist ].agg(''.join, axis=1)
    #stbdf = stbdf.astype('int8')
    #booldf = stbdf.groupby(by=stbdf.index, sort=False).sum() > min_rows
    #idxlist = list( (booldf.any(axis=1) == True).index )
    #outdf = stbdf[stbdf.index.isin(idxlist)]
    
    # use throwaway column
    stbdf = df.astype('string')
    clist = list( stbdf.columns )
    vclist = list( stbdf.columns )
    stbdf['scode'] = stbdf[ clist ].agg(''.join, axis=1)
    mapping = { k : 'int8' for k in vclist }
    stbdf = stbdf.astype(mapping)
    
    b = stbdf.groupby(by=stbdf['scode'], sort=False).size() > min_rows
    scodelist = list( b[b.values == True].index)
    outdf = stbdf[ stbdf['scode'].isin(scodelist) ].copy()
    outdf.drop(columns='scode', inplace=True)
    return outdf
    


def plot_binarized(sbdf, expid=None, info='binarized'):
    '''
    sorted, binarized matrix -> plot graphic
    '''
    if expid is None:
        expid = 'M000'

    num_vbcs = len(sbdf)
    #kws = dict(cbar_kws=dict(orientation='horizontal'))  
    plt.figure(figsize=(10,8))
    #norm = LogNorm()
    #g = sns.heatmap(sbdf, yticklabels=False, norm=norm, cmap='Blues',cbar=False)
    g = sns.heatmap(sbdf, yticklabels=False, cmap='Blues',cbar=False)
    #g.ax_cbar.set_title('scaled log10(cts)')
    #x0, _y0, _w, _h = g.cbar_pos
    #g.ax_cbar.set_position((0.8, .2, .03, .4))
    #g.ax_cbar.set_position([x0, 0.9, g.ax_row_dendrogram.get_position().width, 0.05])
    g.figure.suptitle(f'{expid}\n{num_vbcs} VBCs\n{info}')
    #g.ax_heatmap.set_title(f'Scaled {clustermap_scale}(umi_count)')       
    return g


def get_plot_binarized(nbcdf, expid=None, info='binarized'):
    '''
    input: normalized matrix. 
    output: plot object
    
    '''
    if expid is None:
        expid = 'M000'
    nbcdf = nbcdf.astype('float')
    sbdf = binarize_sort_matrix(nbcdf )
    g = plot_binarized(sbdf, expid = expid, info=info)
    return g


def make_plot_binarized(config, infile, outfile=None, expid=None, sampdf=None ):
    '''
    take normalized barcode matrix (nbcm.tsv) and do plot PDF. 
    
    '''    
    import matplotlib.pyplot as plt

    logging.debug(f'make_plot_binarized(): infile={infile} outfile={outfile} expid={expid}')

    if expid is None:
        expid = 'M000'
      
    filepath = os.path.abspath(infile)    
    dirname = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    (base, ext) = os.path.splitext(filename) 
       
    if outfile is None:
        outfile = f'{dirname}/{expid}.binarized.pdf'
    
    nbcdf = load_df(filepath)          
    logging.info(f'plotting file: {filename}')
    logging.debug(f'inbound DF = {nbcdf}')   
    g = get_plot_binarized(nbcdf, expid )
    plt.savefig(outfile)
    logging.info(f'wrote plot(s) to {outfile}')    



def make_plot_binarized_motifs(config, infile, 
                               outfile=None, 
                               expid=None, 
                               min_targets=2, 
                               max_targets=5, 
                               min_rows=20 ):
    '''
    take normalized barcode matrix (nbcm.tsv)
    Remove all VBCs that project to less than <n_motifs> targets. 
    make plot of remainder.      
    
    '''    
    import matplotlib.pyplot as plt

    logging.debug(f'make_plot_binarized(): infile={infile} outfile={outfile} expid={expid}')

    if expid is None:
        expid = 'M000'
      
    filepath = os.path.abspath(infile)    
    dirname = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    (base, ext) = os.path.splitext(filename) 
       
    if outfile is None:
        outfile = f'{dirname}/{expid}.binarized.pdf'
    
    nbcdf = load_df(filepath) 
    nbcdf = nbcdf.astype('float')
             
    logging.info(f'plotting file: {filename}')
    logging.debug(f'inbound DF = {nbcdf}')

    sbdf = binarize_sort_matrix(nbcdf )
    fdf = sbdf[ sbdf.sum(axis=1) < max_targets ]
    fdf = fdf[fdf.sum(axis=1) > min_targets ]
    fdf = threshold_binarized(fdf, min_rows=min_rows)
    #g = plot_binarized(fdf, label = label)
    g = get_plot_binarized(fdf, expid )
    plt.savefig(outfile)
    logging.info(f'wrote plot(s) to {outfile}')    



def plot_clustermap(nbcdf, exp_id, brain_id):
    '''
    https://github.com/mwaskom/seaborn/issues/1207
    
    '''
    num_vbcs = len(nbcdf)
    plt.figure(figsize=(10,8))
    g = sns.clustermap(nbcdf, yticklabels=False, cmap='Blues',cbar=False, z_score=1)    
    g.figure.suptitle(f'{exp_id} brain={brain_id}\nBooleansorted\n{num_vbcs} VBCs total')


def plot_frequency_heatmap(nbcdf, exp_id, brain_id):
    '''
    rectangular heatmap, with square colors representing molecule counts. 
    '''
    num_vbcs = len(nbcdf)
    bdf = pd.DataFrame(columns=list(nbcdf.columns))
    for col in list(nbcdf.columns):
        bdf[col] = nbcdf[col].apply(apply_binarize).astype('uint8')
    nbcdf['code'] = bdf.apply(apply_encode, axis=1) 
    
      
    sdf = boolean_sort(nbcdf)
    #logging.debug(f'sortedf =\n{sdf}')
    #kws = dict(cbar_kws=dict(orientation='horizontal'))  
    plt.figure(figsize=(10,8))
    g = sns.heatmap(sdf, yticklabels=False, cmap='Blues',cbar=False)
    #g.ax_cbar.set_title('scaled log10(cts)')
    #x0, _y0, _w, _h = g.cbar_pos
    #g.ax_cbar.set_position((0.8, .2, .03, .4))
    #g.ax_cbar.set_position([x0, 0.9, g.ax_row_dendrogram.get_position().width, 0.05])
    g.figure.suptitle(f'{exp_id} brain={brain_id}\nBooleansorted\n{num_vbcs} VBCs total')
    #g.ax_heatmap.set_title(f'Scaled {clustermap_scale}(umi_count)')       
    return g


def normalize_log_z(nbcdf, logscale='log10'):
    '''
    log scales all values. 
    compresses them to 0.0 - 1.0
    
    '''
    lsdf = normalize_scale(nbcdf, 'log10')
    max = lsdf.max().max()
    lsdf = lsdf / max
    return lsdf


def make_freq_lineplot(df, logx=False, logy=True, col='count', title=None):
    '''
   
    
    '''
    plt.figure(figsize=(10,8))
    yvals = ser.values
    if logy:
        yvals = np.log10(yvals)
    xvals = ser.index
    if logx:
        xvals = np.log10(xvals)
        
    g = sns.lineplot(data = df, x=df.index, y = yvals)
    g.figure.suptitle(f'{title}')     
    return g
    



def make_plots(config, sampdf, infile, outfile=None, outdir=None, exp_id=None ):
    '''
    take normalized barcode matrix (nbcm.tsv) and do plots. 
    
    '''    
    from matplotlib.backends.backend_pdf import PdfPages as pdfpages
    import matplotlib.pyplot as plt

    logging.debug(f'make_plots(): infile={infile} outfile={outfile} outdir={outdir} exp_id={exp_id}')

    if exp_id is None:
        exp_id = 'MAPseqEXP'
      
    filepath = os.path.abspath(infile)    
    dirname = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    (base, ext) = os.path.splitext(filename) 
    
    if outdir is None:
        outdir = './'
    
    if outfile is None:
        outfile = f'{outdir}{expid}.plots.pdf'
    
    brain_id = base.split('.')[0]
    logging.debug(f'handling file base/brain: {brain_id}, writing {outfile}')
    nbcmdf = load_df(filepath)          
    logging.info(f'plotting file: {filename}')
    logging.debug(f'inbound DF = {nbcmdf}')   
    
    
    page_dims = (11.7, 8.27)
    with pdfpages(outfile) as pdfpages:
        #g = plot_frequency_heatmap(nbcmdf, exp_id, brain_id)
        #pdfpages.savefig(g.figure)
        g = plot_binarized(nbcmdf, exp_id, brain_id )
        pdfpages.savefig(g.figure)

    
    logging.info(f'wrote plot(s) to {outfile}')    
    #plt.imshow(sbdf.values, interpolation="nearest", cmap='Blues', aspect='auto')
    #plt.show()
      
    
    
    
    
    