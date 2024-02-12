import logging
import pandas as pd
import numpy as np
from natsort import natsorted
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
import math
import seaborn as sns
import scipy

from mapseq.utils import *
from mapseq.stats import *


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


def binarize(val):
    if val == 0.0:
        return 0
    else:
        return 1

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
    bdf = pd.DataFrame(columns=list(df.columns))
    for col in list(df.columns):
        bdf[col] = df[col].apply(binarize)
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
    sdf.drop('code', axis=1, inplace =True)
    return sdf 

 

def plot_binarized(nbcdf, exp_id, brain_id):
    '''
    all VBCs represented by rows, binarized. 
    
    '''
    bdf = pd.DataFrame(columns=list(nbcdf.columns))
    for col in list(nbcdf.columns):
        bdf[col] = nbcdf[col].apply(binarize)
    sbdf = bdf.sort_values(by=list(bdf.columns))
    logging.debug(f'sorted binarized DF = {sbdf}')
    num_vbcs = len(sbdf)
    #kws = dict(cbar_kws=dict(orientation='horizontal'))  
    plt.figure(figsize=(10,8))
    g = sns.heatmap(sbdf, yticklabels=False, cmap='Blues',cbar=False)
    #g.ax_cbar.set_title('scaled log10(cts)')
    #x0, _y0, _w, _h = g.cbar_pos
    #g.ax_cbar.set_position((0.8, .2, .03, .4))
    #g.ax_cbar.set_position([x0, 0.9, g.ax_row_dendrogram.get_position().width, 0.05])
    g.figure.suptitle(f'{exp_id} brain={brain_id}\nBinarized\n{num_vbcs} VBCs total')
    #g.ax_heatmap.set_title(f'Scaled {clustermap_scale}(umi_count)')       
    return g


def plot_frequency_heatmap(nbcdf, exp_id, brain_id):
    '''
    rectangular heatmap, with square colors representing molecule counts. 
    
    '''
    num_vbcs = len(nbcdf)
    sdf = boolean_sort(nbcdf)
    logging.debug(f'sortedf =\n{sdf}')
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
        g = plot_binarized(nbcmdf, exp_id, brain_id )
        pdfpages.savefig(g.figure)
        g = plot_frequency_heatmap(nbcmdf, exp_id, brain_id)
        pdfpages.savefig(g.figure)
    
    logging.info(f'wrote plot(s) to {outfile}')    
    #plt.imshow(sbdf.values, interpolation="nearest", cmap='Blues', aspect='auto')
    #plt.show()
      
    
    
    
    
    