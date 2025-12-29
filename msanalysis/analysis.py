import logging
import math

import pandas as pd
import numpy as np
import seaborn as sns
import scipy
from natsort import natsorted


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages as pdfpages   
logging.getLogger('matplotlib').setLevel(logging.WARNING)    
from matplotlib.colors import LogNorm

from mapseq.utils import *
from mapseq.stats import *
from mapseq.core import *

SAMPLEINFO_COLUMNS = [  'usertube', 
                        'ourtube', 
                        'samplename', 
                        'siteinfo', 
                        'rtprimer', 
                        'brain', 
                        'region', 
                        'matrixcolumn'] 

####################################################
#
#                     UTILITIES
#
####################################################

def get_mainbase(filepath):
    '''
    for any full or relative filename XXXXX.y.z.w.ext 
    returns just XXXXX (first portion before dot). 
    
    '''
    dirname = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    (base, ext) = os.path.splitext(filename)
    base = base.split('.')[0]
    return base

def apply_binarize(val):
    if val == 0.0:
        return 0
    else:
        return 1

def apply_encode(row):
    '''
    normally axis=1 (row)
    Assumes binary input vlues in row (0|1)
    Returns coded sum of totals.  0 1 1 0 1  -> 1101.0 
    
    '''
    sum = 0
    m = 1
    for i, v in enumerate( row ):
        m = m * 10
        sum += v * m
    return sum

def normalize_log_z(nbcdf, logscale='log10'):
    '''
    log scales all values. 
    compresses them to 0.0 - 1.0
    
    '''
    lsdf = normalize_scale(nbcdf, 'log10')
    max = lsdf.max().max()
    lsdf = lsdf / max
    return lsdf

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
    apply binarization to dataframe. 
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
    
#
# HIGH-LEVEL BARCODE MATRIX PLOTTING 
#
#  These accept the normalized barcode matrix. (real normalized by spike-ins). 
#
#  definitions. 
#  projection frequency   number of distinct neurons (VBCs) to target 
#  projection strength    number of unique VBC molecules, irrespective of neurons (# of unique VBCs)
#  projection intensity   number of unique VBC molecules for a given neuron (unique VBC)
#
#

def make_plots_pdf(infiles,
                   outfile,
                   binarized=True,
                   motif_min=0,
                   motif_max=99,
                   sampdf=None, 
                   label_column=None,
                   cp=None
                   ):
    '''
    @arg infiles        Barcode matrix files. reals, spikes, or normalized.
    @arg outfile        Outfile name.  Will respect .pdf or .png
    @arg sampdf         Standard samplefile .xlsx or .tsv
    @arg label_column   Column in sampdf to map to. [ label | region ] 
    @arg cp             Standard mapseq config. 
    '''


def make_binarized_plots_pdf(infiles, 
                             outfile,  
                             label_column=None, 
                             sampdf=None,
                             titletext=None,  
                             cp=None):
    '''
    Make appropriate plot files for all input XYZ.nbcm.tsv files. 
    One per page. 

    
    '''    
    import matplotlib.pyplot as plt    
    from matplotlib.backends.backend_pdf import PdfPages as pdfpages

    logging.debug(f'make_binarized_plots_pdf(): infiles={infiles} outfile={outfile} label_column={label_column}')

    if cp is None:
        cp= get_default_config()

    project_id = cp.get('project','project_id')

    if outfile is None:
        outdir = os.path.abspath('./')
        outfile = f'{outdir}/{project_id}.binarized_plot.pdf'

    page_dims = (11.7, 8.27)
    with pdfpages(outfile) as pdfpages:
        for infile in infiles:
            brain = get_mainbase(infile)
            filepath = os.path.abspath(infile)
            label = f'{project_id} brain: {brain}'
            if titletext is not None:
                label += f'\n{titletext}'
            logging.info(f'plotting {infile} -> {outfile} ')
        
            nbcdf = load_mapseq_matrix_df(filepath)          
            #logging.info(f'plotting file: {filename}')
            logging.debug(f'inbound DF = {nbcdf} columns={nbcdf.columns}')   
            if label_column is not None:
                logging.debug('altering labels for plot columns.')
                if label_column in SAMPLEINFO_COLUMNS:
                    lcol = sampdf[label_column]
                    if sampdf is not None:
                        oldcolumns = nbcdf.columns
                        logging.debug(f'renaming columns to labels in sampleinfo...')
                        sampdf['bclabel'] = 'BC'+ sampdf['rtprimer']
                        newcolumns = []
                        for col in nbcdf.columns:
                            newval = sampdf[sampdf['bclabel'] == col][label_column].values[0]
                            if newval in newcolumns:
                                nidx = newcolumns.count(newval) + 1
                                newval = f'{newval}.{nidx}'
                            newcolumns.append( newval )
                        nbcdf.columns = newcolumns
                        scol = natsorted(list(nbcdf.columns))
                        nbcdf = nbcdf[scol]  
                        logging.debug(f'reset and sorted columns in matrix. old={oldcolumns} new={list(nbcdf.columns)} ')    
                    else:
                        logging.warning('no sampleinfo specified, so ignoring label.')
                else:
                    logging.warning(f'label_column {label_column} not valid sampleinfo column')
            else:
                logging.debug('No label specified, so using data column names.')
        
            ax = get_plot_binarized(nbcdf, label )
            #pdfpages.savefig(g.fig)
            pdfpages.savefig(ax.figure)
    logging.info(f'wrote plot(s) to {outfile}')


def make_heatmap_matrix_pdf(infiles, 
                            outfile,  
                            label_column=None, 
                            sampdf=None,
                            titletext=None,  
                            cp=None):
    '''
    Make appropriate plot files for all input XYZ.nbcm.tsv files. 
    One per page. 
    
    '''    
    import matplotlib.pyplot as plt    
    from matplotlib.backends.backend_pdf import PdfPages as pdfpages

    logging.debug(f'make_heatmap_plots_pdf(): infiles={infiles} outfile={outfile} label_column={label_column}')

    if cp is None:
        cp= get_default_config()

    project_id = cp.get('project','project_id')

    if outfile is None:
        outdir = os.path.abspath('./')
        outfile = f'{outdir}/{project_id}.heatmap_plot.pdf'

    page_dims = (11.7, 8.27)
    with pdfpages(outfile) as pdfpages:
        for infile in infiles:
            brain = get_mainbase(infile)
            filepath = os.path.abspath(infile)
            logging.info(f'plotting {infile} -> {outfile} ')
            nbcdf = load_mapseq_matrix_df(filepath)          
            label = f'brain={brain} n_vbcs={len(nbcdf)}'
            if titletext is not None:
                label += f'\n{titletext}'                        
            logging.debug(f'inbound DF = {nbcdf} columns={nbcdf.columns}')   
            if label_column is not None:
                logging.debug('altering labels for plot columns.')
                if label_column in SAMPLEINFO_COLUMNS:
                    lcol = sampdf[label_column]
                    if sampdf is not None:
                        oldcolumns = nbcdf.columns
                        logging.debug(f'renaming columns to labels in sampleinfo...')
                        sampdf['bclabel'] = 'BC'+ sampdf['rtprimer']
                        newcolumns = []
                        for col in nbcdf.columns:
                            newval = sampdf[sampdf['bclabel'] == col][label_column].values[0]
                            if newval in newcolumns:
                                nidx = newcolumns.count(newval) + 1
                                newval = f'{newval}.{nidx}'
                            newcolumns.append( newval )
                        nbcdf.columns = newcolumns
                        scol = natsorted(list(nbcdf.columns))
                        nbcdf = nbcdf[scol]  
                        logging.debug(f'reset and sorted columns in matrix. old={oldcolumns} new={list(nbcdf.columns)} ')    
                    else:
                        logging.warning('no sampleinfo specified, so ignoring label.')
                else:
                    logging.warning(f'label_column {label_column} not valid sampleinfo column')
            else:
                logging.debug('No label specified, so using data column names.')
        
            ax = get_plot_heatmap( nbcdf, label, cp=cp)
            pdfpages.savefig(ax.figure)
    logging.info(f'wrote plot(s) to {outfile}')


def make_binarized_motifs(config, infile, 
                               outfile=None, 
                               expid=None, 
                               min_targets=2, 
                               max_targets=5, 
                               min_rows=20 ):
    '''
    Take normalized barcode matrix (nbcm.tsv)
    Remove all VBCs that project to less than <min_targets> targets. 
    Remove all that project to more than <max_targets>
    
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


def make_clustermap_pdf( infiles,
                         sampdf, 
                         outfile,  
                         config ):
    '''

    
    '''
    from matplotlib.backends.backend_pdf import PdfPages as pdfpages
    sys.setrecursionlimit(recursion)  
    
    clustermap_scale = config.get('plots','clustermap_scale') # log10 | log2
    cmap = config.get('plots','heatmap_cmap', fallback='Blues')
    logging.debug(f'clustermap_scale={clustermap_scale} cmap={cmap}')

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

#
# DATAFRAME/DATA SUBFUNCTIONS
#





#
#  PLOTTING SUB-FUNCTIONS
#
def get_plot_binarized(nbcdf, expid=None, info='binarized'):
    '''
    input: normalized matrix. 
    output: plot object
    
    '''
    if expid is None:
        expid = 'M000'
    nbcdf = nbcdf.astype('float')
    sbdf = binarize_sort_matrix(nbcdf )
    ax = plot_binarized(sbdf, expid = expid, info=info)
    return ax

def get_plot_heatmap(nbcdf, label='heatmap', cp=None):
    '''
    input: normalized matrix. 
    output: plot object
    
    '''
    if cp is None:
        cp = get_default_config()
    cmap = cp.get('plots','heatmap_cmap')
    project_id = cp.get('project','project_id')
    logging.debug(f'cmap={cmap}')

    nbcdf = nbcdf.astype('float')
    adf = get_binarized_sorted_intensity_matrix(nbcdf, lognorm = True)    
    g = sns.heatmap(adf, cmap=cmap, cbar=False)
    g.figure.suptitle(f'{project_id}\n{label}')
    plt.xlabel('Targets')
    plt.ylabel('Binarized combinations') 
    return g

def plot_binarized(sbdf, expid=None, info='binarized'):
    '''
    sorted, binarized matrix -> plot graphic
    '''
    if expid is None:
        expid = 'M000'

    num_vbcs = len(sbdf)
  
    plt.figure(figsize=(10,8))
    ax = sns.heatmap(sbdf, yticklabels=False, cmap='Blues',cbar=False)
    ax.figure.suptitle(f'{expid}\n{num_vbcs} VBCs\n{info}')    
    return g


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
  
    plt.figure(figsize=(10,8))
    g = sns.heatmap(sdf, yticklabels=False, cmap='Blues',cbar=False)
    g.figure.suptitle(f'{exp_id} brain={brain_id}\nBooleansorted\n{num_vbcs} VBCs total')
    return g


def get_binarized_sorted_intensity_matrix(df, lognorm = True):
    '''
    Take matrix. 
    Binarize by bin_code 
    Aggregate by bin_code, summing cell values (umi counts or normalized). 
    '''
    cols = list(df.columns)
    bdf = pd.DataFrame(columns=cols, dtype='uint8')
    for col in cols:
        bdf[col] = df[col].apply(apply_binarize).astype('int8')
    bdf = bdf.sort_values(by=cols, ascending=False)
    sdf = df.reindex(bdf.index)
    
    sdf['code'] = bdf.apply(apply_encode, axis=1)       
    
    agg_params = {}
    for col in cols:
        agg_params[col] = 'sum'
    adf = sdf.groupby(['code'], observed=True).agg( agg_params ).reset_index()
    adf = adf.drop( 'code', axis=1)
    
    if lognorm:
        logging.debug(f'log normalizing for better heatmap values. ')
        adf = adf + 1
        adf = np.log(adf)
    return adf


def plot_heatmap_motifs(nbmdf):
    '''
    From Yara el Zoghby at HTNA 2025 CSHL
    
    '''
    binary_df = (nbmdf > 0).astype(int)
    # Each barcode's pattern across brain areas, as a string or tuple
    patterns = binary_df.apply(lambda row: tuple(row), axis=1)
    # Count frequency of each pattern
    pattern_counts = patterns.value_counts().sort_values(ascending=False)
    # Display top 10 most frequent projection motifs
    for pattern, count in pattern_counts.head(10).items():
        areas_in_pattern = [area for area, present in zip(binary_df.columns, pattern) if present]
        print(f'Pattern {areas_in_pattern}: {count} barcodes')
        import numpy as np
    top_patterns = pattern_counts.head(10).index.to_list()
    top_counts = pattern_counts.head(10).values
    # Build matrix: rows=motifs, cols=brain areas (0/1)
    motif_matrix = np.array(top_patterns)
    plt.figure(figsize=(12, 6))
    sns.heatmap(motif_matrix, cmap='Greys', cbar=False, xticklabels=binary_df.columns, yticklabels=[f'Motif {i+1} ({c})' for i, c in enumerate(top_counts)])
    plt.title('Top 10 Projection Motifs (Brain Area Presence)')
    plt.xlabel('Brain Area')
    plt.ylabel('Motif (with count)')
    plt.tight_layout()
    plt.savefig('top_10_projection_motifs_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()


#
# HIGH-LEVEL SEQUENCE FREQUENCY PLOTTING 
#

def make_freq_lineplot(df, logx=False, logy=True, col='count', title=None):
    '''
   
    
    '''
    plt.figure(figsize=(10,8))
    ser = df[col].sort_values(ascending=False)
    ser.reset_index(inplace=True, drop=True)
    yvals = ser.values
    if logy:
        yvals = np.log10(yvals)
    xvals = ser.index
    if logx:
        xvals = np.log10(xvals)
        
    g = sns.lineplot(data = df, x=df.index, y = yvals)
    g.figure.suptitle(f'{title}')     
    return g
 
 