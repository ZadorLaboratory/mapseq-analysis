#!/usr/bin/env python
#
#  creates analysis plots
#
import argparse
import logging
import os
import sys
import traceback

from configparser import ConfigParser

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

gitpath=os.path.expanduser("~/git/mapseq-processing")
sys.path.append(gitpath)

gitpath=os.path.expanduser("~/git/mapseq-analysis")
sys.path.append(gitpath)

from mapseq.core import *
from mapseq.utils import *
from msanalysis.analysis import *

    
if __name__ == '__main__':
    FORMAT='%(asctime)s (UTC) [ %(levelname)s ] %(filename)s:%(lineno)d %(name)s.%(funcName)s(): %(message)s'
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.WARN)
    
    parser = argparse.ArgumentParser()
      
    parser.add_argument('-d', '--debug', 
                        action="store_true", 
                        dest='debug', 
                        help='debug logging')

    parser.add_argument('-v', '--verbose', 
                        action="store_true", 
                        dest='verbose', 
                        help='verbose logging')
    
    parser.add_argument('-c','--config', 
                        metavar='config',
                        required=False,
                        default=os.path.expanduser('~/git/mapseq-processing/etc/mapseq.conf'),
                        type=str, 
                        help='config file.')    

    parser.add_argument('-e','--expid', 
                    metavar='expid',
                    required=False,
                    default=None,
                    type=str, 
                    help='experiment and/or brain label.')

    parser.add_argument('-s','--sampleinfo', 
                        metavar='sampleinfo',
                        required=False,
                        default=None,
                        type=str, 
                        help='XLS sampleinfo file. ')
   
    parser.add_argument('-o','--outfile', 
                    metavar='outfile',
                    required=False,
                    default=None, 
                    type=str, 
                    help='PDF plot out file. "simple_plots.pdf" if not given')  

    parser.add_argument('-L','--label_column', 
                    metavar='label_column',
                    required=False,
                    default=None, 
                    type=str, 
                    help='Labels for columns. Requires sampleinfo. [region|samplename|rtprimer]')
 
    parser.add_argument('infile' ,
                        metavar='infile', 
                        type=str,
                        default=None, 
                        help="One normalized barcode matrix files (e.g. '<brain>.nbcm.tsv')")
       
    args= parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)   

    cp = ConfigParser()
    cp.read(args.config)
    cdict = {section: dict(cp[section]) for section in cp.sections()}
    
    logging.debug(f'Running with config. {args.config}: {cdict}')
    logging.debug(f'infile={args.infile} outfile={args.outfile} expid={args.expid}')

    # set outdir / outfile
    outdir = os.path.abspath('./')
    outfile = f'{outdir}/binarized_plot.pdf'
    if args.outfile is not None:
        logging.debug(f'outfile specified.')
        outfile = os.path.abspath(args.outfile)
        filepath = os.path.abspath(outfile)    
        dirname = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        (base, ext) = os.path.splitext(filename)   
        head = base.split('.')[0]
        outdir = dirname
        logging.debug(f'outdir set to {outdir}')
    else:
        logging.debug(f'outfile not specified. using default {outfile}')        

    outdir = os.path.abspath(outdir)    
    os.makedirs(outdir, exist_ok=True)

    if args.sampleinfo is not None:
        logging.debug(f'loading sample DF...')
        sampdf = load_sample_info(cp, args.sampleinfo)
        logging.debug(f'\n{sampdf}')
        sampdf.to_csv(f'{outdir}/sampleinfo.tsv', sep='\t')
    else:
        sampdf = None
        
    make_binarized_plot(args.infile, 
                        outfile=args.outfile, 
                        expid=args.expid, 
                        label_column=args.label_column, 
                        sampdf=sampdf, 
                        cp=cp )
    
    