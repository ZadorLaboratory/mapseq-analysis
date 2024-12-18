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
                    default='MAPseqDefault',
                    type=str, 
                    help='explicitly provided experiment id')

    parser.add_argument('-s','--sampleinfo', 
                        metavar='sampleinfo',
                        required=True,
                        default=None,
                        type=str, 
                        help='XLS sampleinfo file or sampleinfo.tsv. ') 
    
    parser.add_argument('-o','--outfile', 
                    metavar='outfile',
                    required=False,
                    default=None, 
                    type=str, 
                    help='PDF plot out file. "simple_plots.pdf" if not given')  

    parser.add_argument('-O','--outdir', 
                    metavar='outdir',
                    required=False,
                    default=None, 
                    type=str, 
                    help='outdir. input file base dir if not given.')     

    parser.add_argument('infile',
                        metavar='infile',
                        type=str,
                        help='Normalized barcode matrix (<brain>.nbcm.tsv) from process_merged.py')
       
    args= parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)   

    cp = ConfigParser()
    cp.read(args.config)
    cdict = {section: dict(cp[section]) for section in cp.sections()}
    
    logging.debug(f'Running with config. {args.config}: {cdict}')
    logging.debug(f'infile={args.infile}')
      
   
    outdir = None
    if args.outdir is not None:
        outdir = os.path.abspath(args.outdir)
    else:
        filepath = os.path.abspath(args.infile)    
        dirname = os.path.dirname(filepath)
        outdir = dirname
    
    if args.outfile is None:
        outfile = f'{outdir}/simple_plots.pdf'
    else:
        outfile = args.outfile
        
    logging.debug(f'outdir={outdir} outfile={outfile} ')
    os.makedirs(outdir, exist_ok=True)    
    sampdf = load_sample_info(cp, args.sampleinfo)
    logging.debug(f'datatypes = {sampdf.dtypes} infile={args.infile} outfile={outfile} expid={args.expid}')
    
    make_plots(cp, sampdf, args.infile, outfile=outfile, outdir=outdir, exp_id=args.expid )
    
