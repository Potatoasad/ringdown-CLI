#!python
# coding: utf-8
#
# Copyright 2022
# Maximiliano Isi <max.isi@ligo.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301, USA.


import ringdown as rd
import os
import argparse
import configparser
from ast import literal_eval
import logging

##############################################################################
# PARSE INPUT
##############################################################################

DEFOUT = "ringdown_fit.nc"
_HELP = "Set up and run a ringdown analysis from a configuration file."

def get_parser():
    p = argparse.ArgumentParser(description=_HELP)
    p.add_argument('config', help="path to configuration file.")
    p.add_argument('-o', '--output', default=None,
                   help="output result path (default: `{}`).".format(DEFOUT))
    p.add_argument('--prior', action='store_true', help="sample from prior.")
    p.add_argument('--force', action='store_true',
                   help="overwrites output file if it already exists.")
    p.add_argument('-v', '--verbose', action='store_true')
    return p

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    print("Loading: {}".format(os.path.abspath(args.config)))
    
    config = configparser.ConfigParser()
    config.read(args.config)
    
    if config.has_section('run'):
        run_kws = {k: literal_eval(v) for k,v in config['run'].items()}
    else:
        run_kws = {}
    run_kws['prior'] = args.prior or run_kws.get('prior', False)
    
    if run_kws['prior']:
        DEFOUT = DEFOUT.replace('fit', 'prior')
    out = args.output or DEFOUT
    
    if os.path.exists(out):
        if args.force:
            logging.warning("overwriting output file: {}".format(out))
        else:
            raise FileExistsError("output file already exists: {}".format(out))
    
    ############################################################################
    # RUN FIT
    ############################################################################
    
    fit = rd.Fit.from_config(config)
    fit.run(**run_kws)
    
    if run_kws['prior']:
        result = fit.prior
    else:
        result = fit.result
    
    ext = os.path.splitext(out)[-1]
    if ext.lower() == '.nc':
        result.to_netcdf(out)
    else:
        result.to_json(out)
    
    print("Saved ringdown fit: {}".format(out))
