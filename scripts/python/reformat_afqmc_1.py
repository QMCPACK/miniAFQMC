#!/usr/bin/env python

import h5py
import numpy 
import time
import os
import sys
from mpi4py import MPI

# old format
# group      /
# group      /Propagators
# group      /Propagators/phaseless_ImpSamp_ForceBias
# dataset    /Propagators/phaseless_ImpSamp_ForceBias/Spvn_block_sizes
# dataset    /Propagators/phaseless_ImpSamp_ForceBias/Spvn_dims
# dataset    /Propagators/phaseless_ImpSamp_ForceBias/Spvn_index_0
# dataset    /Propagators/phaseless_ImpSamp_ForceBias/Spvn_propg1
# dataset    /Propagators/phaseless_ImpSamp_ForceBias/Spvn_vals_0
# dataset    /Propagators/phaseless_ImpSamp_ForceBias/Spvn_vn0
# group      /Wavefunctions
# group      /Wavefunctions/PureSingleDeterminant
# dataset    /Wavefunctions/PureSingleDeterminant/SpHijkl_cols
# dataset    /Wavefunctions/PureSingleDeterminant/SpHijkl_rowIndex
# dataset    /Wavefunctions/PureSingleDeterminant/SpHijkl_vals
# dataset    /Wavefunctions/PureSingleDeterminant/Wavefun
# dataset    /Wavefunctions/PureSingleDeterminant/dims
# dataset    /Wavefunctions/PureSingleDeterminant/hij
# dataset    /Wavefunctions/PureSingleDeterminant/hij_indx
# dataset    /Wavefunctions/PureSingleDeterminant/occups

def reformat_afqmc_1(filein, fileout, MaxInt=2000000):

    f0 = h5py.File(filein, "r") 
    f1 = h5py.File(fileout, "w-") 

    # wavefunction
    h5grp = f1.create_group("Wavefunctions/PureSingleDeterminant")

    # 0: ignore
    # 1: global # terms in Vakbl 
    # 2: Vakbl #rows
    # 3: Vakbl #cols
    # 4: NMO
    # 5: NAEA
    # 6: NAEB
    # 7: should be 0
    #
    dims = f0["/Wavefunctions/PureSingleDeterminant/dims"][:]
    vals = f0["/Wavefunctions/PureSingleDeterminant/SpHijkl_vals"][:]
    cols = f0["/Wavefunctions/PureSingleDeterminant/SpHijkl_cols"][:]
    index = f0["/Wavefunctions/PureSingleDeterminant/SpHijkl_rowIndex"][:]
    assert(index.shape[0] == dims[2]+1)

    # later on add counters for number of terms with various cutoffs
    # to allow to pre-allocate sparse matrices without reading 
    blk=0
    cnt=0
    i0=0
    blksz = numpy.zeros((dims[2]),dtype=numpy.int32)
    rows = numpy.zeros((dims[2]+1),dtype=numpy.int32)
    rows[0]=0
    for ri in range(dims[2]):
        if ((index[ri+1]-index[i0]) > MaxInt) or (ri == dims[2]-1):
            dummy = h5grp.create_dataset("SpHijkl_vals_"+str(blk), data=vals[index[i0]:index[ri+1]])
            dummy = h5grp.create_dataset("SpHijkl_cols_"+str(blk), data=cols[index[i0]:index[ri+1]])
            index_ = index[i0:ri+2]-index[i0]
            dummy = h5grp.create_dataset("SpHijkl_rowIndex_"+str(blk), data=index_)
            index_ = None
            rows[blk+1] = ri+1
            blksz[blk] = index[ri+1]-index[i0]
            i0 = ri+1
            blk += 1

    dummy = h5grp.create_dataset("SpHijkl_block_sizes", data=blksz[0:blk])
    dummy = h5grp.create_dataset("SpHijkl_block_rows", data=rows[0:blk+1])

    # propagator
    h5grp = f1.create_group("Propagators/phaseless_ImpSamp_ForceBias")

    f1.close()

    for name in ('Spvn_block_sizes','Spvn_dims','Spvn_index_0','Spvn_vals_0'):
        cmd = 'h5copy -v -i ' + filein + ' -o ' + fileout+ ' -s "/Propagators/phaseless_ImpSamp_ForceBias/' + str(name) + '" -d "/Propagators/phaseless_ImpSamp_ForceBias/' + str(name) + '" >& /dev/null'
        os.system(cmd)    
    for name in ('SpHijkl_cols','SpHijkl_vals','dims'):
        cmd = 'h5copy -v -i ' + filein + ' -o ' + fileout+ ' -s "/Wavefunctions/PureSingleDeterminant/' + str(name) + '" -d "/Wavefunctions/PureSingleDeterminant/' + str(name) + '" >& /dev/null'
        os.system(cmd)    

    # copy datasets that do not change
    for name in ('Spvn_propg1','Spvn_vn0'):
        cmd = 'h5copy -v -i ' + filein + ' -o ' + fileout+ ' -s "/Propagators/phaseless_ImpSamp_ForceBias/' + str(name) + '" -d "/Propagators/phaseless_ImpSamp_ForceBias/' + str(name) + '" >& /dev/null'
        os.system(cmd)    
    for name in ('Wavefun','hij','hij_indx','occups'):
        cmd = 'h5copy -v -i ' + filein + ' -o ' + fileout+ ' -s "/Wavefunctions/PureSingleDeterminant/' + str(name) + '" -d "/Wavefunctions/PureSingleDeterminant/' + str(name) + '" >& /dev/null'
        os.system(cmd)    
    cmd = 'h5copy -v -i ' + filein + ' -o ' + fileout+ ' -s "/Wavefunctions/PureSingleDeterminant/SpHijkl_rowIndex" -d "/Wavefunctions/PureSingleDeterminant/SpHijkl_row_counts"' + ' >& /dev/null'
    os.system(cmd)    
    cmd = 'h5copy -v -i ' + filein + ' -o ' + fileout+ ' -s "/Wavefunctions/PureSingleDeterminant/SpHijkl_rowIndex" -d "/Wavefunctions/PureSingleDeterminant/SpHijkl_rowIndex"' + ' >& /dev/null'
    os.system(cmd)    
 
reformat_afqmc_1('afqmc.old.h5','afqmc.h5')
