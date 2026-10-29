"""Parallel, order-preserving extraction of valid full-HI catalog records."""
from __future__ import annotations
import numpy as np
from numba import njit, prange, get_num_threads
from .stream_density import JIUTIAN_DTYPE, BOX_SIZE_MPC_H


@njit(parallel=True, cache=True)
def _classify(records, keep, counts, nonfinite, nonpositive, outside):
    for block in prange(len(counts)):
        for i in range(len(records)*block//len(counts), len(records)*(block+1)//len(counts)):
            r=records[i]
            finite=(np.isfinite(r['Pos'][0]) and np.isfinite(r['Pos'][1]) and np.isfinite(r['Pos'][2])
                    and np.isfinite(r['Vel'][0]) and np.isfinite(r['Vel'][1]) and np.isfinite(r['Vel'][2])
                    and np.isfinite(r['HImass']))
            if not finite:
                nonfinite[block]+=1
            elif r['HImass']<=0:
                nonpositive[block]+=1
            else:
                keep[i]=True
                counts[block]+=1
                for axis in range(3):
                    position=np.float64(r['Pos'][axis])
                    if position < -1e-4 or position > BOX_SIZE_MPC_H+1e-4:
                        outside[block]+=1


@njit(parallel=True, cache=True)
def _gather(records, keep, starts, position, velocity, mass, indices, offset):
    for block in prange(len(starts)):
        dest=starts[block]
        for i in range(len(records)*block//len(starts), len(records)*(block+1)//len(starts)):
            if keep[i]:
                for axis in range(3):
                    position[dest,axis]=np.float64(records[i]['Pos'][axis])
                    velocity[dest,axis]=np.float64(records[i]['Vel'][axis])
                mass[dest]=np.float64(records[i]['HImass'])
                indices[dest]=offset+i
                dest+=1


def extract_valid_objects(records, offset=0):
    """Return the same values/order/dtypes as NumPy masking, without top-N cuts.

    Prefix offsets give each block a disjoint output range. No atomics, sorting,
    fast-math, dtype reduction, or changes to RSD/AP/CIC are used.
    """
    records=np.asarray(records)
    if records.ndim!=1 or records.dtype!=JIUTIAN_DTYPE:
        raise ValueError('Expected the audited 40-byte record dtype')
    blocks=max(1,min(get_num_threads()*4,max(1,len(records)//4096)))
    keep=np.zeros(len(records),dtype=np.bool_)
    counts=np.zeros(blocks,dtype=np.int64)
    bad_finite=np.zeros(blocks,dtype=np.int64)
    bad_positive=np.zeros(blocks,dtype=np.int64)
    bad_position=np.zeros(blocks,dtype=np.int64)
    _classify(records,keep,counts,bad_finite,bad_positive,bad_position)
    if bad_position.sum():
        raise ValueError('Real positions outside the declared original box; check units/schema')
    starts=np.zeros(blocks,dtype=np.int64)
    starts[1:]=np.cumsum(counts[:-1])
    size=int(counts.sum())
    position=np.empty((size,3),dtype=np.float64)
    velocity=np.empty((size,3),dtype=np.float64)
    mass=np.empty(size,dtype=np.float64)
    indices=np.empty(size,dtype=np.int64)
    _gather(records,keep,starts,position,velocity,mass,indices,int(offset))
    return position,velocity,mass,indices,{'nonfinite':int(bad_finite.sum()),'nonpositive':int(bad_positive.sum())}
