"""Resumable finite-box fixed-spacing measurement; no old scientific caches."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from full_hi_pipeline import scan_sim_fixed_cell as s
from full_hi_pipeline.particle_first_density import valid_objects, output_lock
from full_hi_pipeline.genus_grid import _atomic_json, _atomic_npz, sha256_file

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/scan_sim_fixed_cell_20260909_v1'
SOURCE=ROOT/'outputs/top_hi_1m_weighted_rsd_all13_20260908_v1'
SNAPS=[118,112,105,94]
SCALES=[15.,20.,34.]
POINTS=[(.3111,-1.)]+[(float(o),float(w)) for o in np.linspace(.2,.42,21) for w in np.linspace(-1.4,-.6,21)]


def run_point(snapshot, omega_m_scan, w_scan, *, spacing=4., padding=40, category='pilot', branches=None):
    branches=list(range(7)) if branches is None else list(branches)
    if not branches or len(set(branches))!=len(branches) or any(b not in range(7) for b in branches):
        raise ValueError('Invalid branch selection')
    target=len(SCALES)*len(branches)
    tag=f'snap{snapshot}_om{omega_m_scan:.7f}_w{w_scan:.7f}_d{spacing:g}_p{padding}'
    directory=OUT/category/tag;directory.mkdir(parents=True,exist_ok=True)
    source=SOURCE/f'snap{snapshot}_selection.npz'
    metadata=json.loads((SOURCE/f'snap{snapshot}.json').read_text())
    config=dict(schema='scan_sim_fixed_cell_v1', snapshot=snapshot,z=metadata['redshift'],
        omega_m_sim=s.OMEGA_M_SIM,w_sim=s.W_SIM,h_sim=s.H_SIM,
        omega_m_scan=omega_m_scan,w_scan=w_scan,h_scan=s.H_SIM,
        spacing=spacing,padding_cells=padding,scales=SCALES,
        m_scan=[float(s.M_SCAN[b-2]) for b in branches if b>=2],branch_indices=branches,
        k_parallel_cut_scan=.1,labels=[s.LABELS[b] for b in branches],source_sha256=sha256_file(source),
        producer_sha256=sha256_file(Path(__file__)),core_sha256=sha256_file(Path(s.__file__)),
        rank_source_sha256=sha256_file(ROOT/'full_hi_pipeline/fast_counting.py'),
        geometry='fixed-Z local AP; sim RSD first; full-box wrapping then high-edge crop',
        topology='negative Euler of induced cubical complex on grid centers; open boundaries; no correction',
        padding='zero density contrast; standard CIC deconvolution; crop output to retained region',
        selection='top 1000000 native HI weights then geometric crop',nu=np.linspace(-2.5,2.5,41).tolist())
    with output_lock(directory/'point'):
        cp=directory/'config.json'
        if cp.exists() and json.loads(cp.read_text())!=config:
            raise ValueError('Stale output configuration: '+str(directory))
        _atomic_json(cp,config)
        if (directory/'summary.json').exists():
            saved=json.loads((directory/'summary.json').read_text())
            for row in saved['measurements']:
                if sha256_file(Path(row['path']))!=row['sha256']:raise ValueError('Changed cached curve')
            return saved
        started=time.time()
        with np.load(source) as f:
            records=f['records'];ids=f['record_indices']
        if len(ids)!=1000000 or hashlib.sha256(ids.tobytes()).hexdigest()!=metadata['selected_indices_sha256']:
            raise ValueError('Selection identity mismatch')
        positions_sim,velocities_sim,weights,_,_=valid_objects(records)
        del records
        positions_rsd_sim=s.add_rsd_sim(positions_sim,velocities_sim,config['z'])
        factors=s.ap_scan(config['z'],omega_m_scan,w_scan)
        positions_rsd_scan,weights_scan,geometry=s.crop_geometry(positions_rsd_sim,weights,factors,spacing)
        delta,conservation=s.deposit(positions_rsd_scan,weights_scan,geometry['grid_shape_scan'],spacing)
        del positions_sim,velocities_sim,positions_rsd_sim,positions_rsd_scan,weights,weights_scan
        spectrum,axes,shape_fft,slices=s.prepare_spectrum(delta,spacing,padding);del delta
        rows=[]
        for radius in SCALES:
            for branch in branches:
                label=s.LABELS[branch]
                dest=directory/f'R{radius:g}_{label}.npz'
                if not dest.exists():
                    _atomic_json(directory/'status.json',dict(status='RUNNING',pid=os.getpid(),radius=radius,
                        branch=label,completed=len(rows),target=target,elapsed_seconds=time.time()-started))
                    field=s.filtered_field(spectrum,axes,shape_fft,slices,radius,branch)
                    measurement=s.measure(field,spacing,np.asarray(config['nu']));del field
                    _atomic_npz(dest,**measurement)
                with np.load(dest) as f:
                    row=dict(radius=radius,branch=label,gaussian_AR3=float(f['amplitude_gaussian'])*radius**3,
                        hermite_AR3=float(f['amplitude_hermite'])*radius**3,
                        field_std=float(f['field_std']),path=str(dest),sha256=sha256_file(dest))
                rows.append(row)
                print(tag,radius,label,row['gaussian_AR3'],flush=True)
        summary=dict(config=config,geometry=geometry,ap_factors_scan=factors.tolist(),
            conservation=conservation,fft_shape_scan=list(shape_fft),measurements=rows,
            elapsed_seconds=time.time()-started)
        _atomic_json(directory/'summary.json',summary)
        _atomic_json(directory/'status.json',dict(status='COMPLETE',completed=target,target=target,
            elapsed_seconds=time.time()-started))
        return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',type=int,default=118)
    parser.add_argument('--omega-m-scan',type=float,default=.3111)
    parser.add_argument('--w-scan',type=float,default=-1.)
    parser.add_argument('--spacing',type=float,default=4.)
    parser.add_argument('--padding',type=int,default=40)
    parser.add_argument('--category',default='pilot')
    args=parser.parse_args()
    run_point(args.snapshot,args.omega_m_scan,args.w_scan,spacing=args.spacing,padding=args.padding,category=args.category)
