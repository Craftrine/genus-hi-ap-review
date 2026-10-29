"""Recalculate saved spectral amplitude ratios; no field measurements. Author: Wells."""
import ast, csv, hashlib, json, math, struct, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT/'outputs/scan_sim_fixed_cell_20260909_v1/spectral_topology_comparison'
source = BASE/'comparison.csv'
rows = {r['branch']:r for r in csv.DictReader(source.open()) if int(r['snapshot']) == 118 and float(r['R']) == 20 and r['branch'] in ('nofg','horizontal_only')}
current = {r['branch']:r for r in csv.DictReader((ROOT/'paper/english_revision_v0.20_full_redshift_table_20261029/data/sample_amplitudes.csv').open()) if r['sample']=='top1m' and int(r['snapshot'])==118}
def matrix_from_npz(name):
    with zipfile.ZipFile(BASE/'snap118_arrays.npz') as z:
        b=z.read(name+'.npy')
    assert b[:6]==b'\x93NUMPY'
    if b[6]==1:
        n=struct.unpack('<H',b[8:10])[0]; off=10
    else:
        n=struct.unpack('<I',b[8:12])[0]; off=12
    h=ast.literal_eval(b[off:off+n].decode())
    assert h['shape']==(3,3) and h['descr']=='<f8' and not h['fortran_order']
    v=struct.unpack('<9d',b[off+n:]); return [v[i:i+3] for i in (0,3,6)]
def det(m):
    a,b,c=m[0]; d,e,f=m[1]; g,h,i=m[2]
    return a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)
matrices={}
for branch,r in rows.items():
    assert float(r['topology_gaussian_AR3'])==float(current[branch]['AR3'])
    m=matrix_from_npz('R20_'+branch+'_M_padded'); matrices[branch]=m
    a=math.sqrt(det(m))*20**3/(4*math.pi**2)
    assert math.isclose(a,float(r['spectral_padded_AR3']),rel_tol=1e-12)
keys=['topology_gaussian_AR3','spectral_padded_AR3','gradient_retained_AR3','isotropic_trace_AR3']
result=[]
for k in keys:
    a,b=(float(rows[x][k]) for x in ['nofg','horizontal_only'])
    result.append(dict(method=k,noFG_AR3=a,horizontal_AR3=b,ratio=b/a))
actual=result[0]['ratio']
for r in result:r['relative_difference_from_topology_percent']=100*(r['ratio']/actual-1)
with (OUT/'ratios.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=result[0].keys());w.writeheader();w.writerows(result)
(OUT/'verification.json').write_text(json.dumps(dict(snapshot=118,z=.10235,R=20,kparallel_cut=.1,current_manuscript_amplitudes_exact_match=True,saved_matrix_determinants_verified=True,matrices=matrices,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),ratios=result),indent=2)+'\n')
print(json.dumps(result,indent=2))
