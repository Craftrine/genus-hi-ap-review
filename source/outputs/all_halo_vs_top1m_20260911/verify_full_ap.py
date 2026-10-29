"""Independent full-catalog direct-AP replay of one non-fiducial point."""
from stream import *
from full_hi_pipeline.genus_grid import _atomic_json

snap=94;om=.3121;w=-1.
meta=json.loads((SELECT/f'snap{snap}.json').read_text());source=Path(meta['source_path']);z=meta['redshift']
alpha=old.ap_scan(z,om,w);_,_,geo=mesh.geometry(np.zeros((1,3)),np.ones(1),alpha)
sh=tuple(geo['grid_shape_scan']);sp=np.array(geo['spacing_scan']);mass=np.zeros(sh,np.float64)
dig=hashlib.sha256();n=0;total=0.;t=time.time()
with source.open('rb') as h:
    while True:
        raw=h.read(40*1000000)
        if not raw:break
        dig.update(raw);r=np.frombuffer(raw,dtype=JIUTIAN_DTYPE)
        p,v,wt,_,_=valid_objects(r,backend='numba');p=old.add_rsd_sim(p,v,z)
        # Explicit AP on every full-catalog object, before CIC. No p*N/L reuse.
        p=np.ascontiguousarray(np.mod(p*alpha,1000*alpha))
        deposit_scalar(p,wt,mass,sp);total+=wt.sum();n+=len(wt)
assert dig.hexdigest()==meta['source_sha256'];assert n==meta['n_valid_all'];assert abs(mass.sum()-total)/total<1e-11
delta=(mass/(total/mass.size)-1).astype(np.float32);del mass
cached=np.load(OUT/f'snap{snap}'/('delta_'+'x'.join(map(str,sh))+'.npy'))
max_delta=float(np.max(abs(delta-cached)));fraction_delta=float(np.mean(delta!=cached));del cached
spec,axes,shape,sl=mesh.prepare_spectrum(delta,sp,160);del delta
rows=[]
for b in BR:
    field=old.filtered_field(spec,axes,shape,sl,20,b);m=mesh.measure(field,sp,NU);del field
    path=OUT/f'snap{snap}'/f'om{om:.7f}_w{w:.7f}'/f'R20_{old.LABELS[b]}.npz'
    with np.load(path) as a:
        np.testing.assert_array_equal(m['negative_euler_counts'],a['negative_euler_counts'])
        np.testing.assert_allclose(m['thresholds'],a['thresholds'],rtol=0,atol=2e-7)
        rows.append(dict(branch=old.LABELS[b],negative_euler_counts_exact=True,max_threshold_difference=float(max(abs(m['thresholds']-a['thresholds'])))))
result=dict(status='PASS',snapshot=snap,omega_m=om,w=w,all_valid_objects=n,source_sha256=dig.hexdigest(),density_max_difference=max_delta,density_fraction_different=fraction_delta,all_four_branches=rows,seconds=time.time()-t,description='Independent direct AP transform and physical-coordinate CIC for every full-catalog object; compared to normalized-coordinate reuse')
_atomic_json(OUT/'full_catalog_direct_AP_verification.json',result);print(json.dumps(result,indent=2))
