"""加速版 runner：12 进程 × 单线程 FFT + 分支共享平滑谱。

优化点（不改变任何数值定义）：
1. 每红移只对谱做一次 R20 高斯平滑，7 个分支共享；
2. FFT workers=2 + NUMBA 2 线程 × 9 进程 = 18 核满载（用户已关闭其他应用腾内存）；
3. 进程池 12 = 48GB 内存下能撑住的上限（单 worker 峰值 phys_footprint 约 3.2GB）。
校验门 G1 不变：A 合同两个已测点必须与生产存档逐元素一致后才全量跑。
"""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '1'
os.environ['NUMBA_NUM_THREADS'] = '2'
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache_wb'
import sys, json, time
from pathlib import Path
import numpy as np
import multiprocessing as mp

PROJ = '/Users/LOCAL_USER/Documents/ChatGPT/genus'
sys.path.insert(0, PROJ)
sys.path.insert(0, '/Users/LOCAL_USER/Documents/ChatGPT/genus_workbuddy/mfs_crosscheck')
from scipy.fft import irfftn
from full_hi_pipeline import scan_sim_fixed_cell as old, scan_sim_nearest_mesh as n
from full_hi_pipeline.particle_first_density import valid_objects
from taper_pipeline import taper_weight

OUT = Path('/Users/LOCAL_USER/Documents/ChatGPT/genus_workbuddy/dense_band_20260917')
SNAPS = (118, 112, 105, 94)
ZS = {118: .10235, 112: .19095, 105: .31970, 94: .59339}
NU = np.linspace(-2.5, 2.5, 41)
NBR = 7
TAPER_CELLS = 4.0
FFTW = 2
CATS = {}

def load_cats():
    for snap in SNAPS:
        src = Path(f'{PROJ}/outputs/top_hi_1m_weighted_rsd_all13_20260908_v1/snap{snap}_selection.npz')
        with np.load(src) as f:
            rec = f['records']
        p, v, wt, _, _ = valid_objects(rec); del rec
        CATS[snap] = (old.add_rsd_sim(p, v, ZS[snap]), wt)
        del p, v

def smooth_spectrum(spec, axes, radius=20.):
    """R20 高斯平滑（7 分支共享一次）。"""
    work = spec.copy()
    for axis, k in enumerate(axes):
        view = [1, 1, 1]; view[axis] = len(k)
        work *= np.exp(-.5*(radius*k)**2).astype(np.float32).reshape(view)
    return work

def branch_field(smoothed, axes, shape_fft, slices, branch, contract):
    work = smoothed.copy()
    if contract == 'A':
        mask = old.removal_mask(axes, branch)
        if mask is not None:
            work *= ~mask
    else:
        wgt = taper_weight(axes, branch, TAPER_CELLS)
        if wgt is not None:
            work *= wgt
    work[0, 0, 0] = 0
    padded = irfftn(work, s=shape_fft, workers=FFTW)
    return np.ascontiguousarray(padded[slices])

def measure_point(om, w, contract):
    curves = np.zeros((NBR, 4, 41)); amps = np.zeros((NBR, 4, 2))
    metas = []
    for si, snap in enumerate(SNAPS):
        p0, wt = CATS[snap]
        alpha = old.ap_scan(ZS[snap], om, w)
        if contract == 'A':
            pp, w2, g = n.geometry(p0.copy(), wt, alpha)
            shape, spacing = g['grid_shape_scan'], g['spacing_scan']
        else:
            lengths = 1000*np.asarray(alpha)
            shape = np.array([250, 250, 250])
            spacing = lengths/shape
            pp = np.ascontiguousarray(np.mod(p0*alpha, lengths)); w2 = wt
        delta, cons = n.deposit(pp, w2, shape, spacing); del pp
        spec, axes, shape_fft, slices = n.prepare_spectrum(delta, spacing, 160); del delta
        smoothed = smooth_spectrum(spec, axes)
        del spec
        for br in range(NBR):
            field = branch_field(smoothed, axes, shape_fft, slices, br, contract)
            m = n.measure(field, spacing, NU)
            curves[br, si] = m['curve']
            amps[br, si] = (m['amplitude_gaussian'], m['amplitude_hermite'])
            del field
        metas.append(dict(snap=int(snap), shape=np.asarray(shape).tolist(),
                          spacing=np.asarray(spacing, float).tolist(),
                          conservation={k: float(v) for k, v in cons.items()}))
        del smoothed
    return dict(omega_m=om, w=w, contract=contract, curves=curves, amplitudes=amps,
                meta=json.dumps(metas))

def ptag(om, w):
    return f'om{om:.7f}_w{w:.7f}'

def worker(task):
    om, w, contract = task
    fp = OUT/f'points_{contract}'/f'{ptag(om, w)}.npz'
    if fp.exists():
        return None
    r = measure_point(om, w, contract)
    np.savez_compressed(fp, **r)
    return fp.name

def gate_replay_A():
    for om, w in [(0.3091, -1.0), (0.3131, -1.0)]:
        r = measure_point(om, w, 'A')
        for si, snap in enumerate(SNAPS):
            f0 = f'{PROJ}/outputs/nearest_mesh_run_20260910/points/snap{snap}_om{om:.7f}_w{w:.7f}/R20_nofg.npz'
            with np.load(f0) as f:
                np.testing.assert_array_equal(r['curves'][0, si], f['curve'])
        print(f'G1 通过（加速版）：A ({om},{w}) 与存档逐元素一致', flush=True)

LOCK = OUT/'.run.lock'

def acquire_lock():
    """单实例锁：防止多个会话/窗口重复拉起同一个任务，导致重复算同一批点。"""
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text().strip())
        except Exception:
            pid = None
        if pid:
            try:
                os.kill(pid, 0)
                print(f'已有实例在跑（PID {pid}），本次退出', flush=True)
                sys.exit(0)
            except ProcessLookupError:
                pass
            except PermissionError:
                print(f'已有实例在跑（PID {pid}，无权限确认存活），本次退出', flush=True)
                sys.exit(0)
    LOCK.write_text(str(os.getpid()))

if __name__ == '__main__':
    # 单实例锁：防止 launchd/手动重复启动造成双池竞争
    import fcntl
    _lock_fd = os.open(OUT/'run_dense_fast.lock', os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print('已有实例在跑，本次退出', flush=True)
        sys.exit(0)
    t0 = time.time()
    acquire_lock()
    print(f'PID {os.getpid()}，加载选样目录…', flush=True)
    load_cats()
    print('校验门 G1（加速版）…', flush=True)
    gate_replay_A()

    plan = json.loads((OUT/'plan.json').read_text())
    tasks = [(om, w, 'A') for om, w in plan['points_A_new']] + \
            [(om, w, 'B') for om, w in plan['points_B']]
    todo = [t for t in tasks if not (OUT/f'points_{t[2]}'/f'{ptag(t[0], t[1])}.npz').exists()]
    print(f'待测 {len(todo)} 点', flush=True)

    done = 0
    with mp.Pool(9, initializer=load_cats) as pool:   # 6 workers ≈ 27GB；>8 会越过 48GB 内存线触发交换，反而更慢
        for _ in pool.imap_unordered(worker, todo, chunksize=2):
            done += 1
            if done % 50 == 0:
                el = time.time()-t0
                print(f'{done}/{len(todo)} 用时 {el/60:.1f} min，ETA {(el/done*(len(todo)-done))/60:.1f} min', flush=True)
    print(f'全部完成，总用时 {(time.time()-t0)/60:.1f} min', flush=True)
    json.dump(dict(n_tasks=len(tasks), elapsed_min=(time.time()-t0)/60,
                   snaps=SNAPS, branches=NBR, taper_cells=TAPER_CELLS,
                   accel='shared-smooth + FFT workers=1 + pool12 x numba1'),
              (OUT/'status.json').open('w'), indent=1)
