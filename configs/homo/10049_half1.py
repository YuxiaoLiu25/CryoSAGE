# CryoSAGE example. Replace /path/to dataset locations before training.
dataset = dict(
    apix=1.23,
    down_method='interp',
    down_size=192,
    f_mu=None,
    f_std=None,
    half='half1',
    lazy=True,
    mask_radius=1,
    random_subset_col='rlnRandomSubset',
    require_random_subset=True,
    root='/path/to/10049/data',
    size=192,
    star='/path/to/10049/data/Refine3D_10049/job001/run_data.star')
eval = dict(every_n_epochs=1)
exp_name = '10049_homo_half1'
loader = dict(batch_size=4, num_workers=4, persistent_workers=False)
loss = dict(mask_radius=1.0, real_mse_scale='parseval', type='real_mse')
model = dict(
    camera_z=5.0,
    ctf='v2',
    gaussian_parameterization='density',
    hetero=False,
    init_density=0.0009,
    init_scale=0.012,
    num_points=32000,
    renderer_gain=100.0,
    scale_max=0.2,
    scale_min=0.001,
    shift_method='fft')
optim = dict(
    eps=1e-15,
    min_lr_ratio=0.1,
    position_lr=5e-05,
    rotation_lr=5e-05,
    scale_lr=0.00025,
    scheduler='cosine',
    strength_lr=0.0005,
    weight_decay=0.0)
seed = 43
train = dict(log_every=100)
trainer = dict(devices=1, max_epochs=20, precision='16-mixed')
