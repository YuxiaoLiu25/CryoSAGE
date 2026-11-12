# CryoSAGE example. Replace /path/to dataset locations before training.
dataset = dict(
    apix=1.067,
    down_method='interp',
    down_size=192,
    f_mu=None,
    f_std=None,
    half='all',
    lazy=True,
    mask_radius=1.0,
    root='/path/to/10176',
    scale_images=1.0,
    size=384,
    star='/path/to/10176/preprocessed_local_j13_pose_384.star')
eval = dict(
    class_label_file=None,
    every_n_epochs=10,
    export_class_volumes=False,
    export_fine_class_volumes=False,
    export_gaussian_states=False,
    export_pca_volumes=True,
    export_umap=True,
    export_umap_volumes=True,
    fine_class_label_file=None,
    latent_hist_bins=90,
    latent_hist_smooth_sigma=1.5,
    latent_plot_title='CryoSAGE EMPIAR-10176',
    latent_scatter_alpha=0.08,
    latent_scatter_bins=70,
    latent_scatter_max_points=0,
    latent_scatter_size=0.45,
    pca_dims=2,
    pca_steps=4,
    umap_fit_max_points=0,
    umap_metric='euclidean',
    umap_min_dist=0.08,
    umap_n_neighbors=50,
    umap_num_representatives=10,
    umap_random_state=42)
exp_name = '10176_hetero'
loader = dict(batch_size=4, num_workers=4)
loss = dict(
    mask_radius=1.0,
    occupancy_off_penalty=0.0001,
    real_mse_scale='parseval',
    type='real_mse')
model = dict(
    camera_z=5.0,
    ctf='v2',
    deform_attributes='xyz_occupancy',
    deform_decoder='factorized',
    deform_hidden=256,
    deform_layers=4,
    deform_pe_freqs=6,
    deform_pe_include_xyz=True,
    deform_scale=0.05,
    deform_tanh=False,
    gaussian_parameterization='density',
    hetero=True,
    init_density=0.0009,
    init_scale=0.012,
    num_points=32768,
    occupancy_factor_dim=16,
    occupancy_factor_hidden=128,
    occupancy_factor_layers=2,
    occupancy_factor_tanh=True,
    occupancy_init=0.95,
    occupancy_temperature=1.0,
    occupancy_temperature_schedule=[
        (
            1,
            1.0,
        ),
        (
            10,
            1.0,
        ),
        (
            20,
            0.7,
        ),
        (
            30,
            0.5,
        ),
    ],
    renderer_gain=1000.0,
    scale_max=0.25,
    scale_min=0.001,
    shift_method='fft',
    z_dim=8)
optim = dict(
    deform_lr=0.0001,
    encoder_lr=0.0001,
    eps=1e-15,
    min_lr_ratio=0.1,
    position_lr=0.0001,
    rotation_lr=0.0001,
    scale_lr=0.0005,
    scheduler='cosine',
    strength_lr=0.001,
    weight_decay=0.0)
seed = 42
train = dict(log_every=500)
trainer = dict(devices=4, max_epochs=30, precision='16-mixed')
vae = dict(
    beta=None,
    beta_warmup_epochs=5,
    encoder_hidden=256,
    encoder_layers=3,
    encoder_mask_radius=1.0,
    encoder_type='resid',
    export_batch_size=32,
    logvar_max=10.0,
    logvar_min=-10.0,
    phase_flip=True)
