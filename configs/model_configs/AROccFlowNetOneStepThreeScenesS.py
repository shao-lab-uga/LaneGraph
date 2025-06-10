from random import shuffle
import sched
from tabnanny import check

from lark import logger
from configs.utils.config import load_config
import os

from evaluation import losses
from evaluation.losses import occupancy_flow_map_loss, trajectory_loss
from models.AROccFlowNet import conv_gru, unet_decoder
# ============= Seed ===================
random_seed = 42
# ============= Path ===================
project_name = 'AROccFlowNetOneStepThreeScenesS'
# checkpoints = "./checkpoints/"
# logs = "./logs/"
exp_dir = './exp/'  # PATH TO YOUR EXPERIMENT FOLDER
project_dir = os.path.join(exp_dir, project_name)
# ============= Dataset Parameters=================
dataset_config = load_config("configs/dataset_configs/I24Motion_config.py")
occupancy_flow_map_config = dataset_config.occupancy_flow_map
occupancy_flow_map_height = occupancy_flow_map_config.occupancy_flow_map_height
occupancy_flow_map_width = occupancy_flow_map_config.occupancy_flow_map_width

task_config = dataset_config.task
num_his_points = task_config.num_his_points
num_waypoints = task_config.num_waypoints

paths_config = dataset_config.paths
generated_data_path = paths_config.generated_data_path
total_data_samples = 30000
# ============= Model Parameters =================
input_dim = 3 # occupancy, flow_x, flow_y
hidden_dim = 128
num_states = 9# TODO: Define the number of states
num_heads = 4
dropout_prob=0.1
num_motion_mode=6 # number of future motion modes
embed_dims = (96, 192, 384, 768)
depths = (3, 3, 9, 3)
shallow_decode = 1
# ============= Train Parameters =================
num_machines = 1
gpu_ids = [0,1]
max_epochs = 30
batch_size = 8
# ============= Optimizer Parameters =================
optimizer_type = 'AdamW'
optimizers_dic = dict(
    AdamW=dict(
        type='AdamW',       # AdamW optimizer
        learning_rate=3e-4,            # Base learning rate
        betas=(0.9, 0.95),  # Slightly higher β2 for smoother updates
        eps=1e-8,           # Avoids division by zero
        weight_decay=1e-6   # Encourages generalization
    ),
    NAdam=dict(
        type='NAdam',       # NAdam optimizer
        learning_rate = 1e-4,
        weight_decay = 1e-4
    )
)
assert optimizer_type in optimizers_dic, f"Optimizer type {optimizer_type} is not supported"
# ============= Scheduler Parameters =================
scheduler_type = 'CosineAnnealingWarmRestarts'
schedulers_dic = dict(
    StepLR=dict(
        type='StepLR',      # StepLR scheduler
        step_size = 3,
        gamma = 0.5
    ),
    CosineAnnealingWarmRestarts=dict(
        type='CosineAnnealingWarmRestarts',  # CosineAnnealingWarmRestarts scheduler
        T_0=2,    # First restart at 2 epochs
        T_mult=2,  # Restart period doubles (2 → 4 → 8 epochs)
        eta_min=1e-6  # Minimum LR to avoid vanishing updates
    )
)
assert scheduler_type in schedulers_dic, f"Scheduler type {scheduler_type} is not supported"


# ============= Test Parameters =================
guidance_scale = 7.5
weight_path = None  # None is the last ckpt you have trained
# ============= Config ===================
config = dict(
    project_name=project_name,
    project_dir=project_dir,
    dataset_config=dataset_config,
    dataloaders=dict(
        datasets=dict(
            train_ratio = 0.8,
            validation_ratio = 0.1,
            test_ratio = 0.1,
            data_path=generated_data_path,
            total_data_samples=total_data_samples,
        ),
        train=dict(
            batch_size=batch_size,
            num_workers=1,
            shuffle=True,
        ),
        val=dict(
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
        ),
        test=dict(
            batch_size=1,
            num_workers=1,
            shuffle=False,
        ),
    ),
    models=dict(
        aroccflownet=dict(
            map_height = occupancy_flow_map_height,
            map_width = occupancy_flow_map_width,
            num_waypoints=num_waypoints,
            hidden_dim=hidden_dim,
            nhead=num_heads,
            dropout_prob=dropout_prob,
            num_layers=2,
            
            convnext=dict(
                hidden_dim=hidden_dim,
                out_channels=hidden_dim,
                shallow_decode=shallow_decode, # deepest layer index should be less than len(embed_dims)-1
                embed_dims = embed_dims,
                depths = depths,
                flow_temporal_depth=num_his_points - 1,
                patch_embedding_occupancy_map=dict(
                    img_size=(occupancy_flow_map_height, occupancy_flow_map_width),
                    patch_size=(4, 4),
                    in_chans=num_his_points,
                    embed_dim=embed_dims[0],
                ),
                patch_embedding_flow_map=dict(
                    img_size=(occupancy_flow_map_height, occupancy_flow_map_width),
                    patch_size=(4, 4),
                    in_chans=2,
                    embed_dim=embed_dims[0],
                ),
            ),
            
            motionpredictor=dict(
                num_states=num_states,
                hidden_dim=embed_dims[-shallow_decode-1],
                num_heads=num_heads,
                dropout_prob=dropout_prob,
                num_layers=1,
                num_motion_mode=num_motion_mode,
                num_waypoints=num_waypoints,
            ),
            unet_decoder=dict(
                embed_dims=embed_dims[:-shallow_decode],
                num_waypoints=num_waypoints,
            ),
        ),
        
        # pretrained='runwayml/stable-diffusion-v1-5',
        # pipeline_name='StableDiffusionControlPipeline',
        # checkpoint=ckpt_2d,
        # with_ema=with_ema,
        # weight_path=weight_path,
    ),
    losses=dict(
        occupancy_flow_map_loss=dict(
            ogm_weight  = 1000,
            occ_weight  = 0,
            flow_weight = 1,
            flow_origin_weight = 1000,
            replica=1.0,
            no_use_warp=False,
            use_pred=False,
            use_focal_loss=True,
            use_gt=False
        ),
        trajectory_loss=dict(
            regression_weight=200,
            classification_weight=1.0
        ),
    ),
    optimizer = optimizers_dic[optimizer_type],
    scheduler = schedulers_dic[scheduler_type],
    train=dict(
        max_epochs=max_epochs,
        checkpoint_interval=1,
        checkpoint_dir=os.path.join(project_dir, 'checkpoints'),
        checkpoint_total_limit=10,
        log_interval=10,
    ),
    test=dict(
        occupancy_flow_map_height=occupancy_flow_map_height,
        occupancy_flow_map_width=occupancy_flow_map_width,
    ),
    loggers=dict(
        tensorboard=dict(
            type='Tensorboard',
            log_dir=os.path.join(project_dir, 'logs'),
        ),
    ),
)