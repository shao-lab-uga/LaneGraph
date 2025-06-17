import os
from utils.config_utils import load_config
# ============= Seed ===================
random_seed = 42
# ============= Path ===================
project_name = 'LaneAndDirectionExtraction'
exp_dir = './exp/'  # PATH TO YOUR EXPERIMENT FOLDER
project_dir = os.path.join(exp_dir, project_name)
# ============= Dataset Parameters=================
dataset_config = load_config("configs/dataset_lane_and_direction_extraction.py")


paths_config = dataset_config.paths
data_attributes_config = dataset_config.data_attributes
processed_data_path = paths_config.processed_data_path
dataset_image_size = data_attributes_config.dataset_image_size
input_image_size = data_attributes_config.input_image_size

training_range = data_attributes_config.training_range
testing_range = data_attributes_config.testing_range
validation_range = data_attributes_config.validation_range
# ============= Model Parameters =================

# ============= Train Parameters =================
num_machines = 1
gpu_ids = [0,1]
batch_size = 8
preload_tiles=4
max_epochs = len(training_range) * dataset_image_size * dataset_image_size / (batch_size * input_image_size * input_image_size)

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
        train=dict(
            data_path=processed_data_path,
            image_size=input_image_size,
            dataset_image_size=dataset_image_size,
            preload_tiles=preload_tiles,
            batch_size=batch_size,
            indrange=training_range,
            training=True,  # Indicates this is for training
        ),
        test=dict(
            data_path=processed_data_path,
            image_size=input_image_size,
            dataset_image_size=dataset_image_size,
            preload_tiles=preload_tiles,
            batch_size=1,
            indrange=testing_range,
            training=False,  # Indicates this is for testing
        ),
        validate=dict(
            data_path=processed_data_path,
            image_size=input_image_size,
            dataset_image_size=dataset_image_size,
            preload_tiles=preload_tiles,
            batch_size=batch_size,
            indrange=validation_range,
            training=False,  # Indicates this is for validation
        ),
    ),
    models=dict(
        # avaiable settings = [
            # ("resnet34", "fpn"),
            # ("resnet50", "fpn"),
            # ('vit_base_patch16_224.dino', 'vitdecoder'),
        # ]
        lane_and_direction_extraction_model=dict(
            backbone_name = 'resnet34',  # Backbone model name
            decoder_type = 'fpn',  # Decoder type, can be 'fpn' or 'deeplabv3plus'
            output_dim=3,  # Output dimension for lane and direction extraction
        )
        
    ),
    losses=dict(
        lane_and_direction_loss=dict(
            lane_cross_entropy_loss_weight=1.0,
            lane_dice_loss_weight=0.3,
            direction_l2_loss_weight=1.0,
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

    ),
    loggers=dict(
        tensorboard=dict(
            type='Tensorboard',
            log_dir=os.path.join(project_dir, 'logs'),
        ),
    ),
)