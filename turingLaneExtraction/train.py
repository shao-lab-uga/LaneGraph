import os
import torch
import argparse
import einops
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model import LaneExtractionModel
from turning_lane_extraction_loss import TurningLaneExtractionLoss
from utils.inference_utils import visualize_lane
from utils.config_utils import load_config
from utils.training_utils import load_checkpoint, save_checkpoint
from dataloader import get_dataloaders





def setup(config, gpu_id):
    """
    Setup the model, optimizer, scheduler, and losses.
    """
    # //[ ] The model config need to be updated if the model is changed
    model_config = config.models
    model = LaneExtractionModel(
        model_config.lane_extraction_model,
        ).to(gpu_id)

    # Load the optimizer
    optimizer_config = config.optimizer
    def get_optimizer(optimizer_config):
        optimizer_type = optimizer_config.type
        if optimizer_type == "AdamW":
            return torch.optim.AdamW(
                model.parameters(), 
                lr=optimizer_config.learning_rate,            # Base learning rate
                betas=optimizer_config.betas,  # Slightly higher β2 for smoother updates
                eps=optimizer_config.eps,           # Avoids division by zero
                weight_decay=optimizer_config.weight_decay   # Encourages generalization
            )
        if optimizer_type == "NAdam":
            return torch.optim.NAdam(
                params=model.parameters(), 
                lr=optimizer_config.learning_rate,
                weight_decay=optimizer_config.weight_decay
            )
    optimizer = get_optimizer(optimizer_config)
    # Load the scheduler
    scheduler_config = config.scheduler
    def get_scheduler(scheduler_config):
        scheduler_type = scheduler_config.type
        if scheduler_type == "StepLR":
            return torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer, 
                step_size=scheduler_config.step_size,
                gamma=scheduler_config.gamma
            )
        if scheduler_type == "CosineAnnealingWarmRestarts":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer, 
                T_0=scheduler_config.T_0,
                T_mult=scheduler_config.T_mult,
                eta_min=scheduler_config.eta_min
            )
    scheduler = get_scheduler(scheduler_config)
    # Get the losses config
    config_losses = config.losses
    lane_extraction_loss_config = config_losses.lane_extraction_loss

    lane_extraction_loss = TurningLaneExtractionLoss(
        device=gpu_id, 
        config=lane_extraction_loss_config
    )
    return model, optimizer, scheduler, lane_extraction_loss

def model_training(gpu_id, world_size, config):
    # scaler = GradScaler(device=gpu_id)
    project_dir = config.project_dir
    dataloaders_config = config.dataloaders
    os.path.exists(project_dir) or os.makedirs(project_dir)

    train_config = config.train
    checkpoint_dir = train_config.checkpoint_dir
    visualize_output_path = train_config.visualize_output_path
    epoch_sisze = train_config.epoch_size
    max_epochs = train_config.max_epochs
    checkpoint_interval = train_config.checkpoint_interval
    checkpoint_total_limit = train_config.checkpoint_total_limit
    log_interval = train_config.log_interval
    torch.autograd.set_detect_anomaly(True)
    
    train_dataloader, validate_dataloader, test_dataloader = get_dataloaders(dataloaders_config)

    logger_config = config.loggers
    tensorboard_config = logger_config.tensorboard
    logger = SummaryWriter(log_dir=tensorboard_config.log_dir)

    model, optimizer, scheduler, lane_extraction_loss = setup(config, gpu_id)
    continue_ep, global_step = load_checkpoint(model, optimizer, scheduler, checkpoint_dir, gpu_id)
    

    for epoch in tqdm(range(continue_ep, max_epochs)):
        if epoch<continue_ep:
            if gpu_id == 0:
                print("\nskip epoch {}/{}".format(epoch + 1, max_epochs))
            continue
        
        model.train()
        for batch_idx in tqdm(range(epoch_sisze)):
            inputs = train_dataloader.get_batch()
            def parse_inputs(inputs):   
                """
                Parse the inputs from the dataloader.
                Args:
                    data: The data from the dataloader (numpy array).
                Returns:
                    input_image: The input image [B, 3, H, W].
                    connector_features: The region mask [B, 7, H, W].
                        connector_features[0:3] are features for the node A,
                        connector_features[3:6] are features for the node B,
                        Note: connector_features[6] is deprecated.
                    lane_groundtruth: The ground truth for lane [B, 1, H, W].
                        lane_groundtruth[0:1] are the ground truth for the lane.
                    direction_groundtruth: The ground truth for direction [B, 2, H, W].
                """
                input_image, connector_features, lane_groundtruth, direction_context = inputs
                batch_size, image_size, _, _ = input_image.shape
                position_encoding = np.zeros((batch_size, image_size, image_size, 2))
                for i in range(image_size):
                    position_encoding[:, i, :, 0] = float(i) / image_size
                    position_encoding[:, :, i, 0] = float(i) / image_size

                input_image = torch.from_numpy(input_image).float().to(gpu_id)  # [B, H, W, 3]
                input_image = einops.rearrange(input_image, 'b h w c -> b c h w') # [B, 3, H, W]

                connector_features = torch.from_numpy(connector_features).float().to(gpu_id)  # [B, H, W, 1]
                connector_features = einops.rearrange(connector_features, 'b h w c -> b c h w')  # [B, 1, H, W]

                lane_groundtruth = torch.from_numpy(lane_groundtruth).float().to(gpu_id) # [B, H, W, 1]
                lane_groundtruth = einops.rearrange(lane_groundtruth, 'b h w c -> b c h w')  # [B, 1, H, W]

                direction_context = torch.from_numpy(direction_context).float().to(gpu_id)  # [B, H, W, 2]
                direction_context = einops.rearrange(direction_context, 'b h w c -> b c h w')  # [B, 2, H, W]

                position_encoding = torch.from_numpy(position_encoding).float().to(gpu_id)  # [B, H, W, 2]
                position_encoding = einops.rearrange(position_encoding, 'b h w c -> b c h w') # [B, 2, H, W]


                input_features_node= torch.cat([
                    input_image,
                    connector_features,  # Features for nodes
                    direction_context,
                    position_encoding
                    ], dim=1)
                

                return input_features_node, lane_groundtruth

            input_features_node, lane_groundtruth = parse_inputs(inputs)


            # forward pass
            lane_predicted= model.forward(
                input_features_node=input_features_node,
            )
            lane_extraction_loss_dic = lane_extraction_loss.compute(
                lane_predicted= lane_predicted,
                lane_groundtruth= lane_groundtruth,
            )
            loss_value = torch.sum(sum(lane_extraction_loss_dic.values()))
            # backward pass
            optimizer.zero_grad()
            # scaler.scale(loss_value).backward()
            loss_value.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            # scaler.step(optimizer)
            optimizer.step()
            # scaler.update()
            lane_extraction_loss.update(lane_extraction_loss_dic)

            global_step += 1
            lane_loss_res_dict = lane_extraction_loss.get_result()
            if gpu_id == 0:
                if batch_idx % log_interval == 0:
                    logger.add_scalars(main_tag="lane_loss", tag_scalar_dict=lane_loss_res_dict, global_step=global_step)
                    input_image, connector_features, lane_groundtruth, direction_context = inputs
                    lane_predicted = einops.rearrange(lane_predicted, 'b c h w -> b h w c')
                    lane_predicted = lane_predicted.detach().cpu().numpy()
                    visualize_lane(visualize_output_path, global_step,
                                            input_satellite_image=input_image,
                                            lane_predicted=lane_predicted,
                                            lane_groundtruth=lane_groundtruth,
                                            direction_groundtruth= direction_context,
                                            visulize_all_samples=False,
                                            visualize_groundtruth=True)
            if (global_step + 1) % 20 == 0:
                train_dataloader.preload()
        scheduler.step()
        
        # [ ] Currently the validation is not implemented, but it can be added later.
        if gpu_id == 0:
            if (epoch+1) % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, checkpoint_dir, checkpoint_total_limit)
        

if __name__ == "__main__":
    # ============= Parse Argument =============
    parser = argparse.ArgumentParser(description="options")
    parser.add_argument("--config", type=str, default="configs/train_lane_extraction.py", help="config file")
    args = parser.parse_args()
    # ============= Load Configuration =============
    config = load_config(args.config)
    world_size = torch.cuda.device_count()
    model_training(0, world_size, config)