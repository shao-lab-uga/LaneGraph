import os
import torch
import argparse
import einops
import warnings
warnings.filterwarnings("ignore")
from model import ReachableLaneExtractionAndValidation
from reachableLaneExtractionValidation.reachable_lane_extraction_validation_loss import ReachableLaneExtractionValidationLoss
from utils.config_utils import load_config
from utils.training_utils import load_checkpoint, save_checkpoint
from reachableLaneExtractionValidation.dataloader import get_dataloaders
from utils.inference_utils import visualize_reachable_lane
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from torch.amp import autocast, GradScaler


def setup(config, gpu_id):
    """
    Setup the model, optimizer, scheduler, and losses.
    """
    # //[ ] The model config need to be updated if the model is changed
    model_config = config.models
    model = ReachableLaneExtractionAndValidation(
        model_config.reachable_lane_extraction_model,
        model_config.reachable_lane_validation_model
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
    reachable_lane_extraction_loss_config = config_losses.reachable_lane_extraction_loss

    reachable_lane_extraction_loss = ReachableLaneExtractionValidationLoss(
        device=gpu_id, 
        config=reachable_lane_extraction_loss_config
    )
    return model, optimizer, scheduler, reachable_lane_extraction_loss

def model_training(gpu_id, world_size, config):
    scaler = GradScaler(device=gpu_id)
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

    model, optimizer, scheduler, reachable_lane_extraction_loss = setup(config, gpu_id)
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
                    reachable_lane_groundtruth: The ground truth for lane [B, 3, H, W].
                        reachable_lane_groundtruth[1:2] are reachable lane for the node A,
                        reachable_lane_groundtruth[2:3] are reachable lane for the node B,
                    reachable_label_groundtruth: The ground truth for reachable label [B, 1].
                    direction_groundtruth: The ground truth for direction [B, 2, H, W].
                """
                                    # def train(self, x_in, x_connector, target, target_label, context, lr):
                input_image, connector_features, reachable_lane_groundtruth, reachable_label_groundtruth, direction_context = inputs
                batch_size, image_size, _, _ = input_image.shape
                position_encoding = np.zeros((batch_size, image_size, image_size, 2))
                for i in range(image_size):
                    position_encoding[:, i, :, 0] = float(i) / image_size
                    position_encoding[:, :, i, 0] = float(i) / image_size

                input_image = torch.from_numpy(input_image).float().to(gpu_id)  # [B, H, W, 3]
                input_image = einops.rearrange(input_image, 'b h w c -> b c h w') # [B, 3, H, W]

                connector_features = torch.from_numpy(connector_features).float().to(gpu_id)  # [B, H, W, 1]
                connector_features = einops.rearrange(connector_features, 'b h w c -> b c h w')  # [B, 1, H, W]

                reachable_lane_groundtruth = torch.from_numpy(reachable_lane_groundtruth).float().to(gpu_id) # [B, H, W, 3]
                reachable_lane_groundtruth = einops.rearrange(reachable_lane_groundtruth, 'b h w c -> b c h w')  # [B, 3, H, W]

                reachable_label_groundtruth = torch.from_numpy(reachable_label_groundtruth).float().to(gpu_id)  # [B, 1]

                direction_context = torch.from_numpy(direction_context).float().to(gpu_id)  # [B, H, W, 2]
                direction_context = einops.rearrange(direction_context, 'b h w c -> b c h w')  # [B, 2, H, W]

                position_encoding = torch.from_numpy(position_encoding).float().to(gpu_id)  # [B, H, W, 2]
                position_encoding = einops.rearrange(position_encoding, 'b h w c -> b c h w') # [B, 2, H, W]

                input_features_node_a = torch.cat([
                    input_image,
                    connector_features[:, 0:3, :, :],  # Features for node A
                    direction_context,
                    position_encoding
                    ], dim=1)

                input_features_node_b = torch.cat([
                    input_image,
                    connector_features[:, 3:6,:, :],  # Features for node B
                    direction_context,
                    position_encoding
                    ], dim=1)
                
                input_features_validation = torch.cat([
                    connector_features,  # Features for both nodes
                    direction_context,
                    position_encoding
                ], dim=1)

                return input_features_node_a, input_features_node_b, input_features_validation, reachable_lane_groundtruth, reachable_label_groundtruth

            input_features_node_a, input_features_node_b, input_features_validation, reachable_lane_groundtruth, reachable_label_groundtruth = parse_inputs(inputs)


            with autocast(device_type='cuda', dtype=torch.float16):
                # forward pass
                reachable_lane_predicted_node_a, reachable_lane_predicted_node_b, reachable_label_predicted = model.forward(
                    input_features_node_a=input_features_node_a,
                    input_features_node_b=input_features_node_b,
                    input_features_validation=input_features_validation
                )

                reachable_lane_extraction_loss_dic = reachable_lane_extraction_loss.compute(
                    reachable_lane_predicted_a= reachable_lane_predicted_node_a,
                    reachable_lane_predicted_b= reachable_lane_predicted_node_b,
                    reachable_lane_groundtruth_a= reachable_lane_groundtruth[:, 1:2, :, :],
                    reachable_lane_groundtruth_b= reachable_lane_groundtruth[:, 2:3, :, :],
                    reachable_label_predicted= reachable_label_predicted,
                    reachable_label_groundtruth= reachable_label_groundtruth,
                )
                loss_value = torch.sum(sum(reachable_lane_extraction_loss_dic.values()))
                if torch.isnan(loss_value) or torch.isinf(loss_value):
                    print("Loss is NaN or Inf, skipping this batch.")
                    print(f"Loss value: {reachable_lane_extraction_loss_dic}")
                    continue
            # backward pass
            optimizer.zero_grad()
            scaler.scale(loss_value).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            reachable_lane_extraction_loss.update(reachable_lane_extraction_loss_dic)

            global_step += 1
            lane_and_direction_loss_res_dict = reachable_lane_extraction_loss.get_result()
            if gpu_id == 0:
                if batch_idx % log_interval == 0:
                    logger.add_scalars(main_tag="lane_and_direction_loss", tag_scalar_dict=lane_and_direction_loss_res_dict, global_step=global_step)
                    input_image, connector_features, reachable_lane_groundtruth, reachable_label_groundtruth, direction_context = inputs
                    reachable_lane_predicted_node_a = einops.rearrange(reachable_lane_predicted_node_a, 'b c h w -> b h w c')
                    reachable_lane_predicted_node_a = reachable_lane_predicted_node_a.detach().cpu().numpy()
                    reachable_lane_predicted_node_b = einops.rearrange(reachable_lane_predicted_node_b, 'b c h w -> b h w c')
                    reachable_lane_predicted_node_b = reachable_lane_predicted_node_b.detach().cpu().numpy()
                    visualize_reachable_lane(visualize_output_path, global_step,
                                            input_satellite_image=input_image,
                                            reachable_lane_predicted_node_a=reachable_lane_predicted_node_a,
                                            reachable_lane_predicted_node_b=reachable_lane_predicted_node_b,
                                            reachable_lane_groundtruth=reachable_lane_groundtruth,
                                            direction_groundtruth= direction_context,
                                            visulize_all_samples=False,
                                            visualize_groundtruth=True)
                    torch.cuda.empty_cache()
            if (global_step + 1) % 50 == 0:
                train_dataloader.preload()
        scheduler.step()
        # [ ] Currently the validation is not implemented, but it can be added later.

        if gpu_id == 0:
            if (epoch+1) % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, checkpoint_dir, checkpoint_total_limit)
                
            

if __name__ == "__main__":
    # ============= Parse Argument =============
    parser = argparse.ArgumentParser(description="options")
    parser.add_argument("--config", type=str, default="configs/train_reachable_lane_extraction_validation.py", help="config file")
    args = parser.parse_args()
    # ============= Load Configuration =============
    config = load_config(args.config)
    world_size = torch.cuda.device_count()
    model_training(0, world_size, config)