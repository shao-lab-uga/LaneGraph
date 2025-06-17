import os
import torch
import warnings
import argparse
from tqdm import tqdm
from utils.training_utils import load_checkpoint, save_checkpoint
from utils.config_utils import load_config
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
import einops
from model import LaneAndDirectionExtractionModel
# warnings.filterwarnings("ignore")
from lane_and_direction_loss import LaneAndDirectionExtractionLoss
from laneAndDirectionExtraction.dataloader import get_dataloaders
def setup(config, gpu_id):
    """
    Setup the model, optimizer, scheduler, and losses.
    """
    # //[ ] The model config need to be updated if the model is changed
    model_config = config.models
    model = LaneAndDirectionExtractionModel(model_config.lane_and_direction_extraction_model).to(gpu_id)

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
    lane_and_direction_loss_config = config_losses.lane_and_direction_loss
    lane_and_direction_loss = LaneAndDirectionExtractionLoss(
        device=gpu_id, 
        config=lane_and_direction_loss_config
    )
    return model, optimizer, scheduler, lane_and_direction_loss

def model_training(gpu_id, world_size, config, enable_ddp=True):
    
    project_dir = config.project_dir
    dataloaders_config = config.dataloaders
    os.path.exists(project_dir) or os.makedirs(project_dir)

    train_config = config.train
    checkpoint_dir = train_config.checkpoint_dir
    max_epochs = train_config.max_epochs
    checkpoint_interval = train_config.checkpoint_interval
    checkpoint_total_limit = train_config.checkpoint_total_limit
    log_interval = train_config.log_interval
    torch.autograd.set_detect_anomaly(True)
    
    train_dataloader, validate_dataloader, test_dataloader = get_dataloaders(dataloaders_config)

    logger_config = config.loggers
    tensorboard_config = logger_config.tensorboard
    logger = SummaryWriter(log_dir=tensorboard_config.log_dir)

    model, optimizer, scheduler, lane_and_direction_loss = setup(config, gpu_id, enable_ddp=enable_ddp)
    continue_ep, global_step = load_checkpoint(model, optimizer, scheduler, checkpoint_dir, gpu_id)
    

    for epoch in tqdm(range(continue_ep, max_epochs)):
        if epoch<continue_ep:
            if gpu_id == 0:
                print("\nskip epoch {}/{}".format(epoch + 1, max_epochs))
            continue
        
        model.train()

        data = train_dataloader.get_batch()
            
        input_image, region_mask, lane_groundtruth, direction_groundtruth = data
        def parse_outputs(outputs):
            """
            Parse the outputs from the model.
            Args:
                outputs: The outputs from the model. [B, H ,W, 3]
            Returns:
                lane_predicted: The predicted lane logits [B, 1, H ,W].
                direction_predicted: The predicted direction logits [B, 2, H ,W].
            """
            lane_predicted = outputs[..., 0:1]
            direction_predicted = outputs[..., 1:3]
            return lane_predicted, direction_predicted
        
        lane_predicted, direction_predicted = parse_outputs(model.forward(input_image))
        loss_dic = lane_and_direction_loss.compute(lane_predicted, direction_predicted, region_mask, lane_groundtruth, direction_groundtruth)
        loss_value = torch.sum(sum(lane_and_direction_loss.values()))
        # backward pass
        optimizer.zero_grad()
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        
        global_step += 1
        
        if gpu_id == 0:
            if epoch % log_interval == 0:
                logger.add_scalars(main_tag="lane_and_direction_loss", tag_scalar_dict=loss_dic, global_step=global_step)  
        
        scheduler.step()
        # [ ] Currently the validation is not implemented, but it can be added later.
        ## validate
        
        # occupancy_flow_map_loss.reset()
        # trajectory_loss.reset()
        # occupancy_flow_map_metrics = OccupancyFlowMapMetrics(gpu_id, no_warp=False)
        
        # torch.cuda.empty_cache()
        # model.eval()
        # if enable_ddp:
        #     val_dataloader.sampler.set_epoch(epoch)
        # vehicles_observed_occupancy_auc = []
        # vehicles_observed_occupancy_iou = []
        # observed_occupancy_cross_entropy = []
        # with torch.no_grad():
        #     loop = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        #     for batch_idx, data in loop:
                
        #         input_dict, ground_truth_dict = parse_data(data, gpu_id, config)
        
        #         input_dict = input_dict['cur']
        #         ground_truth_dict = ground_truth_dict['cur']

        #         # get the input
        #         his_occupancy_map = input_dict['his/observed_occupancy_map']
        #         # add a slight noise to the input
                

        #         # get the ground truth
        #         gt_observed_occupancy_logits = ground_truth_dict['pred/observed_occupancy_map']
        #         gt_valid_mask = ground_truth_dict['pred/valid_mask']
        #         gt_occupancy_flow_map_mask = (torch.sum(gt_valid_mask, dim=-2) > 0)

                
        #         pred_observed_occupancy_logits = model.forward(his_occupancy_map, gt_observed_occupancy_logits, training=False)

        #         loss_dic = occupancy_flow_map_loss.compute_occypancy_map_loss(pred_observed_occupancy_logits, gt_observed_occupancy_logits, gt_occupancy_flow_map_mask)

        #         observed_occupancy_cross_entropy.append(loss_dic['observed_occupancy_cross_entropy'])
        #         pred_observed_occupancy_logits = torch.sigmoid(pred_observed_occupancy_logits)
                
        #         occupancy_flow_map_metrics_dict = occupancy_flow_map_metrics.compute_occupancy_metrics(pred_observed_occupancy_logits, gt_observed_occupancy_logits, gt_occupancy_flow_map_mask)
        #         # print(occupancy_flow_map_metrics_dict)
        #         vehicles_observed_occupancy_auc.append(occupancy_flow_map_metrics_dict['vehicles_observed_occupancy_auc'])
        #         vehicles_observed_occupancy_iou.append(occupancy_flow_map_metrics_dict['vehicles_observed_occupancy_iou'])
                
        #     occupancy_flow_map_metrics_res_dict = {'vehicles_observed_occupancy_auc': torch.mean(torch.stack(vehicles_observed_occupancy_auc)),
        #                                             'vehicles_observed_occupancy_iou': torch.mean(torch.stack(vehicles_observed_occupancy_iou))}
        #     occupancy_flow_map_loss_res_dict = {'observed_occupancy_cross_entropy': torch.mean(torch.stack(observed_occupancy_cross_entropy))}
        #     if gpu_id == 0:
        #         logger.add_scalars(main_tag="val_occupancy_flow_map_metrics", tag_scalar_dict=occupancy_flow_map_metrics_res_dict, global_step=global_step)
        #         logger.add_scalars(main_tag="val_occupancy_flow_map_loss", tag_scalar_dict=occupancy_flow_map_loss_res_dict, global_step=global_step)

        if gpu_id == 0:
            if (epoch+1) % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, checkpoint_dir, checkpoint_total_limit)


if __name__ == "__main__":
    # ============= Parse Argument =============
    parser = argparse.ArgumentParser(description="options")
    parser.add_argument("--config", type=str, default="configs/train_lane_and_direction_extraction.py", help="config file")
    args = parser.parse_args()
    # ============= Load Configuration =============
    config = load_config(args.config)
    world_size = torch.cuda.device_count()
    model_training(0, world_size, config, enable_ddp=False)




        # def visualization(self, step, result=None, batch=None):
        # direction_img = np.zeros((self.image_size, self.image_size, 3))

        # if step % SAVE_FIGURE_INTERVAL == 0:
        #     # ind = ((step // 100) * self.batch_size) % 128
        #     ind = (step // SAVE_FIGURE_INTERVAL) * self.batch_size

        #     # batch[3] = np.clip(batch[3], -1, 1)
        #     # result[1] = np.clip(result[1], -1, 1)

        #     for i in range(self.batch_size):
        #         idStr = "_{}_{}_{}".format(
        #             int(step // (self.epochsize)), step, i
        #         )  # convention: epoch, step, index of figure

        #         Image.fromarray(
        #             ((batch[0][i, :, :, :] + 0.5) * 255).astype(np.uint8)
        #         ).save(os.path.join(self.validationfolder, "input{}.jpg".format(idStr)))
        #         Image.fromarray(((batch[1][i, :, :, 0]) * 255).astype(np.uint8)).save(
        #             os.path.join(self.validationfolder, "mask{}.jpg".format(idStr))
        #         )
        #         Image.fromarray(((batch[2][i, :, :, 0]) * 255).astype(np.uint8)).save(
        #             os.path.join(self.validationfolder, "target{}.jpg".format(idStr))
        #         )
        #         if self.use_sdmap:
        #             Image.fromarray(
        #                 ((batch[4][i, :, :, 0]) * 255).astype(np.uint8)
        #             ).save(
        #                 os.path.join(self.validationfolder, "sdmap{}.jpg".format(idStr))
        #             )

        #         direction_img[:, :, 2] = batch[3][i, :, :, 0] * 127 + 127
        #         direction_img[:, :, 1] = batch[3][i, :, :, 1] * 127 + 127
        #         direction_img[:, :, 0] = 127

        #         Image.fromarray(direction_img.astype(np.uint8)).save(
        #             os.path.join(
        #                 self.validationfolder, "targe_direction{}.jpg".format(idStr)
        #             )
        #         )

        #         Image.fromarray(((result[1][i, :, :, 0]) * 255).astype(np.uint8)).save(
        #             os.path.join(self.validationfolder, "output{}.jpg".format(idStr))
        #         )

        #         direction_img[:, :, 2] = (
        #             np.clip(result[1][i, :, :, 1], -1, 1) * 127 + 127
        #         )
        #         direction_img[:, :, 1] = (
        #             np.clip(result[1][i, :, :, 2], -1, 1) * 127 + 127
        #         )
        #         direction_img[:, :, 0] = 127

        #         Image.fromarray(direction_img.astype(np.uint8)).save(
        #             os.path.join(
        #                 self.validationfolder, "output_direction{}.jpg".format(idStr)
        #             )
        #         )

        # return False