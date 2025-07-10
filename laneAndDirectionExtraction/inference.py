import os
import torch
import argparse
import einops
import warnings
warnings.filterwarnings("ignore")
from laneAndDirectionExtraction.model import UnetResnet34
from utils.config_utils import load_config
from laneAndDirectionExtraction.dataloader import get_dataloaders
from utils.inference_utils import visualizatize_lane_and_direction, load_model
from tqdm import tqdm
import scipy.ndimage

def setup(config, gpu_id):
    """
    Setup the model, optimizer, scheduler, and losses.
    """
    # //[ ] The model config need to be updated if the model is changed
    model_config = config.models
    model = UnetResnet34(model_config.lane_and_direction_extraction_model).to(gpu_id)
    return model

def model_inference(gpu_id, world_size, config):
    
    project_dir = config.project_dir
    dataloaders_config = config.dataloaders
    os.path.exists(project_dir) or os.makedirs(project_dir)

    test_config = config.test
    checkpoint_path = test_config.checkpoint_path
    max_epochs = test_config.max_epochs
    epoch_sisze = test_config.epoch_size
    visualize_output_path = test_config.visualize_output_path
    log_interval = test_config.log_interval
    torch.autograd.set_detect_anomaly(True)
    
    _, _, test_dataloader = get_dataloaders(dataloaders_config)
    model = setup(config, gpu_id)
    load_model(model=model, checkpoint_path=checkpoint_path)
    
    global_step = 0
    for epoch in tqdm(range(max_epochs)):
        model.train()
        for batch_idx in tqdm(range(epoch_sisze)):
            inputs = test_dataloader.get_batch()
            def parse_inputs(inputs):   
                """
                Parse the inputs from the dataloader.
                Args:
                    data: The data from the dataloader (numpy array).
                Returns:
                    input_image: The input image [B, 3, H, W].
                    region_mask: The region mask [B, 1, H, W].
                    lane_groundtruth: The ground truth for lane [B, 1, H, W].
                    direction_groundtruth: The ground truth for direction [B, 2, H, W].
                """
                input_image, region_mask, lane_groundtruth, direction_groundtruth = inputs
                
                input_image = torch.from_numpy(input_image).float().to(gpu_id)  # [B, H, W, 3]
                input_image = einops.rearrange(input_image, 'b h w c -> b c h w') # [B, 3, H, W]

                return input_image

            input_image, region_mask, lane_groundtruth, direction_groundtruth = parse_inputs(inputs)

            def parse_outputs(outputs):
                """
                Parse the outputs from the model.
                Args:
                    outputs: The outputs from the model. [B, 4, H ,W]
                Returns:
                    lane_predicted: The predicted lane logits [B, 2, H ,W].
                    direction_predicted: The predicted direction logits [B, 2, H ,W].
                """
                lane_predicted = outputs[:, 0:2, :, :]  # [B, 2, H, W]
                direction_predicted = outputs[:, 2:4, :, :]
                return lane_predicted, direction_predicted

            lane_predicted, direction_predicted = parse_outputs(model.forward(input_image))
            if gpu_id == 0:
                if batch_idx % log_interval == 0:
                    input_image, region_mask, lane_groundtruth, direction_groundtruth = inputs
                    lane_predicted = einops.rearrange(lane_predicted, 'b c h w -> b h w c')
                    lane_predicted = lane_predicted.detach().cpu().numpy()
                    direction_predicted = einops.rearrange(direction_predicted, 'b c h w -> b h w c')
                    direction_predicted = direction_predicted.detach().cpu().numpy()
                    visualizatize_lane_and_direction(visualize_output_path, global_step,
                                                    input_satellite_image=input_image,
                                                    region_mask=region_mask,
                                                    lane_predicted=lane_predicted,
                                                    direction_predicted=direction_predicted,
                                                    lane_groundtruth=lane_groundtruth,
                                                    direction_groundtruth=direction_groundtruth,
                                                    visulize_all_samples=False,
                                                    visualize_groundtruth=True
                                                    )
            global_step += 1
            if (global_step + 1) % 50 == 0:
                test_dataloader.preload()
                
            

if __name__ == "__main__":
    # ============= Parse Argument =============
    parser = argparse.ArgumentParser(description="options")
    parser.add_argument("--config", type=str, default="configs/eval_lane_and_direction_extraction.py", help="config file")
    args = parser.parse_args()
    # ============= Load Configuration =============
    config = load_config(args.config)
    world_size = torch.cuda.device_count()
    model_inference(0, world_size, config)