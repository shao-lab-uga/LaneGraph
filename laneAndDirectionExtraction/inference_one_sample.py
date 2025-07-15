import os
import torch
import argparse
import einops
import warnings
warnings.filterwarnings("ignore")
from laneAndDirectionExtraction.model import UnetResnet34
from utils.config_utils import load_config
import imageio
import numpy as np
from utils.inference_utils import visualize_lane_and_direction_inference, load_model


import sys
sys.path.append('..')
from image_postprocessing import normalize_image_for_model_input


def setup(config, gpu_id):
    """
    Setup the model, optimizer, scheduler, and losses.
    """

    model_config = config.models
    model = UnetResnet34(model_config.lane_and_direction_extraction_model).to(gpu_id)
    return model

def model_inference(input_image, gpu_id, config):
    
    project_dir = config.project_dir
    os.path.exists(project_dir) or os.makedirs(project_dir)

    test_config = config.test
    checkpoint_path = test_config.checkpoint_path
    visualize_output_path = test_config.visualize_output_path
    
    model = setup(config, gpu_id)
    load_model(model=model, checkpoint_path=checkpoint_path)
    
    def parse_inputs(input_image):   
        """
        convert the input image to the format required by the model.
        Args:
            input_image: The original input image [H, W, 3].
        Returns:
            input_image: The input image [1, 3, H, W].
        """
        
        input_image = torch.from_numpy(input_image).float().to(gpu_id)  # [H, W, 3]
        input_image = einops.rearrange(input_image, 'h w c -> 1 c h w')  # [1, 3, H, W]
        return input_image
    
    input_image = parse_inputs(input_image)  # [B, 3, H, W]
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
    with torch.no_grad():
        model.eval()
        lane_predicted, direction_predicted = parse_outputs(model.forward(input_image))
    
    input_image = einops.rearrange(input_image, 'b c h w -> b h w c')
    input_image = input_image.cpu().numpy()
    lane_predicted = einops.rearrange(lane_predicted, 'b c h w -> b h w c')
    lane_predicted = lane_predicted.cpu().numpy()
    direction_predicted = einops.rearrange(direction_predicted, 'b c h w -> b h w c')
    direction_predicted = direction_predicted.cpu().numpy()
    visualize_lane_and_direction_inference(
        visualize_output_path=visualize_output_path,
        input_satellite_image=input_image,
        lane_predicted=lane_predicted,
        direction_predicted=direction_predicted,
        visulize_all_samples=True
    )
        
            

if __name__ == "__main__":
    # ============= Parse Argument =============
    parser = argparse.ArgumentParser(description="options")
    parser.add_argument("--config", type=str, default="configs/eval_lane_and_direction_extraction.py", help="config file")
    parser.add_argument("--input_image_path", type=str, default="./test_img.jpg", help="path to the input satellite image")
    args = parser.parse_args()
    # ============= Load Configuration =============
    config = load_config(args.config)
    input_satillite_image = imageio.imread(args.input_image_path)  # Load the input image
    input_satillite_image = normalize_image_for_model_input(input_satillite_image)
    model_inference(input_satillite_image, 0, config)

    