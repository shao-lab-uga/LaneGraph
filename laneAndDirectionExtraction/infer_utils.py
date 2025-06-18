from PIL import Image 
import numpy as np
import os

def visualizatize_lane_and_direction(visualize_output_path, step, input_satellite_image, region_mask, lane_predicted, direction_predicted,  lane_groundtruth, direction_groundtruth, visulize_all=False):
    """
    visualization function for lane and direction extraction.
    Args:
        visualize_output_path: str (path to save the visualizations)
        step: int (current step in training)
        
        input_satellite_image: [B, H, W, 3] (input satellite image)
        region_mask: [B, H, W, 1] (mask to include/exclude pixels)
        lane_predicted: [B, H, W, 2] (raw logits for lane)
        direction_predicted: [B, H, W, 2] (raw logits for direction)
        lane_groundtruth: [B, H, W, 1] (ground truth for lane)
        direction_groundtruth: [B, H, W, 2] (ground truth for direction)
        visulize_all: bool (whether to visualize all images in the batch)
    """
    os.path.exists(visualize_output_path) or os.makedirs(visualize_output_path)
    batch_size, _, image_size, _ = input_satellite_image.shape
    visualize_range = range(batch_size) if visulize_all else [0]  # Visualize all images in the batch if visulize_all is True, otherwise only the first image
    for i in visualize_range:
        
        Image.fromarray(((input_satellite_image[i,:,:,:] + 0.5) * 255).astype(np.uint8) ).save(os.path.join(visualize_output_path, f"input_{step}_{i}.jpg"))
        Image.fromarray(((region_mask[i,:,:,0]) * 255).astype(np.uint8) ).save(os.path.join(visualize_output_path, f"region_mask_{step}_{i}.jpg"))
        Image.fromarray(((lane_groundtruth[i,:,:,0]) * 255).astype(np.uint8) ).save(os.path.join(visualize_output_path, f"lane_groundtruth{step}_{i}.jpg"))
        
        direction_groundtruth_image = np.zeros((image_size, image_size, 3))
        direction_groundtruth_image[:,:,2] = direction_groundtruth[i,:,:,0] * 127 + 127
        direction_groundtruth_image[:,:,1] = direction_groundtruth[i,:,:,1] * 127 + 127
        direction_groundtruth_image[:,:,0] = 127
        Image.fromarray(direction_groundtruth_image.astype(np.uint8)).save(os.path.join(visualize_output_path, f"direction_groundtruth_{step}_{i}.jpg"))
        
        Image.fromarray(((lane_predicted[i,:,:,0]) * 255).astype(np.uint8)).save(os.path.join(visualize_output_path, f"lane_predicted_{step}_{i}.jpg"))
        
        direction_predicted_image = np.zeros((image_size, image_size, 3))
        direction_predicted_image[:,:,2] = np.clip(direction_predicted[i,:,:,0],-1,1) * 127 + 127
        direction_predicted_image[:,:,1] = np.clip(direction_predicted[i,:,:,1],-1,1) * 127 + 127
        direction_predicted_image[:,:,0] = 127
        
        Image.fromarray(direction_predicted_image.astype(np.uint8)).save(os.path.join(visualize_output_path, f"direction_predicted_{step}_{i}.jpg"))
            
        
    return False
