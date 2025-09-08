from PIL import Image 
import numpy as np
import os
import torch

# Import new post-processing utilities
import sys
sys.path.append('..')
from utils.image_postprocessing import denormalize_image_for_display, encode_direction_vectors_to_image, apply_softmax_to_logits

def load_model(model, checkpoint_path):
    """
    Loads the latest checkpoint if available and updates the model, optimizer, and scheduler.

    Args:
        model: The model to load state dict into.
    Returns:
    """
    if checkpoint_path is not None:
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Error loading model state dict from {checkpoint_path}: {e}")
    return

def visualize_lane_and_direction_inference(visualize_output_path, input_satellite_image, lane_predicted, direction_predicted, visulize_all_samples=False):
    """
    visualization function for lane and direction extraction.
    Args:
        visualize_output_path: str (path to save the visualizations)
        
        input_satellite_image: [B, H, W, 3] (input satellite image)
        lane_predicted: [B, H, W, 2] (raw logits for lane)
        direction_predicted: [B, H, W, 2] (raw logits for direction)
        visulize_all_samples: bool (whether to visualize all images in the batch)
    """
    os.path.exists(visualize_output_path) or os.makedirs(visualize_output_path)
    batch_size, _, image_size, _ = input_satellite_image.shape
    visualize_range = range(batch_size) if visulize_all_samples else [0]  
    for i in visualize_range:

        display_image = denormalize_image_for_display(input_satellite_image[i,:,:,:])
        Image.fromarray(display_image.astype(np.uint8)).save(os.path.join(visualize_output_path, f"input_{i}.jpg"))


        lane_predicted_softmax = apply_softmax_to_logits(lane_predicted[i,:,:,:])
        lane_predicted_image = np.zeros((image_size, image_size, 3))
        lane_predicted_image[:, :, :] = np.clip(lane_predicted_softmax[:,:,0][..., np.newaxis], 0, 1) * 255
        Image.fromarray(lane_predicted_image.astype(np.uint8)).save(os.path.join(visualize_output_path, f"lane_predicted_{i}.jpg"))
        

        direction_predicted_image = encode_direction_vectors_to_image(direction_predicted[i,:,:,:])
        Image.fromarray(direction_predicted_image.astype(np.uint8)).save(os.path.join(visualize_output_path, f"direction_predicted_{i}.jpg"))

def visualize_lane_and_direction(visualize_output_path, step, input_satellite_image, region_mask, lane_predicted, direction_predicted,  lane_groundtruth, direction_groundtruth, visulize_all_samples=False, visualize_groundtruth=False):
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
        visulize_all_samples: bool (whether to visualize all images in the batch)
        visualize_groundtruth: bool (whether to visualize ground truth)
    """
    os.path.exists(visualize_output_path) or os.makedirs(visualize_output_path)
    batch_size, _, image_size, _ = input_satellite_image.shape
    visualize_range = range(batch_size) if visulize_all_samples else [0]  # Visualize all images in the batch if visulize_all is True, otherwise only the first image
    
    for i in visualize_range:
        if visualize_groundtruth:

            display_image = denormalize_image_for_display(input_satellite_image[i,:,:,:])
            Image.fromarray(display_image.astype(np.uint8)).save(os.path.join(visualize_output_path, f"input_{step}_{i}.jpg"))
            
            Image.fromarray(((region_mask[i,:,:,0]) * 255).astype(np.uint8) ).save(os.path.join(visualize_output_path, f"region_mask_{step}_{i}.jpg"))
            Image.fromarray(((lane_groundtruth[i,:,:,0]) * 255).astype(np.uint8)).save(os.path.join(visualize_output_path, f"lane_groundtruth{step}_{i}.jpg"))


            direction_groundtruth_image = encode_direction_vectors_to_image(direction_groundtruth[i,:,:,:])
            Image.fromarray(direction_groundtruth_image.astype(np.uint8)).save(os.path.join(visualize_output_path, f"direction_groundtruth_{step}_{i}.jpg"))


        lane_predicted_softmax = apply_softmax_to_logits(lane_predicted[i,:,:,:])
        lane_predicted_image = np.zeros((image_size, image_size, 3))
        lane_predicted_image[:, :, :] = np.clip(lane_predicted_softmax[:,:,0][..., np.newaxis], 0, 1) * 255
        Image.fromarray(lane_predicted_image.astype(np.uint8)).save(os.path.join(visualize_output_path, f"lane_predicted_{step}_{i}.jpg"))
        

        direction_predicted_image = encode_direction_vectors_to_image(direction_predicted[i,:,:,:])
        Image.fromarray(direction_predicted_image.astype(np.uint8)).save(os.path.join(visualize_output_path, f"direction_predicted_{step}_{i}.jpg"))


def visualize_reachable_lane(visualize_output_path, step, input_satellite_image, reachable_lane_predicted_node_a, reachable_lane_predicted_node_b, reachable_lane_groundtruth, direction_groundtruth, visulize_all_samples=False, visualize_groundtruth=False):
    """
    visualization function for lane and direction extraction.
    Args:
        visualize_output_path: str (path to save the visualizations)
        step: int (current step in training)
        
        input_satellite_image: [B, H, W, 3] (input satellite image)
        reachable_lane_predicted_node_a: [B, H, W, 2] (raw logits for reachable lane A)
        reachable_lane_predicted_node_b: [B, H, W, 2] (raw logits for reachable lane B)
        reachable_lane_groundtruth: [B, H, W, 1] (ground truth for reachable lane)
        visulize_all_samples: bool (whether to visualize all images in the batch)
        direction_groundtruth: [B, H, W, 2] (ground truth for direction)
        visulize_all: bool (whether to visualize all images in the batch)
    """
    os.path.exists(visualize_output_path) or os.makedirs(visualize_output_path)
    batch_size, _, image_size, _ = input_satellite_image.shape
    visualize_range = range(batch_size) if visulize_all_samples else [0]  # Visualize all images in the batch if visulize_all is True, otherwise only the first image
    for i in visualize_range:
        if visualize_groundtruth:
            Image.fromarray(((input_satellite_image[i,:,:,:] + 0.5) * 255).astype(np.uint8) ).save(os.path.join(visualize_output_path, f"input_{step}_{i}.jpg"))
            Image.fromarray(((reachable_lane_groundtruth[i,:,:,1]) * 255).astype(np.uint8)).save(os.path.join(visualize_output_path, f"lane_groundtruth_a{step}_{i}.jpg"))
            Image.fromarray(((reachable_lane_groundtruth[i,:,:,2]) * 255).astype(np.uint8)).save(os.path.join(visualize_output_path, f"lane_groundtruth_b{step}_{i}.jpg"))

            direction_groundtruth_image = np.zeros((image_size, image_size, 3))
            direction_groundtruth_image[:,:,2] = direction_groundtruth[i,:,:,0] * 127 + 127
            direction_groundtruth_image[:,:,1] = direction_groundtruth[i,:,:,1] * 127 + 127
            direction_groundtruth_image[:,:,0] = 127
            Image.fromarray(direction_groundtruth_image.astype(np.uint8)).save(os.path.join(visualize_output_path, f"direction_groundtruth_{step}_{i}.jpg"))

        reachable_lane_predicted_node_a_softmax = np.exp(reachable_lane_predicted_node_a[i,:,:,0]) / (np.exp(reachable_lane_predicted_node_a[i,:,:,0]) + np.exp(reachable_lane_predicted_node_a[i,:,:,1]))
        reachable_lane_predicted_node_b_softmax = np.exp(reachable_lane_predicted_node_b[i,:,:,0]) / (np.exp(reachable_lane_predicted_node_b[i,:,:,0]) + np.exp(reachable_lane_predicted_node_b[i,:,:,1]))

        reachable_lane_predicted_image_a = np.zeros((image_size, image_size, 3))
        reachable_lane_predicted_image_a[:, :, :] = np.clip(reachable_lane_predicted_node_a_softmax[..., np.newaxis], 0, 1) * 255
        Image.fromarray(reachable_lane_predicted_image_a.astype(np.uint8)).save(os.path.join(visualize_output_path, f"reachable_lane_predicted_a_{step}_{i}.jpg"))
        reachable_lane_predicted_image_b = np.zeros((image_size, image_size, 3))
        reachable_lane_predicted_image_b[:, :, :] = np.clip(reachable_lane_predicted_node_b_softmax[..., np.newaxis], 0, 1) * 255
        Image.fromarray(reachable_lane_predicted_image_b.astype(np.uint8)).save(os.path.join(visualize_output_path, f"reachable_lane_predicted_b_{step}_{i}.jpg"))
        
def visualize_lane(visualize_output_path, step, input_satellite_image, lane_predicted, lane_groundtruth, direction_groundtruth, visulize_all_samples=False, visualize_groundtruth=False):
    """
    visualization function for lane and direction extraction.
    Args:
        visualize_output_path: str (path to save the visualizations)
        step: int (current step in training)
        
        input_satellite_image: [B, H, W, 3] (input satellite image)
        lane_predicted: [B, H, W, 2] (raw logits for reachable lane A)
        reachable_lane_groundtruth: [B, H, W, 1] (ground truth for reachable lane)
        visulize_all_samples: bool (whether to visualize all images in the batch)
        direction_groundtruth: [B, H, W, 2] (ground truth for direction)
        visulize_all: bool (whether to visualize all images in the batch)
    """
    os.path.exists(visualize_output_path) or os.makedirs(visualize_output_path)
    batch_size, _, image_size, _ = input_satellite_image.shape
    visualize_range = range(batch_size) if visulize_all_samples else [0]  # Visualize all images in the batch if visulize_all is True, otherwise only the first image
    for i in visualize_range:
        if visualize_groundtruth:
            Image.fromarray(((input_satellite_image[i,:,:,:] + 0.5) * 255).astype(np.uint8) ).save(os.path.join(visualize_output_path, f"input_{step}_{i}.jpg"))
            Image.fromarray(((lane_groundtruth[i,:,:,0]) * 255).astype(np.uint8)).save(os.path.join(visualize_output_path, f"lane_groundtruth{step}_{i}.jpg"))

            direction_groundtruth_image = np.zeros((image_size, image_size, 3))
            direction_groundtruth_image[:,:,2] = direction_groundtruth[i,:,:,0] * 127 + 127
            direction_groundtruth_image[:,:,1] = direction_groundtruth[i,:,:,1] * 127 + 127
            direction_groundtruth_image[:,:,0] = 127
            Image.fromarray(direction_groundtruth_image.astype(np.uint8)).save(os.path.join(visualize_output_path, f"direction_groundtruth_{step}_{i}.jpg"))

        lane_predicted_softmax = np.exp(lane_predicted[i,:,:,0]) / (np.exp(lane_predicted[i,:,:,0]) + np.exp(lane_predicted[i,:,:,1]))

        lane_predicted_image = np.zeros((image_size, image_size, 3))
        lane_predicted_image[:, :, :] = np.clip(lane_predicted_softmax[..., np.newaxis], 0, 1) * 255
        Image.fromarray(lane_predicted_image.astype(np.uint8)).save(os.path.join(visualize_output_path, f"lane_predicted_{step}_{i}.jpg"))
        
