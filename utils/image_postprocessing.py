"""
Image and tensor post-processing utilities for lane detection and direction inference.

This module provides functions for post-processing model outputs, including:
- Tensor normalization and clipping operations
- Direction image encoding/decoding  
- Segmentation post-processing
- Output visualization utilities
"""

import numpy as np
from typing import Tuple, Optional
import cv2


def normalize_tensor_to_image_range(
    tensor: np.ndarray, 
    input_range: Tuple[float, float] = (-1.0, 1.0),
    output_range: Tuple[int, int] = (0, 255)
) -> np.ndarray:
    """
    Normalize tensor values from input range to output image range.
    
    Args:
        tensor: Input tensor to normalize
        input_range: Expected range of input values (min, max)
        output_range: Target range for output values (min, max)
        
    Returns:
        Normalized tensor as uint8 array
    """
    input_min, input_max = input_range
    output_min, output_max = output_range
    

    clipped = np.clip(tensor, input_min, input_max)
    

    normalized = (clipped - input_min) / (input_max - input_min)
    

    scaled = normalized * (output_max - output_min) + output_min
    
    return scaled.astype(np.uint8)


def encode_direction_vectors_to_image(
    direction_vectors: np.ndarray,
    background_value: int = 127
) -> np.ndarray:
    """
    Encode direction vectors as color image for visualization and storage.
    
    Args:
        direction_vectors: Array of shape (H, W, 2) with direction vectors in range [-1, 1]
        background_value: RGB value for background pixels
        
    Returns:
        RGB image of shape (H, W, 3) with encoded directions
    """
    height, width = direction_vectors.shape[:2]
    direction_image = np.full((height, width, 3), background_value, dtype=np.uint8)
    

    direction_image[:, :, 0] = normalize_tensor_to_image_range(
        direction_vectors[:, :, 0], (-1.0, 1.0), (0, 255)
    )
    direction_image[:, :, 1] = normalize_tensor_to_image_range(
        direction_vectors[:, :, 1], (-1.0, 1.0), (0, 255)
    )
    
    return direction_image


def decode_direction_vectors_from_image(
    direction_image: np.ndarray,
    background_value: int = 127
) -> np.ndarray:
    """
    Decode direction vectors from encoded color image.
    
    Args:
        direction_image: RGB image of shape (H, W, 3) with encoded directions
        background_value: RGB value that represents zero direction
        
    Returns:
        Direction vectors of shape (H, W, 2) in range [-1, 1]
    """
    height, width = direction_image.shape[:2]
    direction_vectors = np.zeros((height, width, 2), dtype=np.float32)
    

    direction_vectors[:, :, 0] = (direction_image[:, :, 0].astype(np.float32) - background_value) / background_value
    direction_vectors[:, :, 1] = (direction_image[:, :, 1].astype(np.float32) - background_value) / background_value
    

    direction_vectors = np.clip(direction_vectors, -1.0, 1.0)
    
    return direction_vectors


def post_process_lane_segmentation(
    raw_segmentation: np.ndarray,
    confidence_threshold: float = 0.5,
    apply_morphological_thinning: bool = True
) -> np.ndarray:
    """
    Post-process raw lane segmentation output.
    
    Args:
        raw_segmentation: Raw segmentation logits or probabilities
        confidence_threshold: Threshold for binarization
        apply_morphological_thinning: Whether to apply morphological thinning
        
    Returns:
        Binary segmentation mask
    """

    if raw_segmentation.max() > 1.0:
        segmentation = raw_segmentation / 255.0
    else:
        segmentation = raw_segmentation
    
    # Binarize
    binary_mask = segmentation > confidence_threshold
    

    if apply_morphological_thinning:
        try:
            from skimage import morphology
            binary_mask = morphology.thin(binary_mask)
        except ImportError:

            binary_mask = cv2_morphological_thinning(binary_mask.astype(np.uint8))
    
    return binary_mask.astype(np.uint8)


def cv2_morphological_thinning(binary_image: np.ndarray) -> np.ndarray:
    """
    Apply morphological thinning using OpenCV operations.
    
    Args:
        binary_image: Binary input image (0 or 255)
        
    Returns:
        Thinned binary image
    """
    # Convert to proper format
    img = binary_image.copy()
    if img.max() == 1:
        img = img * 255
    

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    # Iterative thinning
    prev = None
    while not np.array_equal(img, prev):
        prev = img.copy()
        

        eroded = cv2.erode(img, kernel, iterations=1)
        reconstructed = cv2.dilate(eroded, kernel, iterations=1)
        

        img = cv2.subtract(img, reconstructed)
    
    return (img > 0).astype(np.uint8)


def post_process_model_output(
    lane_output: np.ndarray,
    direction_output: np.ndarray,
    lane_threshold: float = 0.5,
    apply_thinning: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete post-processing pipeline for lane detection model output.
    
    Args:
        lane_output: Raw lane segmentation output
        direction_output: Raw direction vector output  
        lane_threshold: Threshold for lane binarization
        apply_thinning: Whether to apply morphological thinning to lanes
        
    Returns:
        Tuple of (processed_lanes, processed_directions)
    """
    # Process lane segmentation
    processed_lanes = post_process_lane_segmentation(
        lane_output, 
        confidence_threshold=lane_threshold,
        apply_morphological_thinning=apply_thinning
    )
    

    processed_directions = np.clip(direction_output, -1.0, 1.0)
    
    return processed_lanes, processed_directions


def apply_region_mask(
    lane_output: np.ndarray,
    direction_output: np.ndarray, 
    region_mask: np.ndarray,
    mask_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply region mask to filter outputs to valid regions only.
    
    Args:
        lane_output: Lane segmentation output
        direction_output: Direction vector output
        region_mask: Binary mask indicating valid regions
        mask_threshold: Threshold for region mask binarization
        
    Returns:
        Tuple of (masked_lanes, masked_directions)
    """

    if region_mask.max() > 1.0:
        binary_mask = (region_mask / 255.0) > mask_threshold
    else:
        binary_mask = region_mask > mask_threshold
    

    masked_lanes = lane_output * binary_mask
    

    if len(direction_output.shape) == 3 and direction_output.shape[2] == 2:
        masked_directions = direction_output * binary_mask[:, :, np.newaxis]
    else:
        masked_directions = direction_output * binary_mask
    
    return masked_lanes, masked_directions


def create_visualization_output(
    input_image: np.ndarray,
    lane_output: np.ndarray,
    direction_output: np.ndarray,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Create visualization combining input image with lane and direction outputs.
    
    Args:
        input_image: Original input image
        lane_output: Processed lane segmentation
        direction_output: Processed direction vectors
        save_path: Optional path to save visualization
        
    Returns:
        Combined visualization image
    """

    if input_image.max() <= 1.0:
        vis_input = (input_image * 255).astype(np.uint8)
    else:
        vis_input = input_image.astype(np.uint8)
    
    # Handle single channel input
    if len(vis_input.shape) == 2:
        vis_input = cv2.cvtColor(vis_input, cv2.COLOR_GRAY2RGB)
    elif vis_input.shape[2] == 1:
        vis_input = cv2.cvtColor(vis_input.squeeze(), cv2.COLOR_GRAY2RGB)
    
    # Create lane overlay
    lane_overlay = np.zeros_like(vis_input)
    if lane_output.max() <= 1.0:
        lane_mask = (lane_output * 255).astype(np.uint8)
    else:
        lane_mask = lane_output.astype(np.uint8)
    
    lane_overlay[:, :, 1] = lane_mask  # Green channel for lanes
    
    # Create direction overlay
    direction_overlay = encode_direction_vectors_to_image(direction_output)
    
    # Combine visualizations
    combined = cv2.addWeighted(vis_input, 0.6, lane_overlay, 0.4, 0)
    combined = cv2.addWeighted(combined, 0.7, direction_overlay, 0.3, 0)
    
    if save_path:
        cv2.imwrite(save_path, combined)
    
    return combined


def normalize_image_for_model_input(
    image: np.ndarray,
    mean: float = 0.5,
    std: float = 0.5,
    input_range: Tuple[int, int] = (0, 255)
) -> np.ndarray:
    """
    Normalize image for model input.
    
    Args:
        image: Input image 
        mean: Mean value for normalization
        std: Standard deviation for normalization
        input_range: Expected input value range
        
    Returns:
        Normalized image ready for model input
    """
    # Convert to float
    normalized = image.astype(np.float32)
    
    # Scale to [0, 1]
    input_min, input_max = input_range
    normalized = (normalized - input_min) / (input_max - input_min)
    
    # Apply mean/std normalization
    normalized = (normalized - mean) / std
    
    return normalized


def denormalize_model_output(
    output: np.ndarray,
    mean: float = 0.5, 
    std: float = 0.5,
    output_range: Tuple[int, int] = (0, 255)
) -> np.ndarray:
    """
    Denormalize model output back to image range.
    
    Args:
        output: Model output tensor
        mean: Mean value used in normalization
        std: Standard deviation used in normalization  
        output_range: Target output value range
        
    Returns:
        Denormalized image
    """
    # Reverse mean/std normalization
    denormalized = output * std + mean
    
    # Scale to output range
    output_min, output_max = output_range
    denormalized = denormalized * (output_max - output_min) + output_min
    
    # Clip and convert to proper type
    denormalized = np.clip(denormalized, output_min, output_max)
    
    if output_max <= 255:
        return denormalized.astype(np.uint8)
    else:
        return denormalized.astype(np.float32)


def denormalize_image_for_display(normalized_image):
    """
    Denormalizes an image from model format back to display format (0-255).
    
    Args:
        normalized_image (np.ndarray): Normalized image with values in [-0.5, 0.5]
        
    Returns:
        np.ndarray: Denormalized image with values in [0, 255]
    """
    return (normalized_image + 0.5) * 255


def apply_softmax_to_logits(logits):
    """
    Applies softmax to logits to get probabilities.
    
    Args:
        logits (np.ndarray): Raw logits of shape [..., num_classes]
        
    Returns:
        np.ndarray: Softmax probabilities of same shape
    """
    # Apply softmax along the last dimension
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))  # Numerical stability
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def post_process_model_output(
    outputs, 
    threshold=64, 
    morphology_size=(6, 6), 
    apply_thinning=True, 
    save_debug_image=None
):
    """
    Post-process model output from lane and direction extraction.
    
    Args:
        outputs (np.ndarray): Model outputs of shape [B, 4, H, W] where first 2 channels are lane, last 2 are direction
        threshold (int): Threshold for lane binarization
        morphology_size (tuple): Size for morphological operations
        apply_thinning (bool): Whether to apply morphological thinning
        save_debug_image (str, optional): Path to save debug image
        
    Returns:
        tuple: (lane_predicted_image, direction_predicted_image)
    """
    import scipy.ndimage
    from skimage import morphology
    from PIL import Image
    import einops
    

    lane_predicted = outputs[:, 0:2, :, :]  # [B, 2, H, W]
    direction_predicted = outputs[:, 2:4, :, :] # [B, 2, H, W]


    lane_predicted = einops.rearrange(lane_predicted, 'b c h w -> b h w c')
    direction_predicted = einops.rearrange(direction_predicted, 'b c h w -> b h w c')


    lane_predicted_image = np.zeros((outputs.shape[2], outputs.shape[3]))
    lane_predicted_image[:, :] = np.clip(lane_predicted[0,:,:,0], 0, 1) * 255
    

    direction_predicted_image = np.zeros((outputs.shape[2], outputs.shape[3], 2))
    direction_predicted_image[:,:,0] = np.clip(direction_predicted[0,:,:,0],-1,1) * 127 + 127
    direction_predicted_image[:,:,1] = np.clip(direction_predicted[0,:,:,1],-1,1) * 127 + 127
    

    if save_debug_image:
        Image.fromarray(lane_predicted_image.astype(np.uint8)).save(save_debug_image)


    lane_predicted_image = scipy.ndimage.grey_closing(lane_predicted_image, size=morphology_size)
    lane_predicted_image = lane_predicted_image >= threshold
    
    if apply_thinning:
        lane_predicted_image = morphology.thin(lane_predicted_image)

    return lane_predicted_image, direction_predicted_image



def clip_and_normalize_directions(directions, output_range=(0, 255)):
    """Legacy function for direction normalization."""
    return normalize_tensor_to_image_range(directions, (-1.0, 1.0), output_range)


def process_segmentation_output(segmentation, threshold=0.5):
    """Legacy function for segmentation processing."""
    return post_process_lane_segmentation(segmentation, threshold)
