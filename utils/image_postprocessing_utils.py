import cv2
import numpy as np
from typing import Tuple

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

