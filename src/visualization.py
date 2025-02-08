from typing import List

import numpy as np


def apply_fog_of_war(
    frames: List[np.ndarray],
    threshold: int = 100,
    decay: float = 0.0,
    alpha: float = 0.5,
    min_visibility: float = 0.0,
    gamma: float = 1.1,
) -> List[np.ndarray]:
    """
    Apply a fog of war effect to a sequence of image frames.

    Parameters:
        frames (list of numpy arrays): List of image frames (assumed grayscale or single-channel).
        threshold (int): Brightness threshold to reveal areas (0-255).
        decay (float): Factor to gradually fade uncovered areas (0 to 1).
        alpha (float): Blending factor between current frame and fog mask (0 to 1).
        min_visibility (float): Minimum visibility level in dark areas (0 to 1).
        gamma (float): gamma correction for fog area

    Returns:
        list of numpy arrays: Processed frames with fog of war effect applied.

    Raises:
        ValueError: If input parameters are invalid or frames are empty/malformed.
    """
    # Input validation
    if not isinstance(frames, (list, tuple)) or not frames:
        raise ValueError("frames must be a non-empty list/tuple of numpy arrays")

    # Validate parameters
    if not (0 <= threshold <= 255 and 0 <= decay <= 1 and 0 <= alpha <= 1 and 0 <= min_visibility <= 1):
        raise ValueError("Invalid parameter values. All parameters must be within their valid ranges")

    # Convert all frames to float32 at the start for consistent processing
    frames = [frame.astype(np.float32) for frame in frames]

    # Initialize masks with proper dimensions from first frame
    frame_shape = frames[0].shape
    visibility_mask = np.full(frame_shape, min_visibility * 255, dtype=np.float32)
    prev_max_pixels = np.zeros(frame_shape, dtype=np.float32)

    output_frames = []

    for frame in frames:
        # Validate frame dimensions
        if frame.shape != frame_shape:
            raise ValueError("All frames must have the same dimensions")

        # Update visibility based on threshold
        bright_areas = frame > threshold
        visibility_mask = np.maximum(
            visibility_mask * (1 - decay),  # Apply decay to existing visibility
            np.where(bright_areas, 255, min_visibility * 255),  # New visibility
        )

        # Update historical maximum brightness
        prev_max_pixels = np.maximum(prev_max_pixels, frame)

        # Calculate revealed frame with improved blending
        revealed_frame = np.clip(
            alpha * frame + (1 - alpha) * (visibility_mask / 255 * prev_max_pixels**gamma), 0, 255
        ).astype(np.uint8)

        output_frames.append(revealed_frame)

    return output_frames
