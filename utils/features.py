import numpy as np
from scipy.spatial import KDTree

def contrast(image):
    """Computes the standard deviation of grayscale pixel intensities (contrast)."""
    gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])  # Faster dot product
    return gray_image.std()

# Approximate RGB values for each category
color_dict = {
    "#000000": "black",  "#FFFFFF": "white", "#808080": "gray",
    "#FF0000": "red",    "#FFA500": "orange", "#FFFF00": "yellow",
    "#008000": "green",  "#0000FF": "blue", "#800080": "purple",
    "#FFC0CB": "pink",   "#A52A2A": "brown"
}

names = list(color_dict.values())
rgb_values = np.array([tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5)) for hex_color in color_dict.keys()])

# Use NumPy array directly for fast nearest neighbor search
kdt_db = KDTree(rgb_values)

def convert_rgb_to_names(rgb_array):
    """Finds the closest named color for each RGB pixel using KDTree."""
    return np.array(names)[kdt_db.query(rgb_array)[1]]

def color_distrib(image):
    """Computes the normalized color distribution of an image."""
    # Convert grayscale to RGB if needed
    if image.ndim == 2 or image.shape[-1] == 1:
        image = np.repeat(image[..., np.newaxis], 3, axis=-1)
    
    # Reshape for efficient batch processing
    pixels = image.reshape(-1, 3)
    
    # KDTree nearest neighbor search
    color_indices = kdt_db.query(pixels, k=1)[1]
    
    # Compute normalized histogram
    return np.bincount(color_indices, minlength=len(color_dict)) / pixels.shape[0]
