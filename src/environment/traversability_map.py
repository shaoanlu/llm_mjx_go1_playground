import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List
from pathlib import Path


@dataclass
class TraversabilityMapConfig:
    grid_size: Tuple[int, int] = (5, 5)
    image_size: int = 50  # 50x50 RGB image representing traversibility of a 5x5 maze
    threshold: int = 125  # White color


class TraversabilityMap:
    def __init__(self, config: TraversabilityMapConfig):
        self.config = config
        self.grid = np.empty(config.grid_size)
        self.img2grid_scale = self._calculate_scale_factor()

    def _calculate_scale_factor(self) -> int:
        """Calculate the scaling factor between image and grid spaces."""
        return self.config.image_size // self.config.grid_size[0]

    def _convert_image_to_grid_coords(self, x: int, y: int) -> Tuple[int, int]:
        """Convert image coordinates to grid coordinates.

        Args:
            x: Image x-coordinate
            y: Image y-coordinate

        Returns:
            Tuple of (grid_x, grid_y) coordinates
        """
        grid_x = y // self._scale_factor
        grid_y = (self.config.image_size - x - 1) // self._scale_factor
        return grid_x, grid_y

    def _validate_image(self, image: np.ndarray) -> None:
        """Validate image dimensions and format."""
        expected_shape = (self.config.image_size, self.config.image_size)
        if image.shape[:2] != expected_shape:
            raise ValueError(
                f"Image dimensions {image.shape[:2]} do not match expected {expected_shape}"
            )

    def load_from_image(self, image_path: str | Path) -> np.ndarray:
        """Load and process traversability grid from image.
        Thr traversity map is a binary 2D grid where 1 represents a valid position and 0 an invalid one.
        This method involes a transformation of coordinates from image to grid space and a mapping of pixel values in
        the image of size image_size to the traversability map of size grid_size.
        Image coordinates are (0,0) at top-left while grid coordinates are (0,0) at
        bottom-left, requiring y-axis inversion and appropriate scaling.

        Args:
            image_path: Path to input image file

        Returns:
            Binary grid where True represents traversable positions

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image format or dimensions are invalid
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = plt.imread(image_path)
        self._validate_image(image)
        self._image = image

        # Process each grid cell using original coordinate transformation
        for x in range(self.config.grid_size[0]):
            for y in range(self.config.grid_size[1]):
                img_x = (
                    self.config.image_size
                    - self._scale_factor * x
                    - int(np.ceil(self._scale_factor / 2))
                )
                img_y = self._scale_factor * y + int(np.floor(self._scale_factor / 2))

                color = image[img_x, img_y][0]
                self._grid[y, x] = color >= self.config.threshold

        return self._grid.copy()

    def is_valid_position(self, position: np.ndarray) -> bool:
        """Check if a position is valid and traversable in the grid."""
        try:
            x, y = position.astype(int)
            return self._grid[x, y]
        except IndexError:
            return False

    @property
    def grid(self) -> np.ndarray:
        """Return a copy of the current traversability grid."""
        return self._grid.copy()
