import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.environment.traversability_map import TraversabilityMap, TraversabilityMapConfig


class TestTraversabilityMapConfig(unittest.TestCase):
    """Test TraversabilityMapConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = TraversabilityMapConfig()
        self.assertEqual(config.grid_size, (5, 5))
        self.assertEqual(config.image_size, 50)
        self.assertEqual(config.threshold, 125)

    def test_custom_config(self):
        """Test custom configuration values"""
        config = TraversabilityMapConfig(grid_size=(10, 10), image_size=100, threshold=200)
        self.assertEqual(config.grid_size, (10, 10))
        self.assertEqual(config.image_size, 100)
        self.assertEqual(config.threshold, 200)


class TestTraversabilityMap(unittest.TestCase):
    """Test TraversabilityMap class functionality"""

    def setUp(self):
        """Set up test environment"""
        self.config = TraversabilityMapConfig(grid_size=(5, 5), image_size=50, threshold=125)
        self.trav_map = TraversabilityMap(self.config)

        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment"""
        # Remove temporary directory and its contents
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    def test_initialization(self):
        """Test initialization of TraversabilityMap"""
        self.assertEqual(self.trav_map.config.grid_size, (5, 5))
        self.assertEqual(self.trav_map._scale_factor, 10)
        self.assertEqual(self.trav_map._grid.shape, (5, 5))

    def test_scale_factor_calculation(self):
        """Test scale factor calculation"""
        # Test with different grid and image sizes
        configs = [
            ((5, 5), 50, 10),  # 50/5 = 10
            ((10, 10), 100, 10),  # 100/10 = 10
            ((4, 4), 40, 10),  # 40/4 = 10
        ]

        for grid_size, image_size, expected_scale in configs:
            config = TraversabilityMapConfig(grid_size=grid_size, image_size=image_size)
            trav_map = TraversabilityMap(config)
            self.assertEqual(trav_map._scale_factor, expected_scale)

    def test_coordinate_conversion(self):
        """Test conversion between image and grid coordinates"""
        test_cases = [
            ((0, 0), (0, 4)),  # Top-left corner
            ((49, 49), (4, 0)),  # Bottom-right corner
            ((25, 25), (2, 2)),  # Center
            ((0, 49), (4, 4)),  # Top-right corner
            ((49, 0), (0, 0)),  # Bottom-left corner
        ]

        for img_coords, expected_grid_coords in test_cases:
            grid_coords = self.trav_map._convert_image_to_grid_coords(*img_coords)
            self.assertEqual(
                grid_coords,
                expected_grid_coords,
                f"Failed conversion for image coords {img_coords}",
            )

    def test_image_validation(self):
        """Test image validation"""
        # Test valid image
        valid_image = np.zeros((50, 50, 3))
        self.trav_map._validate_image(valid_image)  # Should not raise

        # Test invalid image sizes
        invalid_sizes = [
            (49, 50, 3),  # Wrong width
            (50, 49, 3),  # Wrong height
            (100, 100, 3),  # Too large
            (25, 25, 3),  # Too small
        ]

        for size in invalid_sizes:
            invalid_image = np.zeros(size)
            with self.assertRaises(ValueError):
                self.trav_map._validate_image(invalid_image)

    @patch("matplotlib.pyplot.imread")
    def test_load_from_image(self, mock_imread):
        """Test loading traversability grid from image"""
        # Create mock image with known traversable and non-traversable areas
        mock_image = np.zeros((50, 50, 3))
        # Set bottom-left areas as traversable (white)
        # Image has origin at botton-left
        mock_image[mock_image.shape[0] // 2 - 1 :, : mock_image.shape[1] // 2 + 1] = 255
        mock_imread.return_value = mock_image

        # Create a temporary image file
        test_image_path = Path(self.temp_dir) / "test_image.png"
        with open(test_image_path, "w") as f:
            f.write("dummy image content")

        # Load the mock image
        grid = self.trav_map.load_from_image(test_image_path)

        # Verify grid properties
        self.assertEqual(grid.shape, (5, 5), msg=f"{grid.shape=}")
        self.assertTrue(np.all(grid[:3, :3]), msg=f"{grid=}")  # Top-left should be traversable
        self.assertFalse(
            np.all(grid[3:, 3:]), msg=f"{grid=}"
        )  # Bottom-right should be non-traversable

    def test_load_from_nonexistent_image(self):
        """Test loading from non-existent image file"""
        with self.assertRaises(FileNotFoundError):
            self.trav_map.load_from_image("nonexistent.png")

    def test_is_valid_position(self):
        """Test position validation"""
        # Setup test grid
        self.trav_map._grid = np.array(
            [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]]
        )

        # Test valid positions
        valid_positions = [
            np.array([0, 0]),  # Top-left corner
            np.array([2, 2]),  # Center
            np.array([4, 4]),  # Bottom-right corner
        ]
        for pos in valid_positions:
            self.assertTrue(self.trav_map.is_valid_position(pos), f"Position {pos} should be valid")

        # Test invalid positions
        invalid_positions = [
            np.array([0, 2]),  # Known invalid cell
            np.array([-1, 0]),  # Out of bounds negative
            np.array([5, 5]),  # Out of bounds positive
            np.array([2, 5]),  # Partial out of bounds
        ]
        for pos in invalid_positions:
            self.assertFalse(
                self.trav_map.is_valid_position(pos), f"Position {pos} should be invalid"
            )

    def test_grid_property(self):
        """Test grid property returns copy"""
        original_grid = np.ones((5, 5))
        self.trav_map._grid = original_grid.copy()

        # Get grid copy and modify it
        grid_copy = self.trav_map.grid
        grid_copy[0, 0] = 0

        # Verify original grid is unchanged
        self.assertEqual(self.trav_map._grid[0, 0], 1, msg=f"{self.trav_map._grid=}")
        np.testing.assert_array_equal(
            self.trav_map._grid, original_grid, err_msg=f"{self.trav_map._grid=}"
        )


if __name__ == "__main__":
    unittest.main()
