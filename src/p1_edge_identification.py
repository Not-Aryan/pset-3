import os
import sys
import env
import src.utils.engine as engine

import numpy as np
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt

def find_contours(binary_image: np.ndarray, foreground: int=1) -> np.ndarray:
    """Find boundaries of objects in a binary image."""
    height, width = binary_image.shape
    visited = np.zeros_like(binary_image, dtype=bool)
    
    # 8-connected neighborhood directions (clockwise from top-left)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    
    contours = []
    
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            if binary_image[row, col] == foreground and not visited[row, col]:
                is_boundary = False
                for dr, dc in directions:
                    if binary_image[row + dr, col + dc] != foreground:
                        is_boundary = True
                        break
                
                if is_boundary:
                    contour_points = []
                    start_row, start_col = row, col
                    current_row, current_col = row, col
                    entry_direction = 0
                    
                    while True:
                        contour_points.append((current_row, current_col))
                        visited[current_row, current_col] = True
                        
                        found_next = False
                        
                        for i in range(8):
                            direction_idx = (entry_direction + i) % 8
                            dr, dc = directions[direction_idx]
                            next_row, next_col = current_row + dr, current_col + dc
                            
                            if (0 <= next_row < height and 0 <= next_col < width and 
                                binary_image[next_row, next_col] == foreground):
                                
                                is_next_boundary = False
                                for neighbor_dr, neighbor_dc in directions:
                                    neighbor_row = next_row + neighbor_dr
                                    neighbor_col = next_col + neighbor_dc
                                    if (0 <= neighbor_row < height and 0 <= neighbor_col < width and
                                        binary_image[neighbor_row, neighbor_col] != foreground):
                                        is_next_boundary = True
                                        break
                                
                                if is_next_boundary:
                                    current_row, current_col = next_row, next_col
                                    entry_direction = (direction_idx + 4) % 8
                                    found_next = True
                                    break
                        
                        if not found_next or (current_row == start_row and current_col == start_col and len(contour_points) > 1):
                            break
                    
                    contours.extend(contour_points)
    
    return np.array(contours)


class ContourImage():
    def __init__(self, image: Image):
        self.image = image
        self.binarized_image = None

    def binarize(self, threshold=128) -> None:
        """Convert the image to a binary image."""
        img_array = np.array(self.image)
        
        if len(img_array.shape) == 3:
            grayscale = np.mean(img_array, axis=2).astype(np.float32)
        else:
            grayscale = img_array.astype(np.float32)
        
        binary = (grayscale > threshold).astype(np.float32)
        self.binarized_image = binary

    def show(self) -> None:
        self.to_PIL().show()

    def fill_border(self):
        """Fill the border of the binarized image with zeros."""
        if self.binarized_image is None:
            raise ValueError("Image must be binarized before filling border")
            
        height, width = self.binarized_image.shape
        
        bordered_image = self.binarized_image.copy()
        
        bordered_image[0, :] = 0 
        bordered_image[height-1, :] = 0 
        bordered_image[:, 0] = 0
        bordered_image[:, width-1] = 0
        
        self.binarized_image = bordered_image

    def to_PIL(self) -> Image:
        color_array = np.stack([self.binarized_image]*3, axis=-1) * 255
        color_array = color_array.astype(np.uint8)
        return Image.fromarray(color_array)
    
    def prepare(self) -> np.ndarray:
        self.binarize()
        self.fill_border()
        return self.binarized_image


def find_chessboard_contours(image: Image) -> np.ndarray:
    image = ContourImage(image)
    return find_contours(image.prepare())

def draw_corners(pil_img: Image, 
                 corners: np.ndarray, 
                 color: tuple=(255, 0, 0), 
                 radius: int=5) -> Image:
    img_with_corners = pil_img.copy()
    draw = ImageDraw.Draw(img_with_corners)
    
    for (y, x) in corners:
        left_up_point = (x - radius, y - radius)
        right_down_point = (x + radius, y + radius)
        draw.ellipse([left_up_point, right_down_point], outline=color, width=2)
    
    return img_with_corners

if __name__ == "__main__":
    if not os.path.exists(env.p1.output):
        os.makedirs(env.p1.output)
    # engine.get_distorted_chessboard(env.p1.chessboard_path)

    image = Image.open(env.p1.chessboard_path)
    contours = find_chessboard_contours(image)

    result_img = draw_corners(image, contours, color=(255, 0, 0), radius=5)
    result_img.save(env.p1.contours_path)
    plt.imshow(result_img)
    plt.title("Chessboard Contours")
    plt.show()