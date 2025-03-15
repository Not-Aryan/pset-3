import os
import sys
import env
import src.utils.utils as utils

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage


def get_3D_object_points(chessboard_size: tuple) -> np.ndarray:
    """Get the 3D object points of a chessboard"""
    columns, rows = chessboard_size
    
    object_points = np.zeros((rows * columns, 3), dtype=np.float32)
    
    for i in range(rows):
        for j in range(columns):
            idx = i * columns + j
            object_points[idx] = [j, i, 0]
    
    return object_points


def undistort_image(image: np.ndarray, 
                    camera_matrix: np.ndarray, 
                    dist_coeffs: np.ndarray) -> np.ndarray:
    """Undistort an image using camera parameters"""
    height, width = image.shape[:2]
    
    # Create meshgrid of pixel coordinates
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    
    # Extract camera parameters
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Normalize coordinates (convert from pixel to camera coordinates)
    x_norm = (x - cx) / fx
    y_norm = (y - cy) / fy
    
    # Calculate radius squared
    r2 = x_norm**2 + y_norm**2
    r4 = r2**2
    r6 = r2**3
    
    # Extract distortion coefficients
    k1, k2, p1, p2, k3 = dist_coeffs.flatten()
    
    # Apply radial distortion
    x_distorted = x_norm * (1 + k1*r2 + k2*r4 + k3*r6)
    y_distorted = y_norm * (1 + k1*r2 + k2*r4 + k3*r6)
    
    # Apply tangential distortion
    x_distorted = x_distorted + 2*p1*x_norm*y_norm + p2*(r2 + 2*x_norm**2)
    y_distorted = y_distorted + p1*(r2 + 2*y_norm**2) + 2*p2*x_norm*y_norm
    
    # Convert back to pixel coordinates
    x_pixel = x_distorted * fx + cx
    y_pixel = y_distorted * fy + cy
    
    # Create output image
    undistorted = np.zeros_like(image)
    
    # Remap each channel
    for c in range(image.shape[2]):
        undistorted[:, :, c] = ndimage.map_coordinates(image[:, :, c], 
                                                      [y_pixel.flatten(), x_pixel.flatten()],
                                                      order=1).reshape(height, width)
    
    return undistorted

def load_grayscale_image(image: np.ndarray) -> np.ndarray:
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    return gray_image


def calibrate_camera(object_points: np.ndarray, 
                     corners: np.ndarray, 
                     image_size: tuple) -> tuple:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        [object_points], [corners], image_size, None, None
    )

    return camera_matrix, dist_coeffs


def find_chessboard_corners(image: np.ndarray, chessboard_size: tuple) -> np.ndarray:
    ret, corners = cv2.findChessboardCorners(image, chessboard_size, None)

    if ret is False:
        raise ValueError("Verify correct dimensions of chessboard")
    
    return corners


def refine_corners(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)

    return corners


def draw_corners(image: np.ndarray, chessboard_size: tuple, corners: np.ndarray):
    cv2.drawChessboardCorners(image, chessboard_size, corners, True)
    plt.imshow(image)
    plt.title("Chessboard Corners")
    plt.show()


if __name__ == "__main__":
    if not os.path.exists(env.p2.output):
        os.makedirs(env.p2.output)  
    expected_camera_matrix = np.load(env.p2.expected_camera_matrix)
    expected_dist_coeffs = np.load(env.p2.expected_dist_coeffs)
    # Part 2.a
    # Calculate focal ratio: f_r = 1 / (2*tan(FoV/2))
    f_r = 1 / (2 * np.tan(np.deg2rad(45) / 2))
    
    image_info = Image.open(env.p1.chessboard_path)
    width, height = image_info.size
    
    focal_length = f_r * min(height, width)
    
    c_x = width / 2
    c_y = height / 2
    
    ideal_intrinsic_matrix = np.array([
        [focal_length, 0, c_x],
        [0, focal_length, c_y],
        [0, 0, 1]
    ])

    # Part 2.b
    chessboard_size = (14, 9)  # (columns, rows)
    
    image = utils.load_image(env.p1.chessboard_path)
    grayscale_image = load_grayscale_image(image)
    corners = find_chessboard_corners(grayscale_image, chessboard_size)
    corners = refine_corners(grayscale_image, corners)
    draw_corners(image, chessboard_size, corners)
    Image.fromarray(image).save(env.p2.chessboard_corners)

    # Part 2.c
    object_points = get_3D_object_points(chessboard_size)
    camera_matrix, dist_coeffs = calibrate_camera(object_points, corners, grayscale_image.shape[::-1])
    print("Camera Matrix:")
    print(camera_matrix)
    assert np.allclose(camera_matrix, expected_camera_matrix, atol=1e-2), f"Camera matrix does not match this expected matrix:\n{expected_camera_matrix}"
    np.save(env.p2.camera_matrix, camera_matrix)
    print("\nDistortion Coefficients:")
    print(dist_coeffs)
    assert np.allclose(dist_coeffs, expected_dist_coeffs, atol=1e-2), f"Distortion coefficients do not match these expected coefficients:\n{expected_dist_coeffs}"
    np.save(env.p2.dist_coeff, dist_coeffs)

    # Part 2.d
    undistorted_image = undistort_image(image, camera_matrix, dist_coeffs)
    plt.imshow(undistorted_image)
    plt.title("Undistorted Image")
    plt.show()
    Image.fromarray(undistorted_image).save(env.p2.undistorted_image)