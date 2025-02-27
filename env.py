from pathlib import Path
import os

PROJECT_DIR = Path(__file__).resolve().parent
project_data = PROJECT_DIR / 'data'
src = PROJECT_DIR / 'src'

class p1:
    data = project_data / 'p1_edge_identification'
    chessboard_path = data / 'chessboard.png'
    contours_path = data / 'contours.png'

class p2:
    data = project_data / 'p2_calibrate_camera'
    chessboard_corners = data / 'corners.png'
    camera_matrix = data / 'camera_matrix.npy'
    dist_coeff = data / 'dist_coeff.npy'
    undistorted_image = data / 'undistorted_image.png'

class p3:
    data = project_data / 'p3_fundamental_matrix'
    test_obj = data / 'test.obj'
    test_texture = data / 'test.mtl'
    im1 = data / 'im1.png'
    im2 = data / 'im2.png'
    pts_1 = data / 'pts_1.txt'
    pts_2 = data / 'pts_2.txt'

    const_im1 = data / 'const_im1.png'
    const_im2 = data / 'const_im2.png'

    lls_img = data / 'lls_img.png'
    norm_img = data / 'norm_img.png'


class p4:
    data = project_data / 'p4_image_rectification'
    aligned_epipolar = data / 'aligned_epipolar.png'
    cv_matches = data / 'cv_matches.png'

class p5:
    data = project_data / 'p5_3D_reconstruction'
    arc_obj = data / 'arc_de_triomphe' / 'model.obj'
    arc_texture = data / 'arc_de_triomphe' / 'model.mtl'
    chessboard = data / 'chessboard.png'
    raw_images = data / 'raw_images'
    statue_images = data / 'statue'
    undistorted_images = data / 'undistorted_images'
    rotation_matrix = data / 'rotation_matrix.npy'
    translation_matrix = data / 'translation_matrix.npy'