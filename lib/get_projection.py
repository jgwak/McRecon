import lib.binvox_rw as binvox_rw
from math import radians
from mathutils import Matrix, Vector, Euler
import autograd.numpy as np
import itertools

# Default parameters of the blender renderer.
F_MM = 35.  # Focal length
SENSOR_SIZE_MM = 32.

# Blender renderer setup.
PIXEL_ASPECT_RATIO = 1.  # pixel_aspect_x / pixel_aspect_y
RESOLUTION_PCT = 100
SKEW = 0.
CAM_MAX_DIST = 1.75
CAM_ROT = Matrix(((1.910685676922942e-15, 4.371138828673793e-08, 1.0),
                   (1.0, -4.371138828673793e-08, -0.0),
                   (4.371138828673793e-08, 1.0, -4.371138828673793e-08)))

# Constant parameters for the renderer.
IMG_W = 127 + 10  # Rendering image size. Network input size + cropping margin.
IMG_H = 127 + 10


def readBinvoxParams(binbox_f):
    with open(binbox_f, 'rb') as f:
        voxel = binvox_rw.read_as_3d_array(f)
        voxel_d = voxel.dims[0]
        voxel_t = voxel.translate
        voxel_s = voxel.scale
    return (voxel_d, *voxel_t, voxel_s)


def getBinvoxProj(voxel_d, voxel_t_x, voxel_t_y, voxel_t_z, voxel_s):
    """Calculate 4x4 projection matrix from voxel to obj coordinate"""
    # Calculate rotation and translation matrices.
    # Step 1: Voxel coordinate to binvox coordinate.
    S_vox2bin = Matrix.Scale(1 / (voxel_d - 1), 4)

    # Step 2: Binvox coordinate to obj coordinate.
    voxel_t = min(voxel_t_x, voxel_t_y, voxel_t_z)
    RST_bin2obj = (Matrix.Rotation(radians(90), 4, 'X') *
                   Matrix.Translation([voxel_t] * 3) *
                   Matrix.Scale(voxel_s, 4))

    return np.matrix(RST_bin2obj * S_vox2bin)


def getBlenderProj(az, el, distance_ratio, img_w=IMG_W, img_h=IMG_H):
    """Calculate 4x3 3D to 2D projection matrix given viewpoint parameters."""

    # Calculate intrinsic matrix.
    scale = RESOLUTION_PCT / 100
    f_u = F_MM * img_w * scale / SENSOR_SIZE_MM
    f_v = F_MM * img_h * scale * PIXEL_ASPECT_RATIO / SENSOR_SIZE_MM
    u_0 = img_w * scale / 2
    v_0 = img_h * scale / 2
    K = np.matrix(((f_u, SKEW, u_0), (0, f_v, v_0), (0, 0, 1)))

    # Calculate rotation and translation matrices.
    # Step 1: World coordinate to object coordinate.
    sa = np.sin(np.radians(-az))
    ca = np.cos(np.radians(-az))
    se = np.sin(np.radians(-el))
    ce = np.cos(np.radians(-el))
    R_world2obj = np.transpose(np.matrix(((ca * ce, -sa, ca * se),
                                          (sa * ce, ca, sa * se),
                                          (-se, 0, ce))))

    # Step 2: Object coordinate to camera coordinate.
    R_obj2cam = np.transpose(np.matrix(CAM_ROT))
    R_world2cam = R_obj2cam * R_world2obj
    cam_location = np.transpose(np.matrix((distance_ratio * CAM_MAX_DIST,
                                           0,
                                           0)))
    T_world2cam = -1 * R_obj2cam * cam_location

    # Step 3: Fix blender camera's y and z axis direction.
    R_camfix = np.matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    R_world2cam = R_camfix * R_world2cam
    T_world2cam = R_camfix * T_world2cam

    RT = np.hstack((R_world2cam, T_world2cam))

    return K, RT


def getRay(voxel_d, voxel_t_x, voxel_t_y, voxel_t_z, voxel_s, img_w=IMG_W,
           img_h=IMG_H):
    """Convert camera and binvox parameter to 3D ray."""
    def _getRay(cam_param):
        """Inner function with camera parameter to calculate derivative against.
        """
        az, el, distance_ratio = cam_param
        K, RT = getBlenderProj(az, el, distance_ratio, img_w=img_w, img_h=img_h)
        W2B = getBinvoxProj(voxel_d, voxel_t_x, voxel_t_y, voxel_t_z, voxel_s)
        # Calculate camera location from camera matrix.
        invrot = RT[:, :3].transpose()
        invloc = - np.matrix(invrot) * np.matrix(RT[:, 3]).T
        camloc = np.matrix((*np.array(invloc.T)[0], 1)) * np.linalg.inv(W2B).T
        camloc = camloc[0, :3] / camloc[0, 3]
        # Calculate direction vector of ray for each pixel of the image.
        pixloc = np.matrix(list(itertools.product(range(img_h),
                                                  range(img_w),
                                                  (1,))))
        pixloc = pixloc * (np.linalg.inv(W2B) *
                           np.linalg.pinv(RT) *
                           np.linalg.inv(K)).T
        pixloc = pixloc[:, :3] / pixloc[:, 3]
        raydir = camloc - pixloc
        return np.array(camloc)[0].astype(np.float32), raydir.astype(np.float32)
    return _getRay
