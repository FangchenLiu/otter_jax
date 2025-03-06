from typing import Union, List, Tuple
import json
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import tensorflow as tf


def gram_schmidt(vectors: np.ndarray) -> np.ndarray:
    """
    Apply Gram-Schmidt process to a set of vectors
    vectors are indexed by rows

    vectors: batchsize, N, D

    return: batchsize, N, D
    """
    if len(vectors.shape) == 2:
        vectors = vectors[None]

    basis = np.zeros_like(vectors)
    basis[:, 0] = vectors[:, 0] / np.linalg.norm(vectors[:, 0], axis=-1, keepdims=True)
    for i in range(1, vectors.shape[1]):
        v = vectors[:, i]
        for j in range(i):
            v -= np.sum(v * basis[:, j], axis=-1, keepdims=True) * basis[:, j]
        basis[:, i] = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return basis


def rot_mat_to_rot_6d(rot_mat):
    """
    Convert a rotation matrix to 6d representation
    rot_mat: N, 3, 3

    return: N, 6
    """
    rot_6d = rot_mat[:, :2, :]  # N, 2, 3
    return rot_6d.reshape(-1, 6)  # N, 6


def rot_6d_to_rot_mat(rot_6d):
    """
    Convert a 6d representation to rotation matrix
    rot_6d: N, 6

    return: N, 3, 3
    """
    rot_6d = rot_6d.reshape(-1, 2, 3)
    # assert the first two vectors are orthogonal
    if not np.allclose(np.sum(rot_6d[:, 0] * rot_6d[:, 1], axis=-1), 0, atol=1e-8):
        rot_6d = gram_schmidt(rot_6d)

    rot_mat = np.zeros((rot_6d.shape[0], 3, 3))
    rot_mat[:, :2, :] = rot_6d
    rot_mat[:, 2, :] = np.cross(rot_6d[:, 0], rot_6d[:, 1])
    return rot_mat


def euler_to_rot_6d(euler: np.ndarray, format="XYZ") -> np.ndarray:
    """
    Convert euler angles to 6d representation
    euler: N, 3
    """
    rot_mat = Rotation.from_euler(format, euler, degrees=False).as_matrix()
    return rot_mat_to_rot_6d(rot_mat)


def action_delta2abs(delta_action, proprio):
    """'
    delta_action: Seq, 16, 10
    """
    flatten_delta_action = delta_action.copy().reshape(-1, 10)
    delta_action_rot = rot_6d_to_rot_mat(flatten_delta_action[:, 3:9])
    delta_action_mat = np.zeros((flatten_delta_action.shape[0], 4, 4))
    delta_action_mat[:, :3, :3] = delta_action_rot
    delta_action_mat[:, :3, 3] = flatten_delta_action[:, :3]
    delta_action_mat[:, 3, 3] = 1

    proprio_rot = rot_6d_to_rot_mat(proprio.copy()[:, 3:9])
    proprio_mat = np.zeros((proprio.shape[0], 4, 4))
    proprio_mat[:, :3, :3] = proprio_rot
    proprio_mat[:, :3, 3] = proprio[:, :3]
    proprio_mat[:, 3, 3] = 1
    proprio_mat_broadcast = np.repeat(proprio_mat, delta_action.shape[1], 0)

    abs_action_mat = proprio_mat_broadcast @ delta_action_mat
    return abs_action_mat.reshape(delta_action.shape[0], delta_action.shape[1], 4, 4)


def project2cam(obs, coor_w, extrinsics, intrinsics, vis_steps=4):
    # Points to project (origin and axes endpoints in homogeneous coordinates)

    coor_w_flatten = coor_w.reshape(-1, 4, 4)
    T_cx = np.linalg.inv(extrinsics) @ coor_w_flatten
    pts = np.einsum(
        "ij,kj->ik", intrinsics, T_cx[:, :3, :].transpose(0, 2, 1).reshape(-1, 3)
    ).T
    pts_normalized = pts[:, :2] / pts[:, 2:]
    pts_normalized[:, 0] = pts_normalized[:, 0] / 640
    pts_normalized[:, 1] = pts_normalized[:, 1] / 480
    pts_normalized = pts_normalized.reshape(
        coor_w.shape[0], coor_w.shape[1], 4, 2
    )  # for x,y,z axis and origin

    H, W = obs[0].shape[:2]
    pts_resize = pts_normalized * H
    xy_axis = pts_resize[:, :, :2, :] - pts_resize[:, :, -1:, :]
    xy_axis = (
        10 * xy_axis / np.linalg.norm(xy_axis, axis=-1, keepdims=True)
    )  # 10 pixels vector
    xy_coor = pts_resize[:, :, -1:, :] + xy_axis

    # Plot the projected points
    traj_len, action_horizon = coor_w.shape[0], coor_w.shape[1]
    norm = mcolors.Normalize(vmin=0, vmax=action_horizon - 1)
    cmap = plt.get_cmap("viridis")
    colors = cmap(norm(range(action_horizon)))[::-1]
    xcmap = plt.get_cmap("Reds")
    xcolors = xcmap(norm(range(action_horizon)))
    ycmap = plt.get_cmap("Greens")
    ycolors = ycmap(norm(range(action_horizon)))

    fig, axes = plt.subplots(2, vis_steps // 2)
    idxs = np.random.choice(pts_resize.shape[0], vis_steps, replace=False)
    for j, idx in enumerate(idxs):
        axes.flatten()[j].imshow(obs[idx])
        # axes.flatten()[j].scatter(xy_coor[idx, :, 0, 0] , xy_coor[idx, :,0, 1],color=xcolors, s=4)
        # axes.flatten()[j].scatter(xy_coor[idx, :, 1, 0] , xy_coor[idx, :,1, 1],color=ycolors, s=4)
        axes.flatten()[j].scatter(
            pts_resize[idx, :, -1, 0], pts_resize[idx, :, -1, 1], color=colors, s=6
        )
        axes.flatten()[j].axis("off")

    # for i,pt in enumerate(pts_resize):
    #     axes.flatten()[i].imshow(obs[i])
    #     axes.flatten()[i].scatter(pt[:,-1, 0], pt[:,-1, 1],color=colors)

    #     axes.flatten()[i].scatter(xy_coor[i, :, 0, 0] , xy_coor[i, :,0, 1],color=xcolors)
    #     axes.flatten()[i].scatter(xy_coor[i, :, 1, 0] , xy_coor[i, :,1, 1],color=ycolors)
    plt.tight_layout()
    plt.close("all")
    return fig


def euler_XYZ_to_matrix(euler_angles_rad):
    # input shape: (batch_size, 3)
    # Extract individual angles
    alpha, beta, gamma = tf.unstack(euler_angles_rad, axis=-1)

    # import pdb;pdb.set_trace()
    # Compute rotation matrix around X-axis
    R_x = tf.stack(
        [
            [tf.ones_like(alpha), tf.zeros_like(alpha), tf.zeros_like(alpha)],
            [tf.zeros_like(alpha), tf.cos(alpha), -tf.sin(alpha)],
            [tf.zeros_like(alpha), tf.sin(alpha), tf.cos(alpha)],
        ],
        axis=0,
    )

    # Compute rotation matrix around Y-axis
    R_y = tf.stack(
        [
            [tf.cos(beta), tf.zeros_like(beta), tf.sin(beta)],
            [tf.zeros_like(beta), tf.ones_like(beta), tf.zeros_like(beta)],
            [-tf.sin(beta), tf.zeros_like(beta), tf.cos(beta)],
        ],
        axis=0,
    )

    # Compute rotation matrix around Z-axis
    R_z = tf.stack(
        [
            [tf.cos(gamma), -tf.sin(gamma), tf.zeros_like(gamma)],
            [tf.sin(gamma), tf.cos(gamma), tf.zeros_like(gamma)],
            [tf.zeros_like(gamma), tf.zeros_like(gamma), tf.ones_like(gamma)],
        ],
        axis=0,
    )

    R_x = tf.transpose(R_x, perm=[2, 0, 1])
    R_y = tf.transpose(R_y, perm=[2, 0, 1])
    R_z = tf.transpose(R_z, perm=[2, 0, 1])

    # Compute the final rotation matrix
    R = tf.linalg.matmul(R_x, tf.linalg.matmul(R_y, R_z))
    return R


def quaternion_to_matrix(quaternion) -> tf.Tensor:
    """Convert a quaternion to a rotation matrix.

    Note:
    In the following, A1 to An are optional batch dimensions.

    Args:
    quaternion: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
        represents a normalized quaternion.
    name: A name for this op that defaults to
        "rotation_matrix_3d_from_quaternion".

    Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represent a 3d rotation matrix.

    Raises:
    ValueError: If the shape of `quaternion` is not supported.
    """
    quaternion = tf.convert_to_tensor(value=quaternion)
    quaternion = quaternion / tf.norm(tensor=quaternion, axis=-1, keepdims=True)

    x, y, z, w = tf.unstack(quaternion, axis=-1)
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    matrix = tf.stack(
        (
            1.0 - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            1.0 - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            1.0 - (txx + tyy),
        ),
        axis=-1,
    )  # pyformat: disable
    output_shape = tf.concat((tf.shape(input=quaternion)[:-1], (3, 3)), axis=-1)
    return tf.reshape(matrix, shape=output_shape)


_MANO_MCP_IND = [0, 1, 5, 9, 13, 17]  # the first is the wrist
_MANO_THUMB_TIP_IND = 4
_MANO_INDEX_TIP_IND = 8


def estimate_plane(ycb_pose_array):
    """
    ycb_pose_array: np.array of shape (num_objects, 3) of translations
    Returns the quaternion and centroid of the plane passing through all YCB Objects.
    """
    CoG_pos = ycb_pose_array
    # print("CoG_pos: ", CoG_pos)

    # Create the design matrix

    A = np.column_stack((CoG_pos[:, 0], CoG_pos[:, 1], np.ones(CoG_pos.shape[0])))

    # Solve the least square problem to find the coeff of the plane
    V = np.linalg.lstsq(A, CoG_pos[:, 2], rcond=None)[0]

    # Compute the normal of the plane
    normal = np.array([V[0], V[1], -1])
    normal = normal / np.linalg.norm(normal)

    # Compute the quaternion
    angle = np.arccos(np.dot(normal, np.array([0, 0, 1])))

    # Compute axis angle representation
    w = np.cos(angle / 2)
    axis = np.sin(angle / 2) * normal
    x, y, z = axis[0], axis[1], axis[2]

    # Compute the centroid of the YCB Objects projected on the plane
    centroid = np.mean(CoG_pos, axis=0)
    quat = np.array([x, y, z, w], dtype=np.float32)

    # Translate the centroid 0.2m below the YCB objects
    # centroid += 0.4 * normal
    rot = Rotation.from_quat(quat).as_matrix()
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = centroid

    return mat


def approx_parrallel_jaw_gripper(hand_joint_3d):
    """
    use the thumb and index tip to estimate the gripper distance
    """
    thumb_tip_pos = hand_joint_3d[_MANO_THUMB_TIP_IND]
    index_tip_pos = hand_joint_3d[_MANO_INDEX_TIP_IND]
    distance = np.linalg.norm(thumb_tip_pos - index_tip_pos)
    return distance


def get_hand_ori(hand_joint_3d):
    """
    estimate the hand orientation from the 3d joints in the world frame
    this function fit a plane for the palm using the mcp translation
    """
    # extract all the mcp and wrist
    mcp_points = hand_joint_3d[_MANO_MCP_IND]  # shape (6,3)
    palm_T = estimate_plane(mcp_points)
    return palm_T
