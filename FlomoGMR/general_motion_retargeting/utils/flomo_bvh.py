"""
Loader for BVH files converted by Flomo's fbx2bvh.

These files use standard BVH naming (Hips, LeftUpLeg, LeftFoot, etc.),
positions in meters, Y-up coordinate system, and mixed channel counts
(6 for root, 3 for other joints). Rotation order is ZXY.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.xsens_vendor.BVHParser import (
    BVHParser, Anim, remove_quat_discontinuities,
)


def load_flomo_bvh(bvh_file, skip_first_frame=False):
    """
    Load a Flomo-converted Xsens BVH file and return per-frame global
    positions and orientations in MuJoCo's Z-up coordinate system.

    Returns:
        frames:       list of dicts, each {joint_name: (position_3d, quaternion_wxyz)}
        human_height: estimated height in meters (from T-pose)
        frame_time:   seconds per frame
    """
    # we use xsens_vendor's BVHParser (not lafan_vendor's extract.py) because it
    # handles mixed channel counts (6 root + 3 others). axis_order="zxy" remaps
    # BVH Y-up to MuJoCo Z-up. scale=1.0 because flomo already outputs meters.
    parser = BVHParser(axis_order="zxy", scale=1.0)
    with open(bvh_file, "r") as f:
        bvh_text = f.read()

    rotations, positions = parser.parse(bvh_text)

    # --- euler-to-quaternion conversion (bypassing BVHParser's buggy version) ---
    #
    # BVHParser.parse() returns euler angles in degrees, shuffled into
    # [X_val, Y_val, Z_val] order (BVH value names, not MJ axes).
    # BVHParser's own euler_to_quat uses extrinsic "xyz" which is wrong because
    # the axis_order="zxy" remap means MJ_X=BVH_Z, MJ_Y=BVH_X, MJ_Z=BVH_Y.
    #
    # the BVH rotation order is intrinsic ZXY: R = Rz_bvh * Rx_bvh * Ry_bvh.
    # after conjugating by the axis remap, each BVH axis maps to its MJ axis:
    #   Rz_bvh(Z_val) → Rx_mj(Z_val)
    #   Rx_bvh(X_val) → Ry_mj(X_val)
    #   Ry_bvh(Y_val) → Rz_mj(Y_val)
    # so R_mj = Rx_mj(Z_val) * Ry_mj(X_val) * Rz_mj(Y_val) = intrinsic XYZ.
    # we reorder the values to match: [Z_val, X_val, Y_val] for XYZ convention.
    nframes, njoints, _ = rotations.shape
    rot_reordered = np.stack(
        [rotations[..., 2], rotations[..., 0], rotations[..., 1]], axis=-1
    )
    flat_quats = Rot.from_euler(
        'XYZ', rot_reordered.reshape(-1, 3), degrees=True
    ).as_quat(scalar_first=True)
    quats = flat_quats.reshape(nframes, njoints, 4).astype(np.float32)
    quats = remove_quat_discontinuities(quats)

    offsets = np.array(parser.offsets)
    parents = np.array(parser.parents, dtype=int)
    anim = Anim(quats, positions, offsets, parents, parser.names)

    # forward kinematics: compose parent rotations and translate offsets into
    # world space, giving us global position + orientation per joint per frame.
    global_data = utils.quat_fk(anim.quats, anim.pos, anim.parents)
    global_positions = global_data[1]   # (frames, joints, 3)
    global_quats = global_data[0]       # (frames, joints, 4) wxyz

    # --- yaw correction ---
    #
    # BVHParser "zxy" produces MJ = (bvh_z, bvh_x, bvh_y), so MJ X=forward.
    # but geno.xml was generated with MJ = (bvh_x, -bvh_z, bvh_y), so MJ X=lateral.
    # these differ by a -90° rotation around Z. without this fix the entire motion
    # is yawed 90° relative to the skeleton.
    #
    # positions: simple matrix multiply.
    # orientations: must use CONJUGATION (R * Q * R_inv), not left-multiply (R * Q).
    # left-multiply would add a spurious -90° to every orientation. conjugation
    # re-expresses the same physical rotation in the new coordinate frame, so
    # identity stays identity (critical for T-pose).
    yaw_fix = Rot.from_euler('z', -90, degrees=True)
    yaw_fix_mat = yaw_fix.as_matrix().astype(np.float32)
    yaw_fix_quat = yaw_fix.as_quat(scalar_first=True).astype(np.float32)
    yaw_fix_inv = np.array(
        [yaw_fix_quat[0], -yaw_fix_quat[1], -yaw_fix_quat[2], -yaw_fix_quat[3]],
        dtype=np.float32,
    )

    orig_shape = global_positions.shape
    global_positions = (yaw_fix_mat @ global_positions.reshape(-1, 3).T).T.reshape(orig_shape)

    q_fix = np.broadcast_to(yaw_fix_quat[np.newaxis, np.newaxis, :], global_quats.shape)
    q_inv = np.broadcast_to(yaw_fix_inv[np.newaxis, np.newaxis, :], global_quats.shape)
    global_quats = utils.quat_mul(q_fix, utils.quat_mul(global_quats, q_inv))

    # --- pack into per-frame dicts for GMR ---
    start_frame = 1 if skip_first_frame else 0
    frames = []
    for frame in range(start_frame, global_positions.shape[0]):
        result = {}
        for i, bone in enumerate(anim.bones):
            result[bone] = (global_positions[frame, i], global_quats[frame, i])
        frames.append(result)

    # --- estimate human height from T-pose (frame 0) ---
    # GMR uses this to scale the motion so proportions match the target skeleton.
    # BVHParser appends "_end_site" to end site names, so we check both variants.
    height_frame = global_positions[0]
    head_idx = None
    left_toe_idx = None
    right_toe_idx = None
    for i, name in enumerate(anim.bones):
        if name in ("Head_end_site", "Head"):
            head_idx = i
        if name in ("LeftToeBase_end_site", "LeftToeBase"):
            left_toe_idx = i
        if name in ("RightToeBase_end_site", "RightToeBase"):
            right_toe_idx = i

    if head_idx is not None and left_toe_idx is not None and right_toe_idx is not None:
        # index [2] is the up axis (Z in MuJoCo after the axis remap)
        human_height = height_frame[head_idx][2] - min(
            height_frame[left_toe_idx][2], height_frame[right_toe_idx][2]
        )
    else:
        human_height = 1.75
        print(f"Warning: could not compute human height, using default {human_height}m")

    print(f"Loaded {len(frames)} frames, human height: {human_height:.3f}m, "
          f"frame time: {parser.frame_time:.6f}s ({1.0 / parser.frame_time:.1f} fps)")

    return frames, human_height, parser.frame_time
