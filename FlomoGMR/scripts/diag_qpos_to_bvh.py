"""
Diagnostic: convert MuJoCo qpos to BVH using world-space FK instead of raw qpos quaternions.

The problem: MuJoCo ball joint quaternions are local rotations in the body's REST frame,
which for the Geno skeleton has non-world-aligned axes (e.g. arm Y points along the bone).
BVH expects euler angles relative to world-aligned parent frames.

The fix: use mj_forward to get world-space body orientations, then convert to local
BVH rotations by inverting the parent's world orientation.
"""
import argparse
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.flomo_bvh import load_flomo_bvh


def bvh_save(filename, data):
    channelmap_inv = {'x': 'Xrotation', 'y': 'Yrotation', 'z': 'Zrotation'}
    rots = data['rotations']
    poss = data['positions']
    offsets = data['offsets']
    parents = data['parents']
    names = data.get('names', ["joint_" + str(i) for i in range(len(parents))])
    order = data.get('order', 'zyx')
    frametime = data.get('frametime', 1.0/30.0)

    with open(filename, 'w') as f:
        t = ""
        f.write("%sHIERARCHY\n" % t)
        f.write("%sROOT %s\n" % (t, names[0]))
        f.write("%s{\n" % t)
        t += '\t'
        f.write("%sOFFSET %.9g %.9g %.9g\n" % (t, offsets[0][0], offsets[0][1], offsets[0][2]))
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" %
            (t, channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))

        jseq = [0]

        def save_joint(f, offsets, order, parents, names, t, i, jseq):
            jseq.append(i)
            f.write("%sJOINT %s\n" % (t, names[i]))
            f.write("%s{\n" % t)
            t += '\t'
            f.write("%sOFFSET %.9g %.9g %.9g\n" % (t, offsets[i][0], offsets[i][1], offsets[i][2]))
            f.write("%sCHANNELS 3 %s %s %s \n" % (t,
                channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))
            end_site = True
            for j in range(len(parents)):
                if parents[j] == i:
                    t, jseq = save_joint(f, offsets, order, parents, names, t, j, jseq)
                    end_site = False
            if end_site:
                f.write("%sEnd Site\n" % t)
                f.write("%s{\n" % t)
                t += '\t'
                f.write("%sOFFSET %.9g %.9g %.9g\n" % (t, 0.0, 0.0, 0.0))
                t = t[:-1]
                f.write("%s}\n" % t)
            t = t[:-1]
            f.write("%s}\n" % t)
            return t, jseq

        for i in range(len(parents)):
            if parents[i] == 0:
                t, jseq = save_joint(f, offsets, order, parents, names, t, i, jseq)

        t = t[:-1]
        f.write("%s}\n" % t)
        f.write("MOTION\n")
        f.write("Frames: %i\n" % len(rots))
        f.write("Frame Time: %f\n" % frametime)

        for i in range(rots.shape[0]):
            for j in jseq:
                if j == 0:
                    f.write("%.9g %.9g %.9g %.9g %.9g %.9g " % (
                        poss[i, j, 0], poss[i, j, 1], poss[i, j, 2],
                        rots[i, j, 0], rots[i, j, 1], rots[i, j, 2]))
                else:
                    f.write("%.9g %.9g %.9g " % (rots[i, j, 0], rots[i, j, 1], rots[i, j, 2]))
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh_file", required=True, type=str)
    parser.add_argument("--output_bvh", required=True, type=str)
    parser.add_argument("--skip_tpose", action="store_true")
    args = parser.parse_args()

    frames, human_height, frame_time = load_flomo_bvh(
        args.bvh_file, skip_first_frame=args.skip_tpose
    )

    retargeter = GMR(src_human="bvh_flomo", tgt_robot="geno", actual_human_height=human_height)
    model = retargeter.model
    data = mj.MjData(model)

    qpos_list = []
    print(f"Retargeting {len(frames)} frames...")
    for i, frame_data in enumerate(frames):
        qpos_list.append(retargeter.retarget(frame_data).copy())
        if (i + 1) % 100 == 0:
            print(f"  frame {i + 1}/{len(frames)}")

    num_bodies = model.nbody  # includes world body at index 0
    num_joints = num_bodies - 1  # our BVH joints (body 1..nbody-1)
    joint_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) for i in range(1, num_bodies)]
    parents = [model.body_parentid[i] - 1 if model.body_parentid[i] > 0 else -1
               for i in range(1, num_bodies)]

    # offsets: MuJoCo Z-up -> BVH Y-up, meters -> cm
    offsets = []
    for i in range(1, num_bodies):
        mj_pos = model.body_pos[i]
        offsets.append(np.array([mj_pos[0] * 100, mj_pos[2] * 100, -mj_pos[1] * 100]))

    qpos_array = np.array(qpos_list)
    bvh_poss = np.zeros((qpos_array.shape[0], num_joints, 3))
    bvh_rots = np.zeros((qpos_array.shape[0], num_joints, 3))

    # the key idea: use MuJoCo FK to get world-space body orientations (xquat),
    # then derive local BVH rotations from those.
    # BVH local rotation = inverse(parent_world_rot_bvh) * child_world_rot_bvh
    # where world rotations are converted from MuJoCo Z-up to BVH Y-up.

    # coordinate change rotation: MuJoCo Z-up -> BVH Y-up
    # MJ(x,y,z) -> BVH(x,z,-y), which is a -90deg rotation around X
    # as a scipy rotation:
    coord_rot = R.from_matrix([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0]
    ])

    for f in range(qpos_array.shape[0]):
        data.qpos[:] = qpos_array[f]
        mj.mj_forward(model, data)

        # xquat[i] = world orientation of body i, in MuJoCo wxyz format
        # xpos[i] = world position of body i, in MuJoCo Z-up meters

        # convert world orientations to BVH Y-up frame
        # R_bvh = coord_rot * R_mj
        world_rots_bvh = []
        for i in range(num_bodies):
            mj_quat_wxyz = data.xquat[i]
            r_mj = R.from_quat([mj_quat_wxyz[1], mj_quat_wxyz[2], mj_quat_wxyz[3], mj_quat_wxyz[0]])
            r_bvh = coord_rot * r_mj
            world_rots_bvh.append(r_bvh)

        # root position: world body index is 0, root body is index 1
        root_xpos = data.xpos[1]  # MuJoCo world position in meters
        bvh_poss[f, 0] = [root_xpos[0] * 100, root_xpos[2] * 100, -root_xpos[1] * 100]

        # root rotation: BVH root local rot = world rot in BVH frame
        # (parent is world, which has identity orientation in BVH frame)
        bvh_rots[f, 0] = world_rots_bvh[1].as_euler('ZYX', degrees=True)

        # other joints: local_rot = inv(parent_world_rot) * child_world_rot
        for j in range(1, num_joints):
            body_idx = j + 1  # MuJoCo body index (offset by 1 for world body)
            parent_body_idx = model.body_parentid[body_idx]
            local_rot = world_rots_bvh[parent_body_idx].inv() * world_rots_bvh[body_idx]
            bvh_rots[f, j] = local_rot.as_euler('ZYX', degrees=True)

    bvh_save(args.output_bvh, {
        'rotations': bvh_rots,
        'positions': bvh_poss,
        'offsets': offsets,
        'parents': parents,
        'names': joint_names,
        'order': 'zyx',
        'frametime': frame_time,
    })
    print(f"Saved to {args.output_bvh}")
    print(f"  {num_joints} joints, {qpos_array.shape[0]} frames, {1.0/frame_time:.1f} fps")
