import argparse
import pathlib
import os
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.xsens import load_xsens_file

def bvh_save(filename, data):
    channelmap_inv = {'x': 'Xrotation', 'y': 'Yrotation', 'z': 'Zrotation'}
    rots, poss, offsets, parents = data['rotations'], data['positions'], data['offsets'], data['parents']
    names = data.get('names', ["joint_" + str(i) for i in range(len(parents))])
    order = data.get('order', 'zyx')
    frametime = data.get('frametime', 1.0/60.0)
    
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
                        poss[i,j,0], poss[i,j,1], poss[i,j,2], 
                        rots[i,j,0], rots[i,j,1], rots[i,j,2]))
                else:
                    f.write("%.9g %.9g %.9g " % (rots[i,j,0], rots[i,j,1], rots[i,j,2]))
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh_file", required=True, type=str)
    parser.add_argument("--output_bvh", required=True, type=str)
    parser.add_argument("--scale", default=0.01, type=float)
    args = parser.parse_args()

    # Load Xsens data
    xsens_args = argparse.Namespace(
        bvh_file=args.bvh_file,
        scale=args.scale,
        start=None,
        end=None,
        reset_to_zero=False,
        bvh_format="3DSM"
    )
    lafan1_data_frames, actual_human_height, frame_time = load_xsens_file(xsens_args)

    retargeter = GMR(src_human="bvh_xsens", tgt_robot="geno", actual_human_height=actual_human_height)
    model = retargeter.model
    qpos_list = []
    
    print(f"Retargeting {len(lafan1_data_frames)} frames...")
    for frame_data in lafan1_data_frames:
        qpos_list.append(retargeter.retarget(frame_data).copy())

    num_joints = model.nbody - 1
    joint_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
    parents = [model.body_parentid[i] - 1 if model.body_parentid[i] > 0 else -1 for i in range(1, model.nbody)]
    offsets = [model.body_pos[i] * 100.0 for i in range(1, model.nbody)]

    qpos_array = np.array(qpos_list)
    bvh_poss = np.zeros((qpos_array.shape[0], num_joints, 3))
    bvh_rots = np.zeros((qpos_array.shape[0], num_joints, 3))
    
    for f in range(qpos_array.shape[0]):
        q = qpos_array[f]
        # Mapping MJ(x, y, z) back to BVH(x, z, -y)
        bvh_poss[f, 0] = [q[0]*100, q[2]*100, -q[1]*100]
        # Root rot
        bvh_rots[f, 0] = R.from_quat([q[4], q[5], q[6], q[3]]).as_euler('zyx', degrees=True)
        q_idx = 7
        for j in range(1, num_joints):
            bvh_rots[f, j] = R.from_quat([q[q_idx+1], q[q_idx+2], q[q_idx+3], q[q_idx]]).as_euler('zyx', degrees=True)
            q_idx += 4

    bvh_save(args.output_bvh, {'rotations': bvh_rots, 'positions': bvh_poss, 'offsets': offsets, 'parents': parents, 'names': joint_names, 'order': 'zyx', 'frametime': frame_time})
    print(f"Saved retargeted BVH to {args.output_bvh}")
