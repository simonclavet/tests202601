import os

def bvh_to_mujoco_xml(bvh_path, output_xml_path):
    with open(bvh_path, 'r') as f:
        lines = f.readlines()

    xml_lines = [
        '<mujoco model="geno">',
        '  <compiler angle="radian" meshdir="meshes" autolimits="true"/>',
        '  <option timestep="0.005" iterations="50" tolerance="1e-10" solver="Newton" jacobian="dense" cone="elliptic"/>',
        '  <size njmax="500" nconmax="100"/>',
        '  <visual>',
        '    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>',
        '    <rgba haze="0.15 0.25 0.35 1"/>',
        '    <global azimuth="-140" elevation="-20"/>',
        '  </visual>',
        '  <asset>',
        '    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>',
        '    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="1 1 1" rgb2="1 1 1" markrgb="0 0 0" width="300" height="300"/>',
        '    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0"/>',
        '  </asset>',
        '  <worldbody>',
        '    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>',
        '    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>'
    ]

    stack = []
    indent = 4
    # track End Site nesting depth so we skip their closing braces.
    # without this, every End Site '}' prematurely closes a JOINT body
    # and the entire hierarchy shifts by one level per End Site.
    end_site_depth = 0

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('End Site'):
            # entering an End Site block — skip everything until its '}'
            end_site_depth += 1
        elif line == '{' and end_site_depth > 0:
            # opening brace of an End Site block, just skip
            pass
        elif line == '}' and end_site_depth > 0:
            # closing brace of an End Site block, don't pop the joint stack
            end_site_depth -= 1
        elif line.startswith('ROOT') or line.startswith('JOINT'):
            name = line.split()[1]
            # Find next OFFSET
            while 'OFFSET' not in lines[i]:
                i += 1
            offset_line = lines[i].strip().split()
            pos = [float(x) / 100.0 for x in offset_line[1:]]

            # Map BVH (X, Y_up, Z_forward) to MuJoCo (X, -Z_forward, Y_up)
            # or simpler: X -> X, Y -> Z, Z -> -Y to make Z up.
            mj_pos = f"{pos[0]} {-pos[2]} {pos[1]}"

            xml_lines.append(' ' * indent + f'<body name="{name}" pos="{mj_pos}">')

            if name == 'Hips':
                xml_lines.append(' ' * indent + f'  <freejoint name="{name}"/>')
            else:
                # 3-DOF ball joint for humanoid joints
                xml_lines.append(' ' * indent + f'  <joint name="{name}" type="ball"/>')

            # Add site for IK
            xml_lines.append(' ' * indent + f'  <site name="{name}" size="0.01" rgba="1 0 0 1"/>')
            # Add geom for visualization (connecting to parent)
            # For simplicity, just a small sphere at the joint
            xml_lines.append(' ' * indent + f'  <geom type="sphere" size="0.015" rgba="0.5 0.5 0.5 1"/>')

            stack.append(name)
            indent += 2
        elif line == '}':
            if stack:
                stack.pop()
                indent -= 2
                xml_lines.append(' ' * indent + '</body>')
        i += 1

    xml_lines.append('  </worldbody>')
    xml_lines.append('</mujoco>')

    os.makedirs(os.path.dirname(output_xml_path), exist_ok=True)
    with open(output_xml_path, 'w') as f:
        f.write('\n'.join(xml_lines))

if __name__ == "__main__":
    bvh_to_mujoco_xml(r"F:\experiments\lafan1-resolved\Geno_bind.bvh", r"F:\experiments\GMR\assets\geno\geno.xml")
