"""
Batch retarget all Flomo Xsens BVH files in a directory to Geno/LAFAN1 skeleton.

Usage:
    python scripts/flomo_batch_retarget.py \
        --input_dir "F:\experiments\tests202601\Flomo\data\timi" \
        --skip_tpose

    # or with a separate output directory:
    python scripts/flomo_batch_retarget.py \
        --input_dir "F:\experiments\tests202601\Flomo\data\timi" \
        --output_dir some/other/dir \
        --skip_tpose

Output files are placed next to the inputs by default, with .bvh replaced by .geno.bvh:
    xs_20251101_aleblanc_lantern_nav-003.fbx.bvh  ->  xs_20251101_aleblanc_lantern_nav-003.fbx.geno.bvh
"""
import argparse
import os
import glob
import subprocess
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch retarget Flomo BVH files to Geno/LAFAN1")
    parser.add_argument("--input_dir", required=True, type=str, help="Directory containing .bvh files")
    parser.add_argument("--output_dir", type=str, help="Output directory (default: same as input)")
    parser.add_argument("--skip_tpose", action="store_true", help="Skip first frame (T-pose)")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    bvh_files = sorted(glob.glob(os.path.join(args.input_dir, "*.bvh")))
    # don't re-retarget our own output files
    bvh_files = [f for f in bvh_files if not f.endswith(".geno.bvh")]
    if not bvh_files:
        print(f"No .bvh files found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(bvh_files)} BVH files in {args.input_dir}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    retarget_script = os.path.join(script_dir, "flomo_to_geno_bvh.py")

    failed = []
    for i, bvh_file in enumerate(bvh_files):
        basename = os.path.basename(bvh_file)
        # foo.fbx.bvh -> foo.fbx.geno.bvh
        name = basename[:-4] if basename.endswith(".bvh") else basename
        output_file = os.path.join(output_dir, name + ".geno.bvh")

        print(f"\n[{i + 1}/{len(bvh_files)}] {basename}")

        if os.path.exists(output_file):
            print(f"  already exists, skipping")
            continue

        cmd = [sys.executable, retarget_script,
               "--bvh_file", bvh_file,
               "--output_bvh", output_file]
        if args.skip_tpose:
            cmd.append("--skip_tpose")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FAILED: {result.stderr[-500:] if result.stderr else 'no error output'}")
            failed.append(basename)
        else:
            lines = result.stdout.strip().split('\n')
            for line in lines[-2:]:
                print(f"  {line}")

    print(f"\n{'='*60}")
    print(f"Done. {len(bvh_files) - len(failed)}/{len(bvh_files)} succeeded.")
    if failed:
        print(f"Failed ({len(failed)}):")
        for f in failed:
            print(f"  {f}")
