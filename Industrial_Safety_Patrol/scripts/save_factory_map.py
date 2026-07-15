#!/usr/bin/env python3
"""SLAM으로 생성된 Occupancy Grid Map을 factory_map.yaml/.pgm으로 저장."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Save /map topic to factory_map.yaml and factory_map.pgm"
    )
    parser.add_argument(
        "--output",
        default="/workspace/maps/factory_map",
        help="Output path prefix without extension (default: /workspace/maps/factory_map)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for /map topic (default: 10)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving map to {output_path}.yaml / {output_path}.pgm ...")
    print("Ensure SLAM Toolbox is running and the map has been explored.")

    cmd = [
        "ros2",
        "run",
        "nav2_map_server",
        "map_saver_cli",
        "-f",
        str(output_path),
        "--ros-args",
        "-p",
        f"save_map_timeout:={args.timeout}",
    ]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("Map save failed. Check that /map is being published.", file=sys.stderr)
        sys.exit(result.returncode)

    yaml_file = output_path.with_suffix(".yaml")
    pgm_file = output_path.with_suffix(".pgm")
    if yaml_file.exists() and pgm_file.exists():
        print(f"Map saved successfully:")
        print(f"  {yaml_file}")
        print(f"  {pgm_file}")
    else:
        print("Map saver finished but output files were not found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
