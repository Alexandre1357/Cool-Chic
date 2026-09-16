import argparse
import os
import subprocess
import sys

resolution = "1k"

typeToFilename = {
    "diff": "diff",
    "normal_xy": "nor_gl_xy",
    "rough": "rough",
    "rm": "rm",
}

datasets = [
    { "name": "Rails001",        "types": "diff,normal_xy,rm"    },
    # { "name": "Bricks090",       "types": "diff,normal_xy,rough" },
    # { "name": "Carpet015",       "types": "diff,normal_xy,rough" },
    # { "name": "MetalPlates013",  "types": "diff,normal_xy,rm"    },
    # { "name": "PavingStones070", "types": "diff,normal_xy,rough" },
    # { "name": "Wood063",         "types": "diff,normal_xy,rough" },
    # { "name": "aerial_rocks_02", "types": "diff,normal_xy,rough" },
    # { "name": "forrest_sand_01", "types": "diff,normal_xy,rough" },
    # { "name": "red_dirt_mud_01", "types": "diff,normal_xy,rough" },
    # { "name": "roof_09",         "types": "diff,normal_xy,rough" },
]

configs = [
    { "name": "hop7_3e-4_brdf07_l1_test3", "file": "hop7", "tune": "brdf07_l1", "lambda": "0.0003", "iterations": 2000 },
    # { "name": "hop7_3e-4_brdf07_rel_mse", "file": "hop7", "tune": "brdf07_rel_mse", "lambda": "0.0003", "iterations": 10000 },
    # { "name": "hop7_3e-4_brdf07_l1", "file": "hop7", "tune": "brdf07_l1", "lambda": "0.0003", "iterations": 10000 },
    # { "name": "hop7_3e-4_brdf09_l1", "file": "hop7", "tune": "brdf09_l1", "lambda": "0.0003", "iterations": 10000 },
    # { "name": "hop7_3e-4_brdf07_l1_lpips", "file": "hop7", "tune": "brdf07_l1_lpips", "lambda": "0.0003", "iterations": 10000 },
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Coolchic experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source_path",
        required=True,
        type=str,
    )

    args = parser.parse_args()

    for config in configs:
        config_name = config["name"]
        config_file = config["file"]
        tune = config["tune"]
        lambda_var = config["lambda"]
        num_iters = config["iterations"]

        for dataset in datasets:
            texture_name = dataset["name"]
            texture_dir = f"{args.source_path}/{texture_name}/{resolution}"
            texture_types = dataset["types"]

            input_paths = []
            skip = False
            for type in texture_types.split(","):
                file_name = typeToFilename[type]
                if type not in typeToFilename:
                    print(f"WARNING! type {type} invalid")
                    skip = True
                    break

                input_path = f"{texture_dir}/{file_name}.png"
                if not os.path.exists(input_path):
                    print(f"WARNING! path {input_path} invalid")
                    skip = True
                    break

                input_paths.append(input_path)

            if skip:
                continue

            input_arg = ",".join(input_paths)
            output_path = f"bitstream_output/{config_name}/{texture_name}.cool"
            work_dir = f"./workdir7ch/{config_name}/{texture_name}"

            if not os.path.exists(work_dir):
                os.makedirs(work_dir)

            print(f"Encoding {texture_name}")

            run_string = [sys.executable, "cc_encode.py"] + [
                f"-i={input_arg}",
                f"-o={output_path}",
                f"--workdir={work_dir}",
                f"--dec_cfg_residue=cfg/dec/intra/{config_file}.cfg",
                f"--n_itr={num_iters}",
                f"--lmbda={lambda_var}",
                f"--tune={tune}",
                "--start_lr=0.01",
            ]

            subprocess.run(run_string)

    print("All textures encoded successfully")