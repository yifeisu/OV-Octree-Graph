import os

import yaml

if __name__ == '__main__':
    with open('scannetv2_val.txt', 'r') as f:
        data = f.readlines()

    for scene in data:
        scene = scene.strip()
        print(scene)

        lines = []
        cam_info_path = os.path.join(f"./scene_datasets/scannet/aligned_scans/", scene, f"{scene}.txt")
        with open(cam_info_path, 'r') as fp:
            line = fp.readline().strip()
            while line:
                lines.append(line)
                line = fp.readline().strip()

        scene_data = {
            'dataset_name': 'scannet',
            'camera_params':
                {
                    'image_height': 968,
                    'image_width': 1296,
                    'fx': None,
                    'fy': None,
                    'cx': None,
                    'cy': None,
                    'png_depth_scale': 1000.0
                }
        }

        prefixes = {
            "depthHeight": "image_height",
            "depthWidth": "image_width",
            "fx_depth": "fx",
            "fy_depth": "fy",
            "mx_depth": "cx",
            "my_depth": "cy"
        }
        for prefix, key in prefixes.items():
            found = False
            for line in lines:
                if line.startswith(prefix):
                    found = True
                    data = line.split('=')[1][1:]
                    scene_data['camera_params'][key] = eval(data)
                    break
            assert found, f"{scene}-{prefix}"

        with open(f"../datasets/configs/scannet/{scene}.yaml", 'w') as f:
            cfg = yaml.dump(scene_data, f)
