"""
This code provides the server-side code needed to run Gr00t.
This code is originally from the Nvidia Isaac-GR00T repo: https://github.com/NVIDIA/Isaac-GR00T

You may need to adapt this code and install the dependencies to run it on your machine.

To install the gr00t repository, you can use the following command:
- git clone https://github.com/phospho-app/Isaac-GR00T.git /path/to/save
- pip install -e /path/to/save
"""

import os
import argparse
from argparse import Namespace

# The following imports require the gr00t repository to be installed
from gr00t.eval.robot import RobotInferenceServer  # type: ignore
from gr00t.experiment.data_config import ConfigGenerator  # type: ignore
from gr00t.model.policy import Gr00tPolicy  # type: ignore

# model_path = ""  # Change this to your model path
parser = argparse.ArgumentParser(description="Run the Gr00t inference server.")
parser.add_argument(
    "--model_id", 
    type=str, 
    required=True,  # 必須の引数
    help="Path to the trained model directory."
)
args = parser.parse_args()
# Open your trained model and check the experiment_cfg/metadata.json file

# Look for the name of the embodiment tag
embodiment_tag = "new_embodiment"  # Change this to your embodiment tag, in most cases it will just be "new_embodiment"

# Please fill with the number of arms and cameras used to train the model
data_config = ConfigGenerator(num_arms=1, 
                              num_cams=2,  
                              video_keys= ["video.image_cam_0", "video.image_cam_1"], #,    #["video.cam_context", "video.cam_wrist"],
                              state_keys = ["state.arm_0"], #["state.single_arm", "state.gripper"],
                              action_keys = ["action.arm_0"] #["action.single_arm", "action.gripper"])
    )

# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"Model path {model_path} does not exist.")
# else:
#     print(f"Model path {model_path} found.")

args = Namespace(
    model_path=args.model_id,
    embodiment_tag=embodiment_tag,
    data_config=data_config,
    server=True,
    client=False,
    host="0.0.0.0",
    port=8080,
    denoising_steps=4,
)

modality_config = data_config.modality_config()
modality_transform = data_config.transform()

policy = Gr00tPolicy(
    model_path=args.model_path,
    modality_config=modality_config,
    modality_transform=modality_transform,
    embodiment_tag=args.embodiment_tag,
    denoising_steps=args.denoising_steps,
)

print(f"Policy loaded from {args.model_path}")

server = RobotInferenceServer(policy, port=args.port)

print(f"Server started on {args.host}:{args.port}")

server.run()
