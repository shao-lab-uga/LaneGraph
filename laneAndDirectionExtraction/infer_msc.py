import app.config as config
import sys


from model import LaneModel
import imageio.v3 as imageio
import numpy as np
import plotly.express as px
from pathlib import Path
import torch

inputfile = (
    "/home/lab/development/lab/cleaned/LaneGraph/msc_dataset/samples/sat_0_sam.jpg"
)

cnninput = 640
img = imageio.imread(inputfile)
img = (img.astype(np.float32) / 255.0 - 0.5) * 0.81
dim = np.shape(img)

mask = np.zeros((cnninput, cnninput, 3))

output = np.zeros_like(img)
weights = np.zeros_like(img) + 0.0001

x_in = torch.tensor(img, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)

model = LaneModel(None, size=640, batchsize=1, backbone="resnet34_torch")
model.restoreModel(
    "/home/lab/development/lab/cleaned/LaneGraph/app/LogTmp/2024_12_16_23_13_00/model_epoch_0"
)

out = model.infer(x_in)
img_out = out.cpu().numpy()
img_out = np.squeeze(img_out, axis=0)
img_out = np.transpose(
    img_out,
    (
        1,
        2,
        0,
    ),
)
img_out = (
    img_out[
        :,
        :,
        :3,
    ]
    * 255
    + 0.5
)
fig = px.imshow(img_out)
fig.show()
