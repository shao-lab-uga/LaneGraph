from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.adam import Adam

from app.code_torch.framework.base_classes import ModelManager, SimpleConfig
from app.code_torch.framework.models import UnetResnet34


class TurnValModelManager(ModelManager):
    def __init__(self, config: SimpleConfig) -> None:
        super().__init__(config)

        # Positional encodings for NNs
        size = self.config.image_size
        self.position_code = np.ndarray(
            (config.batch_size, 2, size, size), dtype=np.float32
        )
        for i in range(size):
            self.position_code[:, 0, i, :] = float(i) / size
            self.position_code[:, 0, :, i] = float(i) / size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.network = UnetResnet34(ch_in=10, ch_out=2).to(self.device)
        self.optimizer = Adam(self.network.parameters(), lr=0.001)

    @staticmethod
    def seg_loss_function(output, mask, target) -> torch.Tensor:
        t1 = target
        p = output

        def ce_loss(p, t):
            pp0 = p[:, 0:1, :, :]
            pp1 = p[:, 1:2, :, :]

            loss = -(
                t * pp0 + (1 - t) * pp1 - torch.log(torch.exp(pp0) + torch.exp(pp1))
            )
            loss = torch.mean(loss * mask)
            return loss

        def dice_loss(p, t):
            # return 0
            p = torch.sigmoid(p[:, 0:1, :, :] - p[:, 1:2, :, :])
            numerator = 2 * torch.sum(p * t * mask)
            denominator = torch.sum((p + t) * mask) + 1.0
            return 1 - numerator / denominator

        loss = 0
        loss += ce_loss(p, t1) + dice_loss(p, t1) * 0.333
        return loss

    def train(self, batch: Tuple[np.ndarray, ...], lr: float):
        # Moving channels index from 3 to 1
        batch = tuple(
            arr.transpose(0, 3, 1, 2) if len(arr.shape) == 4 else arr for arr in batch
        )
        x_in, x_connector, target, target_label, context = batch
        context = context / np.max(context)  # normalizing to check error
        self.optimizer.zero_grad()

        # Segmenting from first node
        input_seg_1 = np.concatenate(
            (x_in, x_connector[:, 0:3, :, :], context, self.position_code), axis=1
        )
        input_seg_1 = torch.tensor(input_seg_1, dtype=torch.float32, device=self.device)
        output_seg_1 = self.network(input_seg_1)

        # Segmenting from second node
        input_seg_2 = np.concatenate(
            (x_in, x_connector[:, 3:6, :, :], context, self.position_code), axis=1
        )
        input_seg_2 = torch.tensor(input_seg_2, dtype=torch.float32, device=self.device)
        output_seg_2 = self.network(input_seg_2)

        # Running classification
        output_seg = torch.cat(
            (
                F.softmax(output_seg_1, dim=1)[:, 0:1, :, :],
                F.softmax(output_seg_2, dim=1)[:, 0:1, :, :],
            ),
            dim=1,
        )

        class_input = np.concatenate((x_connector, context, self.position_code), axis=1)
        class_input = torch.tensor(class_input, dtype=torch.float32, device=self.device)
        class_input = torch.cat((output_seg, class_input), dim=1)

        # output_label = F.softmax(output_label)

        target = torch.tensor(target, dtype=torch.float32, device=self.device)
        target_label = torch.tensor(
            target_label, dtype=torch.float32, device=self.device
        )

        # IDK why mask is just 1. remove later?
        seg_loss = self.seg_loss_function(
            output_seg_1[:, 0:2, :, :], 1.0, target[:, 1:2, :, :]
        )
        seg_loss += self.seg_loss_function(
            output_seg_2[:, 0:2, :, :], 1.0, target[:, 2:3, :, :]
        )
        lossCur = seg_loss
        lossCur.backward()

        self.optimizer.param_groups[0]["lr"] = lr
        self.optimizer.step()
        if np.isnan(lossCur.item()):
            raise ValueError("!NAN!")

        return (
            lossCur.item(),
            output_seg.detach().cpu().numpy(),
            None,
        )

    def infer(self, input_im):
        input_arr = torch.tensor(input_im, dtype=torch.float32, device=self.device)
        input_arr = torch.permute(input_arr, (0, 3, 2, 1))
        self.network.eval()
        with torch.no_grad():
            res = self.network(input_arr)
            out = F.softmax(res, dim=1)[0, 0:1, :, :]
        return out.detach().cpu().numpy()

    def save_model(self, ep: int):
        path = self.config.model_folder
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path / f"TurnPair_ep_{ep}",
        )

    def restore_model(self, path: Path):
        checkpoint = torch.load(path)

        self.network.load_state_dict(checkpoint["network"])
        return
