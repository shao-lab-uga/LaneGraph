from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from app.new_code_copy.framework.base_classes import ModelManager
from app.new_code_copy.framework.models import UnetResnet34
from torch.optim.adam import Adam


class TurnExModelManager(ModelManager):
    def __init__(
        self,
        batch_size: int = 4,
        net_name: str = "v1",
        size: int = 640,
    ) -> None:
        super().__init__(batch_size)
        self.net_name = net_name
        self.position_code = np.ndarray((batch_size, 2, size, size), dtype=np.float32)
        for i in range(size):
            self.position_code[:, 0, i, :] = float(i) / size
            self.position_code[:, 0, :, i] = float(i) / size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.network = UnetResnet34(ch_in=14, ch_out=2).to(self.device)

        self.optimizer = Adam(self.network.parameters(), lr=0.001)

    @staticmethod
    def loss_function(output, target, mask) -> torch.Tensor:
        t1 = target  # N,1,640,640
        p = output  # N,3,640,640, 1st is seg, 2nd 3rd direction

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

    def train(self, batch, lr):
        batch = tuple(
            arr.transpose(0, 3, 1, 2) if len(arr.shape) == 4 else arr for arr in batch
        )

        inputs, connectors, targets, target_labels, normals = batch
        self.optimizer.zero_grad()

        input_data = np.concatenate(
            (inputs, connectors, normals, self.position_code), axis=1
        )
        targets = torch.tensor(targets, dtype=torch.float32, device=self.device)

        input_data = torch.tensor(input_data, dtype=torch.float32, device=self.device)
        output = self.network(input_data)

        output = F.softmax(output, dim=1)

        lossCur = self.loss_function(output, targets, 1.0)
        lossCur.backward()

        self.optimizer.param_groups[0]["lr"] = lr
        self.optimizer.step()

        result = (
            lossCur.item(),
            torch.permute(output, (0, 2, 3, 1)).detach().cpu().numpy(),
            None,
        )
        return result

    def infer(self, input) -> np.ndarray:
        return np.zeros((1, 1))

    def save_model(self, path: Path):
        torch.save(self.network.state_dict(), path)

    def restore_model(self, path: Path):
        self.network.load_state_dict(torch.load(path))
