from pathlib import Path

import torch
from torch.optim.adam import Adam

from app.code_torch.framework.base_classes import Config, ModelManager
from app.code_torch.framework.models import UnetResnet34


class LaneExModelManager(ModelManager):
    def __init__(self, config: Config):
        super().__init__(config)

        ch_in = 3
        ch_out = 4

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.network = UnetResnet34(ch_in, ch_out).to(self.device)
        self.optimizer = Adam(self.network.parameters(), lr=0.001)

    @staticmethod
    def loss_function(output, mask, target, target_normal) -> torch.Tensor:
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
        loss += torch.mean(mask * torch.square(target_normal - output[:, 2:4, :, :]))

        return loss

    def train(self, batch, lr):
        # Moving channels index from 1 to 3
        batch = tuple(
            arr.transpose(0, 3, 1, 2) if len(arr.shape) == 4 else arr for arr in batch
        )
        x_in = batch[0]
        x_mask = batch[1]
        target = batch[2]
        target_normal = batch[3]

        input_im = x_in
        input_im = torch.tensor(input_im, dtype=torch.float32, device=self.device)
        x_mask = torch.tensor(x_mask, dtype=torch.float32, device=self.device)
        target = torch.tensor(target, dtype=torch.float32, device=self.device)
        target_normal = torch.tensor(
            target_normal, dtype=torch.float32, device=self.device
        )
        self.optimizer.zero_grad()
        output_res = self.network(input_im)
        output_seg = torch.softmax(output_res[:, 0:2, :, :], dim=1)[:, 0:1, :, :]
        output_dir = output_res[:, 2:4, :, :]

        lossCur = self.loss_function(output_res, x_mask, target, target_normal)
        lossCur.backward()

        output = torch.cat((output_seg, output_dir), dim=1)

        self.optimizer.param_groups[0]["lr"] = lr
        self.optimizer.step()

        result = (
            lossCur.item(),
            torch.permute(output, (0, 2, 3, 1)).detach().cpu().numpy(),
            None,
        )
        return result

    def infer(self, input):
        # TODO Check, prob doesnt work anymore
        x_in = input
        self.network.eval()
        with torch.no_grad():
            x_in = torch.permute(torch.FloatTensor(x_in), (0, 3, 1, 2)).to(self.device)
            result = self.network(x_in)
            return torch.permute(result, (0, 2, 3, 1)).detach().cpu().numpy()

    def save_model(self, ep: int):
        path = self.config.model_folder
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path / f"TurnVal_ep_{ep}",
        )

    def restore_model(self, path: Path):
        checkpoint = torch.load(path)

        self.network.load_state_dict(checkpoint["network"])
