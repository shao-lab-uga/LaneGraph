from time import time
import sys
import json


class TrainingFramework:
    def __init__(self):
        pass

    def run(self, config):
        lr = config["learningrate"]
        lr_decay = config["lr_decay"]
        lr_decay_step = config["lr_decay_step"]
        step = config["step_init"]
        maxstep = config["step_max"]
        last_step = -1
        if "logfile" in config:
            logfile = config["logfile"]
        else:
            logfile = "log.json"

        use_validation = config["use_validation"]

        self.kv = {}

        logs = {}

        def addlog(k, v, s):
            if k in logs:
                logs[k][0].append(s)
                logs[k][1].append(v)
            else:
                logs[k] = [[s], [v]]

        # ===============================
        # initialization actions
        # ===============================
        if config["disable_tensorflow_all"]:
            import torch

            sess = None
        else:
            import tensorflow as tf

            gpu_options = tf.GPUOptions(allow_growth=True)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        model = self.createModel(sess)
        dataloader = self.createDataloader("training")
        # if use_validation:
        # 	dataloader_validation = self.createDataloader("testing")

        loss = 0
        t_load = 0
        t_preload = 0
        t_train = 0
        t_other = 0
        t0 = time()

        lastprogress = -1

        # ===============================
        # begin: training loop
        # ===============================
        while True:
            t_other += time() - t0
            t0 = time()
            batch = self.getBatch(dataloader)

            # from resnet34unet import resnet34unet_v3, unet, unet_dilated
            # with tf.compat.v1.variable_scope("test", reuse=tf.compat.v1.AUTO_REUSE):
            # 	aaa=resnet34unet_v3(tf.convert_to_tensor(batch[0], tf.float32), True, ch_in = 3, ch_out = 4)
            # 	pass

            t1 = time()

            if "torch" in config["model_name"]:
                input = torch.permute(torch.FloatTensor(batch[0]), (0, 3, 1, 2)).to(
                    model.device
                )  # need to rearrange order, N,640,640,4->N,4,640,640
                mask = torch.permute(torch.FloatTensor(batch[1]), (0, 3, 1, 2)).to(
                    model.device
                )  # N,1,640,640
                target = torch.permute(torch.FloatTensor(batch[2]), (0, 3, 1, 2)).to(
                    model.device
                )  # N,1,640,640
                target_normal = torch.permute(
                    torch.FloatTensor(batch[3]), (0, 3, 1, 2)
                ).to(
                    model.device
                )  # N,2,640,640

                # Zero your gradients for every batch
                model.optimizer_torch.zero_grad()

                # Make predictions for this batch
                output_res = model.backboneModelTorch(input)
                output_seg = torch.softmax(output_res[:, 0:2, :, :], dim=1)[
                    :, 0:1, :, :
                ]  # torch.softmax(output_res[:,0:2,:,:], dim=1)[0,0:1,:,:] equal to torch.softmax(output_res[0,0:2,:,:], dim=0)[0:1,:,:]
                output_direction = output_res[:, 2:4, :, :]
                output = torch.concat(
                    [output_seg, output_direction], axis=1
                )  # N,3,640,640

                # Compute the loss and its gradients
                lossCur = model.lossFunctionTorch(
                    output_res, mask, target, target_normal
                )
                lossCur.backward()

                # Adjust learning weights
                model.optimizer_torch.param_groups[0]["lr"] = lr
                model.optimizer_torch.step()

                # agument result
                result = [
                    lossCur.item(),
                    torch.permute(output, (0, 2, 3, 1)).detach().cpu().numpy(),
                    None,
                ]

            else:
                result = self.train(batch, lr)

            t2 = time()
            self.preload(dataloader, step)
            t3 = time()

            t_load += t1 - t0
            t_train += t2 - t1
            t_preload += t3 - t2

            loss += self.getLoss(result)

            if step % 10 == 0:
                progress = self.getProgress(step)
                p = int((progress - int(progress)) * 50)
                loss /= 10.0

                addlog("loss", loss, step)

                s = ""
                ks = sorted(self.kv.keys())
                for k in ks:
                    v = self.kv[k]
                    if v[1] > 0:
                        s += " %s:%E " % (k, v[0] / float(v[1]))
                        addlog(k, v[0] / float(v[1]), step)
                        self.kv[k] = [0, 0]

                sys.stdout.write(
                    "\rstep %d epoch:%.2f " % (step, progress)
                    + ">" * p
                    + "-" * (51 - p)
                    + " loss %f time %f %f %f %f "
                    % (
                        loss,
                        t_preload,
                        t_load,
                        t_train,
                        t_other - t_preload - t_load - t_train,
                    )
                    + s
                )
                sys.stdout.flush()

                p = int((progress - int(progress)) * 100000)
                if int(progress) != int(lastprogress):
                    print("time per epoch", t_other)
                    print(
                        "eta",
                        t_other
                        * (maxstep - step)
                        / max(1, (step - last_step))
                        / 3600.0,
                    )
                    last_step = step
                    t_load = 0
                    t_preload = 0
                    t_train = 0
                    t_other = 0

                loss = 0
                lastprogress = progress

            self.saveModel(step)
            self.visualization(step, result, batch)

            for i in range(len(lr_decay_step)):
                if step == lr_decay_step[i]:
                    lr = lr * lr_decay[i]

            if step == maxstep + 1:
                break

            if step % 1000 == 0 and step > 0:
                json.dump(logs, open(logfile, "w"), indent=2)

            step += 1
        # ===============================
        # end: training loop
        # ===============================

        # ===============================
        # shutdown actions
        # ===============================
        if config["disable_tensorflow_all"]:
            pass
        else:
            sess.close()
        # ===============================
        # ===============================

    # features
    def logvalue(self, k, v):
        if k in self.kv:
            self.kv[k][0] = self.kv[k][0] + v
            self.kv[k][1] = self.kv[k][1] + 1
        else:
            self.kv[k] = [v, 1]

    # virtual methods
    def createDataloader(self, mode):
        print("createDataloader not implemented")
        exit()

    def createModel(self, sess):
        print("createModel not implemented")
        exit()

    def getBatch(self, dataloader):
        print("getBatch not implemented")
        exit()

    def train(self, batch, lr):
        print("train not implemented")
        exit()

    def preload(self, dataloader, step):
        print("preload not implemented")
        exit()

    # placeholder methods
    def getLoss(self, result):
        return 0.0

    def getProgress(self, step):
        return 0.0

    def saveModel(self, step):
        return False

    def visualization(self, step, result=None, batch=None):
        return False
