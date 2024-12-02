import os
import pickle
import time

import scipy.io as sio
import torch
import torch.nn as nn
from torchsummary import summary

import datasets
import plots
import transformer
import utils


class sumToOne(nn.Module):
    def __init__(self, num_ab):
        super(sumToOne, self).__init__()
        self.num_ab = num_ab

    def forward(self, input):
        sum = torch.sum(input, dim=0, keepdim=True)
        sum = sum.repeat(self.num_ab, 1, 1)
        output = input / (sum + 1e-6)
        return output


class absrelu(nn.Module):
    def __init__(self):
        super(absrelu, self).__init__()

    def forward(self, input):
        output = abs(input)
        return output


class AutoEncoder(nn.Module):
    def __init__(self, P, L, size, num_heads):
        super(AutoEncoder, self).__init__()
        self.P, self.L, self.size = P, L, size
        # SpectralEncoder
        self.SpectralEncoder = nn.Sequential(
            nn.Conv2d(L, 128, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(),
            nn.Conv2d(64, P, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(P, momentum=0.9),
        )
        # SpatialEncoder
        self.conv1 = nn.Conv2d(L, 128, kernel_size=(5, 5), stride=1, padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(128, momentum=0.9)
        self.droupout = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64, momentum=0.9)
        self.conv3 = nn.Conv2d(64, P, kernel_size=(3, 3), stride=1, padding=(1, 1))
        # self.conv3 = nn.Conv2d(64, P, kernel_size=(1, 1), stride=1, padding=(0, 0))
        self.bn3 = nn.BatchNorm2d(P, momentum=0.9)
        self.leakyreLU = nn.LeakyReLU()

        # fusion
        self.projection_1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(),
            nn.Conv2d(64, P, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(P, momentum=0.9),
        )
        self.projection_2 = nn.Sequential(
            nn.Conv2d(64, P, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(P, momentum=0.9),
        )
        self.vtrans = transformer.ViT(image_size=size, depth=2, heads=num_heads, mlp_dim=12, pool='cls', l=L, p=P)
        # self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.sumToOne = sumToOne(P)

        # decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(),
        )

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is None:
                pass
            else:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        output1 = self.SpectralEncoder(x)

        output21 = self.conv1(x)
        output21 = self.bn1(output21)
        output21 = self.droupout(output21)
        output21 = self.leakyreLU(output21)

        output22 = self.conv2(output21)
        output22 = self.bn2(output22)
        output22 = self.leakyreLU(output22)
        output23 = self.conv3(output22)
        output23 = self.bn3(output23)

        output_1_qkv = self.projection_1(output21)
        output_2_qkv = self.projection_2(output22)
        qkv = torch.cat([output1, output_1_qkv, output_2_qkv, output23], dim=0)
        abu_est = self.vtrans(qkv)
        # abu_est = self.softmax(abu_est)
        abu_est = self.sumToOne(self.relu(abu_est))

        re_result = self.decoder(abu_est)
        return abu_est, re_result


class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(1e-6, 1)


class Train_test:
    def __init__(self, dataset, device, skip_train=False, save=False, seed=None):
        super(Train_test, self).__init__()
        self.skip_train = skip_train
        self.device = device
        self.dataset = dataset
        self.save = save
        self.save_dir = "trans_mod_" + dataset + "/"
        self.seed = seed    #记录种子
        self.early_stop_patience = 5  # Number of epochs to wait for early stopping
        self.min_delta = 1e-4  # Minimum change to consider as improvement
        os.makedirs(self.save_dir, exist_ok=True)
        if dataset == 'sy30A':
            # seed:3451345(3451346, 5, 12, 14, 15)
            self.P, self.L, self.col = 5, 200, 60
            self.LR, self.EPOCH = 2e-3, 400
            self.num_heads = 10
            self.beta, self.gamma = 0, 1
            self.weight_decay_param = 5e-5
            self.order_abd, self.order_endmem = (0, 1, 2, 3, 4), (0, 1, 2, 3, 4)
            self.data = datasets.Data(dataset, device)
            self.loader = self.data.get_loader(batch_size=self.col ** 2)
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()
            self.deLR = 0.0001
        else:
            raise ValueError("Unknown dataset")

    def run(self, smry):
        net = AutoEncoder(P=self.P, L=self.L, size=self.col, num_heads=self.num_heads).to(self.device)
        if smry:
            summary(net, (1, self.L, self.col, self.col), batch_dim=None)
            return

        net.apply(net.weights_init)
        model_dict = net.state_dict()
        model_dict['decoder.0.weight'] = self.init_weight
        net.load_state_dict(model_dict)

        criterionMSE = nn.MSELoss(reduction='mean').to(self.device)
        criterionSAD = utils.SAD2(self.L).to(self.device)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.LR, weight_decay=self.weight_decay_param)
        ignored_params = list(map(id, net.decoder[0].parameters()))  # 需要微调的参数
        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())  # 需要调整的参数
        optimizer = torch.optim.Adam([
            {'params': base_params},
            {'params': net.decoder[0].parameters(), 'lr': self.deLR}
        ], lr=self.LR, weight_decay=self.weight_decay_param)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
        apply_clamp_inst1 = NonZeroClipper()

        if not self.skip_train:
            time_start = time.time()
            net.train()
            epo_vs_los = []
            for epoch in range(self.EPOCH):
                for i, (x, _) in enumerate(self.loader):

                    x = x.transpose(1, 0).view(1, -1, self.col, self.col)
                    abu_est, re_result = net(x)

                    loss_sad = criterionSAD(re_result.view(self.L, -1).transpose(0, 1),
                                            x.view(self.L, -1).transpose(0, 1))
                    total_loss = self.gamma * torch.sum(loss_sad).float()

                    # loss_re = self.beta * criterionMSE(re_result, x.squeeze())
                    # total_loss = loss_re + loss_sad

                    optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1)
                    optimizer.step()

                    net.decoder.apply(apply_clamp_inst1)

                    if epoch % 10 == 0:
                        print('Epoch:', epoch, '| train loss: %.4f' % total_loss.data)  # ,
                        # '| re loss: %.4f' % loss_re.data,
                        # '| sad loss: %.4f' % loss_sad.data)
                    epo_vs_los.append(float(total_loss.data))

                scheduler.step()
            time_end = time.time()

            if self.save:
                with open(self.save_dir + 'weights.pickle', 'wb') as handle:
                    pickle.dump(net.state_dict(), handle)
                sio.savemat(self.save_dir + f"{self.dataset}_losses.mat", {"losses": epo_vs_los})

            print('Total computational cost:', time_end - time_start)

        else:
            with open(self.save_dir + 'weights.pickle', 'rb') as handle:
                net.load_state_dict(pickle.load(handle))
                time_end = 1
                time_start = 0

        # Testing ================

        net.eval()
        x = self.data.get("hs_img").transpose(1, 0).view(1, -1, self.col, self.col)
        abu_est, re_result = net(x)
        abu_est = abu_est / (torch.sum(abu_est, dim=0))
        abu_est = abu_est.permute(1, 2, 0).detach().cpu().numpy()
        target = torch.reshape(self.data.get("abd_map"), (self.col, self.col, self.P)).cpu().numpy()
        true_endmem = self.data.get("end_mem").numpy()
        est_endmem = net.state_dict()["decoder.0.weight"].cpu().numpy()
        est_endmem = est_endmem.reshape((self.L, self.P))

        abu_est = abu_est[:, :, self.order_abd]
        est_endmem = est_endmem[:, self.order_endmem]

        sio.savemat(self.save_dir + f"{self.dataset}_abd_map.mat", {"A_est": abu_est})
        sio.savemat(self.save_dir + f"{self.dataset}_endmem.mat", {"E_est": est_endmem})

        x = x.view(-1, self.col, self.col).permute(1, 2, 0).detach().cpu().numpy()
        re_result = re_result.view(-1, self.col, self.col).permute(1, 2, 0).detach().cpu().numpy()
        re = utils.compute_re(x, re_result)
        print("RE:", re)

        rmse_cls, mean_rmse = utils.compute_rmse(target, abu_est)
        print("Class-wise RMSE value:")
        for i in range(self.P):
            print("Class", i + 1, ":", rmse_cls[i])
        print("Mean RMSE:", mean_rmse)

        sad_cls, mean_sad = utils.compute_sad(est_endmem, true_endmem)
        print("Class-wise SAD value:")
        for i in range(self.P):
            print("Class", i + 1, ":", sad_cls[i])
        print("Mean SAD:", mean_sad)

        with open(self.save_dir + "log.csv", 'a') as file:
            file.write(f"Seed: {self.seed}, ")
            file.write(f"LR: {self.LR}, ")
            file.write(f"WD: {self.weight_decay_param}, ")
            file.write(f"num_heads: {self.num_heads}, ")
            file.write(f"beta: {self.beta}, ")
            file.write(f"gamma: {self.gamma}, ")
            file.write(f"RE: {re:.4f}, ")
            file.write(f"SAD: {mean_sad:.4f}, ")
            file.write(f"RMSE: {mean_rmse:.4f}\n")

        with open(self.save_dir + "Eva.csv", 'a') as file:
            # file.truncate(0)
            file.write(f"Seed,LR,EPOCH,num_heads,beta,gamma,weight_decay_param,,RE,SAD,RMSE,Time\n")
            file.write(
                f"{self.seed},{self.LR},{self.EPOCH},{self.num_heads},{self.beta},{self.gamma},{self.weight_decay_param},,{re:.5f},{mean_sad:.5f},{mean_rmse:.5f},{time_end - time_start:.5f}\n")

        with open(self.save_dir + "Cls_Eva.csv", 'a') as file:
            # file.truncate(0)
            file.write(f"Seed: {self.seed}, Time: {time_end - time_start:.5f}\n")
            file.write(f"Class-wise RMSE value:\n")
            for i in range(self.P):
                file.write(f"Class {i + 1},{rmse_cls[i]:.5f}\n")
            file.write(f"Mean RMSE,{mean_rmse:.5f}\n")
            file.write(f"Class-wise SAD value:\n")
            for i in range(self.P):
                file.write(f"Class {i + 1},{sad_cls[i]:.5f}\n")
            file.write(f"Mean SAD,{mean_sad:.5f}\n\n")

        plots.plot_abundance(target, abu_est, self.P, self.save_dir)
        plots.plot_endmembers(true_endmem, est_endmem, self.P, self.save_dir)


# =================================================================

if __name__ == '__main__':
    pass
