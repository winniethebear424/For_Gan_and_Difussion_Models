from models.UNet import UNet
from losses import VAELoss
from utils.data_utils import set_seed, get_device, AverageMeter
from utils.trainer import Trainer
import os, torch, time, argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import yaml
import numpy as np

class DiffusionTrainer(Trainer):
    def __init__(self, config, output_dir=None, device=None):
        super().__init__(config, output_dir=output_dir, device=device)

        self.net = self._init_diffuser(config)
        self.net.to(device=self.device)
        self.optimizer = self._init_optimizer(self.net)
        self.timesteps = self.config.diffusion.timesteps

        self.fixed_eval_noise = torch.randn((self.batch_size, 1, self.height, self.width), device=self.device)
        # note provided criterion. use this when you want to compute MSE
        self.criterion = nn.MSELoss(reduction='mean')

        # self.time_dim = self.config.diffusion.time_dim
        self.noise_start = self.config.diffusion.noise_start
        self.noise_end = self.config.diffusion.noise_end


        #############################################################################
        # TODO:                                                                     #
        # 1. Create noise schedule beta across timesteps. 
        #   remember to put beta on the device provided.                          #
        # 2. Compute alpha terms from beta                                         #
        # 3. Calculate cumulative alpha terms                                      #
        # Consider:                                                                #
        # - What       ranges for beta?                                    #
        # - What does the cumulative product represent?                           #
        #############################################################################
        # Primary attributes
        # # 1. create noise
        self.beta = torch.linspace(self.noise_start, self.noise_end, self.timesteps).to(self.device)
        # 2. compute alpha terms from beta
        self.alpha = 1.0 - self.beta
        # 3. calculate cumulative alpha terms
        self.alphas_bar = torch.cumprod(self.alpha, dim=0)  # \bar{α}_t

        # Aliases so both naming conventions work
        self.betas = self.beta
        self.alphas = self.alpha
        self.alphas_cumprod = self.alphas_bar
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    @staticmethod
    def _build_diffuser(cfg):
        return UNet(cfg
                )

    def _init_diffuser(self,cfg):
        net = self._build_diffuser(cfg
                              )
        return net


    def forward_diffusion(self, x_0, t):
        x_t, noise = None, None
        #############################################################################
        # TODO:                                                                     #
        # 1. Extract relevant schedule parameters for timestep t                    #
        # 2. Generate noise sample                                                  #
        # 3. Combine clean data and noise according to schedule                     #
        # Consider:                                                                 #
        # - Why this particular combination of terms?                               #
        # - How does this relate to the reverse process?                            #
        #############################################################################
        # 1. Extract relevant schedule parameters for timestep t                    #
        batch_size = x_0.size(0)
        
        # 1. sample noise
        noise = torch.randn_like(x_0).to(self.device)  # ε_t

        # 2. extract culmulative alpha
        alpha_bar_t = self.alphas_bar[t].view(batch_size, 1, 1, 1)
        
        # 3. forward diffusion                   #
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return x_t, noise




    def train(self):

        start_train = time.time()
        loss_meter = AverageMeter()
        iter_meter = AverageMeter()

        self.fixed_eval_batch.to(self.device)


        for epoch in range(self.n_epochs):

            for i, (data, _ ) in enumerate(self.train_loader):

                self.net.train()
                start = time.time()

                data = data.to(self.device)
                data = 2 * data - 1 # normalize between [-1,1]
                self.batch_size = data.size(0)

               #############################################################################
                # TODO:                                                                    #
                # 1. Sample random timesteps for batch                                     #
                # 2. Apply forward diffusion process:                                      #
                #    - Get noisy samples                                                   #
                #    - Keep track of added noise                                           #
                # 3. Predict noise using current model                                     #
                # 4. Update model parameters using prediction error                        #
                #############################################################################
                loss, out, noise = None, None, None # use this variables to store loss, predicted noise and noise.
                # 1. sample random timesteps for batch
                t = torch.randint(0, self.timesteps, (self.batch_size,), device=self.device)

                # 2. apply forward diffusion
                x_t, noise = self.forward_diffusion(data, t)

                # 3. predict noise using current model
                predicted_noise = self.net(x_t, t)

                # 4. compute MSE loss
                loss = self.criterion(predicted_noise, noise)

                # 5. backward + optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                out = predicted_noise
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################


                loss_meter.update(loss.item(), out.size(0))
                iter_meter.update(time.time()-start)

            if epoch % 5 == 0:
                self.visualize_forward_diffusion(data, epoch=epoch)
                self.visualize_reverse_diffusion(epoch=epoch)
                print(
                    f'Epoch: [{epoch}][{i}/{len(self.train_loader)}]\t'
                    f'Loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})\t'
                    f'Time {iter_meter.val:.3f} ({iter_meter.avg:.3f})\t'
                    )


        self.visualize_reverse_diffusion(epoch=epoch)
        print(f"Completed in {(time.time()-start_train):.3f}")
        self.save_model(self.net, f'{self.output_dir}/diffusion_net_{self.dataset.lower()}.pth')


    @torch.no_grad()
    def sample_timestep(self, x, t):
        x_prev = None
        #############################################################################
        # TODO:                                                                     #
        # 1. Use model to predict noise in current sample given timestep t          #
        # 2. Get schedule parameters for current timestep t                         #
        # 3. Compute mean for reverse transition:                                   #
        #    a. Calculate coefficients using schedule parameters                    #
        #    b. Combine current sample and predicted noise                          #
        # 4. Add variance term if t>0, otherwise return mean                        #
        #############################################################################
    @torch.no_grad()
    def sample_timestep(self, x, t):
        batch_size = x.size(0)
        t_flat = t.view(-1)

        alpha_t     = self.alpha[t_flat].view(batch_size, 1, 1, 1)
        alpha_bar_t = self.alphas_bar[t_flat].view(batch_size, 1, 1, 1)
        beta_t      = self.beta[t_flat].view(batch_size, 1, 1, 1)

        # 1. predict noise
        predicted_noise = self.net(x, t)

        # 2. compute mean
        coef1 = 1 / torch.sqrt(alpha_t)
        coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
        mean  = coef1 * (x - coef2 * predicted_noise)

        # 3. add variance only when t > 0
        noise   = torch.randn_like(x)
        sigma_t = torch.sqrt(beta_t)
        x_prev  = mean + sigma_t * noise * (t_flat > 0).float().view(-1, 1, 1, 1)

        return x_prev

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return x_prev


    @torch.no_grad()
    def sample(self, epoch, x, save_freq=500):

        self.net.eval()

        #############################################################################
        # TODO:                                                                     #
        # 1. compute the sampling using the sample_timestep you defined earlier.
        # 2. consider the order in which we iterate over timesteps. 
        # 3. reference the DDPM algorithm for more info.
        # 4. This function is completed for learning purposes and is not used.
        #############################################################################
        x_current = x.clone()
        for i in reversed(range(self.timesteps)):
            t = torch.full((x_current.size(0),), i, device=self.device, dtype=torch.long)
            x_current = self.sample_timestep(x_current, t)
        normalized_x = (x_current.clone() + 1) / 2
        return normalized_x
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        normalized_x = (x.clone() + 1) / 2
        return normalized_x

    @torch.no_grad()
    def generate(self, n):
        self.batch_size = 16
        x = torch.randn((n, 1, self.height, self.width), device=self.device)
        saved_images = []
        for img_idx in range(0, x.size(0), self.batch_size):
            x_current = x[img_idx:img_idx+self.batch_size, :, :, :]
            for i in reversed(range(self.timesteps)):
                t = torch.full((x_current.size(0),), i, device=self.device, dtype=torch.long)
                x_current = self.sample_timestep(x_current, t)

            normalized_x = (x_current.clone() + 1) / 2
            saved_images.extend(normalized_x)
            
        saved_images = torch.stack(saved_images)
        return saved_images


    @torch.no_grad()
    def visualize_forward_diffusion(self, data, epoch, steps_to_plot=None, max_n=8):
        if steps_to_plot is None:
            steps_to_plot = torch.linspace(0, self.timesteps-1, steps=50, dtype=torch.int32, device=self.device)
        
        x0 = data[:max_n].to(self.device)
        self.batch_size = x0.size(0)
        
        timestep_images = []
        for t_int in steps_to_plot:
            t = torch.full((self.batch_size,), t_int, device=self.device, dtype=torch.long)
            x_t, _ = self.forward_diffusion(x0, t)
            x_t = (x_t + 1) / 2  # Normalize to [0,1]
            timestep_images.append(x_t)
        
        # Stack temporally to get [T, B, C, H, W]
        timestep_images = torch.stack(timestep_images)
        
        # transpose to get [B, T, C, H, W] then reshape [B*T, C, H, W]
        grid = timestep_images.transpose(0, 1).reshape(-1, *x0.shape[1:])
        
        vutils.save_image(
            grid,
            f"{self.output_dir}/epoch{epoch}_forward_diffusion_steps.png",
            nrow=len(steps_to_plot),
            padding=2
        )

    @torch.no_grad()
    def visualize_reverse_diffusion(self, x=None, epoch=0, max_n=8):
        

        if x is None:
            x = torch.randn((max_n, 1, self.height, self.width), device=self.device)
        else:
            x = x[:max_n].to(self.device)
        
        self.batch_size = x.size(0)
        x_current = x.clone()

        save_steps = torch.linspace(self.timesteps-1, 0, steps=50, dtype=torch.int32, device=self.device)
        saved_images = []
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((self.batch_size,), i, device=self.device, dtype=torch.long).unsqueeze(1)
            x_current = self.sample_timestep(x_current, t)
            
            if i in save_steps:
                normalized_x = (x_current.clone() + 1) / 2
                saved_images.append(normalized_x)
        
        # Stack temporally to get [T, B, C, H, W]
        saved_images = torch.stack(saved_images)
        
        # transpose to get [B, T, C, H, W] then reshape [B*T, C, H, W]
        grid = saved_images.transpose(0, 1).reshape(-1, *x.shape[1:])
        
        vutils.save_image(
            grid,
            f"{self.output_dir}/reverse_diffusion_epoch_{epoch}.png",
            nrow=len(save_steps),
            padding=2
        )
