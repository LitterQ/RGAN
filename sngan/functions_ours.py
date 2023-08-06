import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from imageio import imsave
from tqdm import tqdm
from copy import deepcopy
import logging

from utils.inception_score import get_inception_score
from utils.fid_score import calculate_fid_given_paths


logger = logging.getLogger(__name__)
EPSILON = 0.01
EPSILON_dis = 4


def get_normalized_vector(d):
    #d /= (1e-12 + tf.reduce_max(tf.abs(d), 1, keep_dims=True))
    #a = torch.max(torch.abs(d), 1, keepdim=True)
    #print(a.shape)
    d /= (1e-12 + torch.max(torch.abs(d), 1, keepdim=True)[0])
    d /= torch.sqrt(1e-6 + torch.sum(torch.pow(d, 2.0), 1, keepdim=True))
    return d

def generate_adversarial_perturbation(gen_net,dis_net, x, real_validity):
    logit_g = gen_net(x)
    logit_d = dis_net(logit_g)
    #loss = torch.mean(nn.ReLU(inplace=False)(1.0 - real_validity)) + torch.mean(nn.ReLU(inplace=False)(1 + logit_d))
    loss = -torch.mean(logit_d)


    gen_net.zero_grad()
    dis_net.zero_grad()
    loss.backward(retain_graph=True)
    grad = x.grad.data.detach()
    d = get_normalized_vector(grad)
    return EPSILON * d

def adversarial_loss(gen_net, dis_net, x, real_validity):
    r_adv = generate_adversarial_perturbation(gen_net, dis_net, x, real_validity)
    logit_g = gen_net(x + r_adv)
    logit_d = dis_net(logit_g)
    #loss = torch.mean(nn.ReLU(inplace=False)(1.0 - real_validity)) + torch.mean(nn.ReLU(inplace=False)(1 + logit_d))
    loss = -torch.mean(logit_d)
    return loss, r_adv

def generate_adversarial_perturbation_dis(gen_net, dis_net, x,real):
    logit_d = dis_net(x)
    logit_real = dis_net(real)
    loss = torch.mean(nn.ReLU(inplace=False)(1.0 - logit_real)) + torch.mean(nn.ReLU(inplace=False)(1 + logit_d))

    gen_net.zero_grad()
    dis_net.zero_grad()
    loss.backward(retain_graph=True)
    grad = x.grad.data.detach()
    #grad_real = tf.gradients(loss, [real], aggregation_method=2)[0]
    grad_real = real.grad.data.detach()
    #grad = tf.stop_gradient(grad)
    #grad_real = tf.stop_gradient(grad_real)
    return EPSILON_dis * get_normalized_vector(grad), EPSILON_dis * get_normalized_vector(grad_real)

def adversarial_loss_dis(gen_net, dis_net, x,real):
    r_adv,r_real = generate_adversarial_perturbation_dis(gen_net, dis_net, x, real)
    logit_d = dis_net(x+r_adv)
    logit_d_real = dis_net(real+r_real)
    loss = torch.mean(nn.ReLU(inplace=False)(1.0 - logit_d_real)) + torch.mean(nn.ReLU(inplace=False)(1 + logit_d))
    #loss = tf.reduce_mean(logit_d) - tf.reduce_mean(logit_d_real)
    #loss = -loss / gamma
    return loss,r_adv

def train(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch,
          writer_dict, schedulers=None):
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()

    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)
        real_imgs.requires_grad_()

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))
        z.requires_grad_()
        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        fake_imgs.requires_grad_()
        assert fake_imgs.size() == real_imgs.size()

        fake_validity = dis_net(fake_imgs)

        ad_loss, r_adv = adversarial_loss(gen_net, dis_net, z, real_validity)

        result_noise = gen_net(z + r_adv).detach()
        result_noise.requires_grad_()
        dis_noise_fake = dis_net(result_noise)
        disc_cost_noise = torch.mean(nn.ReLU(inplace=False)(1.0 - real_validity)) + \
                          torch.mean(nn.ReLU(inplace=False)(1 + dis_noise_fake))

        #ad_loss_gen, r_adv_gen = adversarial_loss(z_gen, gamma, reuse=True)

        ad_loss_dis, dis_adv_fake = adversarial_loss_dis(gen_net, dis_net, fake_imgs, real_imgs)
        ad_loss_dis_noise, dis_adv = adversarial_loss_dis(gen_net, dis_net, result_noise, real_imgs)

        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=False)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=False)(1 + fake_validity)) + disc_cost_noise + \
                 0.001 * ad_loss_dis + 0.001 * ad_loss_dis_noise
        d_loss.backward(retain_graph=True)
        dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), global_steps)

        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            gen_imgs = gen_net(gen_z)
            gen_z.requires_grad_()
            fake_validity = dis_net(gen_imgs)
            ad_loss_gen, r_adv_gen = adversarial_loss(gen_net, dis_net, gen_z, real_validity)

            # cal loss
            g_loss = -torch.mean(fake_validity) * 1.002 + ad_loss_gen * 1.002
            g_loss.backward(retain_graph=True)
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1


def validate(args, fixed_z, fid_stat, gen_net: nn.Module, writer_dict):
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']

    # eval mode
    gen_net = gen_net.eval()

    # generate images
    sample_imgs = gen_net(fixed_z)
    img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)

    # get fid and inception score
    fid_buffer_dir = os.path.join(args.path_helper['sample_path'], 'fid_buffer')
    os.makedirs(fid_buffer_dir)

    eval_iter = args.num_eval_imgs // args.eval_batch_size
    img_list = list()
    for iter_idx in tqdm(range(eval_iter), desc='sample images'):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

        # Generate a batch of images
        gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        for img_idx, img in enumerate(gen_imgs):
            file_name = os.path.join(fid_buffer_dir, f'iter{iter_idx}_b{img_idx}.png')
            imsave(file_name, img)
        img_list.extend(list(gen_imgs))

    # get inception score
    logger.info('=> calculate inception score')
    mean, std = get_inception_score(img_list)

    # get fid score
    logger.info('=> calculate fid score')
    fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None)

    os.system('rm -r {}'.format(fid_buffer_dir))

    writer.add_image('sampled_images', img_grid, global_steps)
    writer.add_scalar('Inception_score/mean', mean, global_steps)
    writer.add_scalar('Inception_score/std', std, global_steps)
    writer.add_scalar('FID_score', fid_score, global_steps)

    writer_dict['valid_global_steps'] = global_steps + 1

    return mean, fid_score


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten
