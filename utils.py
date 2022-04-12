import torch.nn.functional as F
from torch.fft import fft2,ifft2
import tifffile as tf
import torch.nn as nn
import numpy as np
import math
import torch
import os

def generate_fp(psf,sample,device):

    if psf.ndim == 2:
        z = 1 
        ra,ca = psf.shape
        rb,cb =sample.shape
    else:
        z,ra,ca = psf.shape
        z,rb,cb =sample.shape

    r = ra+rb
    c = ca+cb
    p1 = (r-ra)/2
    p2 = (c-ca)/2

    a1 = torch.zeros(r,c).to(device)
    b1 = torch.zeros(r,c).to(device)

    projections = []

    for i in range(z):
        a1[0:ra,0:ca] = psf[i,:,:]
        b1[0:rb,0:cb] = sample[i,:,:]
        conv1 = ifft2(fft2(a1)*fft2(b1))
        projections.append(torch.real(conv1[int(p1+0.5):int(r-p1+0.5),int(p2+0.5):int(c-p2+0.5)]))

    return torch.stack(projections,dim=0)

def gaussian_map(x,B):
    xp = torch.matmul(2*math.pi*x,B)
    return torch.cat([torch.sin(xp),torch.cos(xp)],dim=-1)

def edge_map_4d(im):

    im = torch.unsqueeze(im,dim=0)
    im = torch.unsqueeze(im,dim=0)

    b,c,h,w = im.size()

    # sharper scharr
    kernel_x = [[-10,0,10], [-101,0,101], [-10,0,10]]
    kernel_y = [[-10,-101,-10], [0,0,0], [10,101,10]]
    kernel_45 = [[101,10,0], [10,0,-10], [0,-10,-101]]
    kernel_135 = [[0,-10,-101], [10,0,-10], [101,10,0]]

    sobel_kernel_x = np.asarray(kernel_x,dtype='float32')
    sobel_kernel_x = sobel_kernel_x.reshape((1, 1, 3, 3))
    sobel_kernel_y = np.asarray(kernel_y,dtype='float32')
    sobel_kernel_y = sobel_kernel_y.reshape((1, 1, 3, 3))
    weight_x = torch.from_numpy(sobel_kernel_x).to(im)
    weight_y = torch.from_numpy(sobel_kernel_y).to(im)

    sobel_kernel_45 = np.asarray(kernel_45,dtype='float32')
    sobel_kernel_45 = sobel_kernel_45.reshape((1, 1, 3, 3))
    sobel_kernel_135 = np.asarray(kernel_135,dtype='float32')
    sobel_kernel_135 = sobel_kernel_135.reshape((1, 1, 3, 3))
    weight_45 = torch.from_numpy(sobel_kernel_45).to(im)
    weight_135 = torch.from_numpy(sobel_kernel_135).to(im)

    edge_map = torch.zeros(4,b,c,h,w).to(im)
    edge_map[0,:,:,:,:] = F.conv2d(im,weight_x,padding=1,stride=1,groups=c)
    edge_map[1,:,:,:,:] = F.conv2d(im,weight_y,padding=1,stride=1,groups=c)
    edge_map[2,:,:,:,:] = F.conv2d(im,weight_45,padding=1,stride=1,groups=c)
    edge_map[3,:,:,:,:] = F.conv2d(im,weight_135,padding=1,stride=1,groups=c)

    return edge_map

def edge_map_2d(im):

    im = torch.unsqueeze(im,dim=0)
    im = torch.unsqueeze(im,dim=0)

    b,c,h,w = im.size()

    # sharper scharr
    kernel_x = [[-10,0,10], [-101,0,101], [-10,0,10]]
    kernel_y = [[-10,-101,-10], [0,0,0], [10,101,10]]

    # scharr
    # kernel_x = [[-3,0,3], [-10,0,10], [-3,0,3]]
    # kernel_y = [[-3,-10,-3], [0,0,0], [3,10,3]]

    sobel_kernel_x = np.asarray(kernel_x,dtype='float32')
    sobel_kernel_x = sobel_kernel_x.reshape((1, 1, 3, 3))
    sobel_kernel_y = np.asarray(kernel_y,dtype='float32')
    sobel_kernel_y = sobel_kernel_y.reshape((1, 1, 3, 3))
    weight_x = torch.from_numpy(sobel_kernel_x).to(im)
    weight_y = torch.from_numpy(sobel_kernel_y).to(im)

    edge_map = torch.zeros(2,b,c,h,w).to(im)
    edge_map[0,:,:,:,:] = F.conv2d(im,weight_x,padding=1,stride=1,groups=c)
    edge_map[1,:,:,:,:] = F.conv2d(im,weight_y,padding=1,stride=1,groups=c)

    return edge_map

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x,y):
        sobel_x = edge_map_2d(x)
        sobel_y = edge_map_2d(y)

        loss_fn = nn.L1Loss()
        loss = loss_fn(sobel_x,sobel_y)

        return loss

def normal(input,refer=None):
    if refer == None:
        out = (input-torch.min(input))/(torch.max(input)-torch.min(input))
    else:
        out = (input-torch.min(refer))/(torch.max(refer)-torch.min(refer))
    return out

def normal_std(input):
    out = (input-torch.min(input))/(torch.max(input)-torch.min(input))
    mean = torch.mean(out)
    delta = torch.std(out)
    thresh = mean+3*delta
    out = (out-torch.min(out))/(thresh-torch.min(out))
    # print(torch.min(out),torch.max(out))
    return out

def load_int16(path):
    input = tf.imread(path)
    input = input.astype(np.float32)
    input = torch.from_numpy(input)
    input = input - 32768
    out = torch.maximum(input,torch.zeros_like(input))
    return out

def load_uint16(path):
    input = tf.imread(path)
    input = input.astype(np.float32)
    out = torch.from_numpy(input)
    return out
    
def PSNR(img1, img2):
    fn = torch.nn.MSELoss()
    mse = fn(img1,img2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

def load_psfs_projections(PSF_dir,Projection_path,radius,padding,device):

    projections_all = load_uint16(Projection_path).to(device)
    uv_res,xy_res,_ = projections_all.size()
    uv_res = int(np.power(uv_res,0.5))
    center_index = [(uv_res+1)/2,(uv_res+1)/2]

    psfs = []
    projections = []
    for i in range(1,int(uv_res*uv_res)):
        index = [i%uv_res,int(i/uv_res)+1]
        distance = np.power(np.power(index[0]-center_index[0],2)+np.power(index[1]-center_index[1],2),0.5)
        if distance <= radius:
            psf_name = 'psf_'+str(i+1)+'.tif'
            psf_path = os.path.join(PSF_dir,psf_name)
            psf = tf.imread(psf_path).astype(np.float32)
            psf = torch.from_numpy(psf)
            z_res,x,_ = psf.size()
            # psf_padding = int((xy_res-x)/2+padding)
            # psf = F.pad(psf,(psf_padding,psf_padding,psf_padding,psf_padding,0,0),'constant')
            psfs.append(psf)
            projections.append(projections_all[i,:,:])
            if distance == 0:
                projection_CA = F.pad(projections_all[i,:,:],(padding,padding,padding,padding),'constant')
            print('Load PSF:',psf_path,'PSF Range:',torch.min(psf).item(),torch.max(psf).item(),'Distance:',distance)
    uv_num = len(psfs)
    print('Projections Used:',uv_num)

    psfs = torch.squeeze(torch.stack(psfs,dim=0))
    projections = torch.stack(projections,dim=0)

    psf_max,_ = torch.max(psfs,dim=0)
    psf_max.to(device)
    projection_CA = projection_CA*(torch.max(psfs)/torch.max(projection_CA))
    projection_CA = projection_CA.repeat(z_res,1,1)
    Pmax = torch.max(torch.sum(generate_fp(psf_max,projection_CA,device),dim=0))

    projections = projections*(Pmax/torch.max(projections))
    print('Load Projections:',Projection_path,'Projections Range:',torch.min(projections).item(),torch.max(projections).item())

    return psfs,projections

def Adjust_lr(optimizer,iter,decay_every,lr_decay,lr_init):

    lr = lr_init * (lr_decay ** (iter // decay_every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return optimizer

def Save_temp(xguess,simulation,projection,writer,save_dir,notes,loss_sum,log_iter,iter,iters):

    avg_loss = loss_sum/log_iter
    writer.add_scalar('MSE Loss',avg_loss,iter)
    z_res,_,_ = xguess.size()
    loss_sum = 0

    with torch.no_grad():
        
        central_slice = torch.unsqueeze(torch.squeeze(xguess[int((z_res-1)/2),:,:]),dim=0)
        first_slice = torch.unsqueeze(torch.squeeze(xguess[0,:,:]),dim=0)

        central_slice = normal(central_slice)
        first_slice = normal(first_slice)
        display = normal(simulation)
        gt = normal(projection)
        
        writer.add_image('Projection',torch.unsqueeze(gt,dim=0),iter)
        writer.add_image('Simulation',torch.unsqueeze(display,dim=0),iter)
        writer.add_image('Central Slice',central_slice,iter)
        writer.add_image('First Slice',first_slice,iter)

        if (iter+1-log_iter) % 8000 == 0 or (iter+1) == iters:
            save_path = os.path.join(save_dir,notes+'_'+str(iter+1)+'.tif')
            xguess = torch.squeeze(xguess).cpu().numpy().astype(np.float16)
            tf.imsave(save_path,xguess)

    return loss_sum

def gen_xyindex(bin_size):

    if bin_size == 1: 
        rand_xy = [(0,0)]
    if bin_size == 2: 
        rand_xy = [(0,0),(0,1),(1,0),(1,1)]
    if bin_size == 3: 
        rand_xy = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]

    return rand_xy