import os
import time
import torch
from utils import gaussian_map,EdgeLoss,generate_fp,load_psfs_projections,Adjust_lr,Save_temp
from model import NeRF
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from einops import rearrange

######## Options #########
device = torch.device("cuda:1")
notes = 'sigma8_dz0.11_dxy0.11_pad64'

lr_init = 2e-5
lr_decay = 0.3
decay_start = 80000
decay_every = 20000
iters = int(decay_start+decay_every*2.5)

sigma = 8
edge_weight = 1

dxy = 110 #nm
dz = 110 #nm

sample_z = 1
padding = 64
log_iter = 200

PSF_name = 'PSF_1P_z149_xy793_uv13'
projection_name = '1500_793.tif'
#############################

# Set Params
PSF_dir = os.path.join('/home/zjy/nerf/PSF',PSF_name)
Projection_path = os.path.join('/home/zjy/nerf/Projections',projection_name)
psfs_path = os.listdir(PSF_dir)

curtime = time.strftime('%m_%d_%H_%M',time.localtime(time.time()))
log_dir = os.path.join('/home/zjy/nerf/log',curtime+'_'+notes+'_'+projection_name[:-4])
save_dir = os.path.join('/home/zjy/nerf/Results',curtime+'_'+notes+'_'+projection_name[:-4])
writer = SummaryWriter(log_dir)
os.mkdir(save_dir)

# Load data
radius = 3
psfs,projections = load_psfs_projections(PSF_dir,Projection_path,radius,padding,device)
uv_num,z_res,x,y = psfs.size()
_,xy_res,_ = projections.size()
psf_padding = int((xy_res-x)/2+padding)
xy_res = xy_res+2*padding

# Prepare for training
interval_xy = [0,1]
interval_z = [0,(z_res*dz)/(xy_res*dxy)]
iter_z = z_res-sample_z+1
embedded_ch = 256
loss_sum = 0

model = NeRF(input_ch=embedded_ch,D=6,skips=[3])
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, betas=(0.9, 0.999))

mse = torch.nn.L1Loss()
edge_loss = EdgeLoss()

B = torch.randn(3,int(embedded_ch/2)).to(device)*sigma
large_xguess = torch.zeros((z_res,xy_res,xy_res)).to(device)

xy_range = torch.linspace(interval_xy[0], interval_xy[1], xy_res+1)[:-1].to(device)
z_range = torch.linspace(interval_z[0], interval_z[1], z_res+1)[:-1].to(device)

# Iteration
for iter in range(iters):

    optimizer.zero_grad()

    z_index = iter%iter_z

    x_sample = xy_range
    y_sample = xy_range
    z_sample = z_range[z_index:z_index+sample_z]

    z,x,y = torch.meshgrid((z_sample,x_sample,y_sample))
    zxy = torch.stack((z,x,y),dim=-1)
    coords = rearrange(zxy, 'z x y c -> (z x y) c')
    embedded = gaussian_map(coords,B)

    # Run NeRF
    intensity = model(embedded)
    xguess_part = rearrange(intensity,'(z x y) c -> z x y c',z=sample_z,x=xy_res,y=xy_res)
    xguess_part = torch.squeeze(xguess_part)

    # Projection 
    rand = int(iter/iter_z)%int(uv_num)
    psf = F.pad(psfs[rand,:,:,:].to(device),(psf_padding,psf_padding,psf_padding,psf_padding,0,0),'constant')
    psf_slices = psf[z_index:z_index+sample_z,:,:]
    large_xguess[z_index:z_index+sample_z,:,:] = xguess_part

    fp_slices = generate_fp(psf_slices,large_xguess[z_index:z_index+sample_z,:,:],device)

    if z_index == 0:
        fp = generate_fp(psf,large_xguess,device)

    fp[z_index:z_index+sample_z,:,:] = fp_slices

    large_simulation = torch.sum(fp,dim=0)
    simulation = large_simulation[padding:xy_res-padding,padding:xy_res-padding]
    projection = projections[rand,:,:]
    
    # Calculate Loss
    loss = mse(simulation,projection)
    eloss = edge_loss(simulation,projection)

    loss_total = loss+eloss*edge_weight
    loss_total.backward()

    optimizer.step()

    large_xguess.detach_()
    fp.detach_()

    # Adjust learning rate
    if ((iter+1) % decay_every == 0) and ((iter+1) >= decay_start):
        optimizer = Adjust_lr(optimizer,iter,decay_every,lr_decay,lr_init)

    # Show & Save
    loss_sum += loss.cpu().item()
    print('Iters:',iter,'Index:',z_index,rand,' Loss:',loss.cpu().item(),' Edge Loss:',eloss.cpu().item())

    if (iter+1) % log_iter == 0:
        xguess = large_xguess[:,padding:xy_res-padding,padding:xy_res-padding]
        loss_sum = Save_temp(xguess,simulation,projection,writer,save_dir,notes,loss_sum,log_iter,iter,iters)