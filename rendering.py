import torch

def compute_accumulated_transmittance(betas):
    '''
    Computes T, the accumulated transmittance.
    
    Note: Term used in the return replaces the following code:
    accumulated_transmittance[:, 1:] = accumulated_transmittance[:, :-1] # effectively gets rid of the last value +
    # repeats the first value. i.e: 1,2,3,4 becomes 1,1,2,3
    accumulated_transmittance[:,0] = 1. # since the sum starts at j=1, the 0th terms shouldn't be included
    
    This is because the return way allows for gradient descent to work correctly in 
c_to_optimize = torch.tensor([0., 0., 0.], requires_grad=True) # starting with no color (0,0,0)
optimizer = torch.optim.SGD({c_to_optimize}, lr=1e-1)pytorch
    '''
    accumulated_transmittance = torch.cumprod(betas, 1) # this is another way to represent the math
    
    return torch.cat((torch.ones(accumulated_transmittance.shape[0], 1, device=accumulated_transmittance.device),
                      accumulated_transmittance[:, :-1]), dim=1)

    
def rendering(model, rays_o, rays_d, tn, tf, interp, nb_bins=100, device='cuda', white_bckgr=True):
    '''
    The rendering function estimates the color of a ray. Rather than directly computing the integral,
    we use an approximation.
    '''
    rays_o = torch.from_numpy(rays_o).reshape(-1, 3).type(torch.float)
    rays_d = torch.from_numpy(rays_d).reshape(-1, 3).type(torch.float)

    t = torch.linspace(tn,tf,nb_bins).to(device) # bounds over which we will compute the integral/sum for determining the color
    # we concatenate on the end a very large number to represent the last index, since we have one less than ttl
    delta = torch.cat((t[1:] - t[:-1], torch.tensor([1e10], device=device))) # t[1:] gives us the value of i+1 for each index. 
    # The other gives us t[i]
    
    # adding virtual dimensions so that the math works out to get the right format
    # [1, nb_rays, 1]
    # [nb_rays, 1, 3]
    x = rays_o.unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1) * rays_d.unsqueeze(1) # [nb_rays, nb_bins, 3]
    # x = torch.unsqueeze(rays_o,1) + torch.unsqueeze(torch.unsqueeze(t,0),-1) * torch.unsqueeze(rays_d,1) # [nb_rays, nb_bins, 3]

    x = x + torch.from_numpy(interp(x)) # FIXME: How should this be implemented?
    
    colors, density = model.intersect(x.reshape(-1,3), rays_d.expand(x.shape[1], x.shape[0], 3).transpose(0,1).reshape(-1,3))
    # [nb_rays, nb_bins, 3]
    # [nb_rays, nb_bins, 1]
    # putting the color and density back to the correct shape
    colors = colors.reshape((x.shape[0], nb_bins, 3)) # [nb_rays, nb_bins, 3]
    density = density.reshape((x.shape[0], nb_bins)) # [nb_rays, nb_bins]
    
    alpha = 1 - torch.exp(- density * delta.unsqueeze(0)) # [nb_rays, nb_bins, 1]
    weights = compute_accumulated_transmittance(1 - alpha) * alpha # adding unsqueeze gets us same shape as density
    
    if white_bckgr:
        c = (weights.unsqueeze(-1) * colors).sum(1) # [nb_rays, 3]
        weight_sum = weights.sum(-1) # [nb_rays] tells us if we are in empty space or not
        return c + 1 - weight_sum.unsqueeze(-1)
    else:
        c = (weights.unsqueeze(-1) * colors).sum(1) # [nb_rays, 3] sum sums all the bins (dim 1)
    
    return c