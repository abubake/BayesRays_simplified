from tqdm import tqdm
from rendering import rendering
import torch
import numpy as np

def training(model, optimizer, scheduler, tn, tf, nb_bins, nb_epochs,
              data_loader, model_save_path='nerfs/off_coral_model_training.pth', device='cpu'):
    '''
    Default NeRF training function. Takes in the model and all rays at once for offline learning;
    performs supervised learning, comparing predicted images to rendered images for comparison
    
    Returns the training loss, and generates a .pth model file.
    '''
    
    training_loss = []
    for epoch in (range(nb_epochs)):
        for batch in tqdm(data_loader):
            o = batch[:, :3].to(device)
            d = batch[:, 3:6].to(device)
            
            target = batch[:, 6:].to(device)
            
            # Find Mean Squared Error
            prediction = rendering(model, o, d, tn, tf, nb_bins=nb_bins, device=device)
            
            loss = ((prediction - target)**2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
            
        scheduler.step()
        
        torch.save(model, model_save_path)
        model.to(device)
        
    return training_loss

def training_with_test(model, optimizer, scheduler, tn, tf, nb_bins, nb_epochs, data_loader, device='cpu'):
    '''
    NeRF training function with testing built in to check psnr of rendered images at each epoch.
    Runs slower than regular training function.
    Currently not flexible for different image sizes, etc.
    
    Returns the training loss, and generates a .pth model file.
    '''
    
    training_loss = []
    for epoch in (range(nb_epochs)):
        for batch in tqdm(data_loader):
            o = batch[:, :3].to(device)
            d = batch[:, 3:6].to(device)
            
            target = batch[:, 6:].to(device)
            prediction = rendering(model, o, d, tn, tf, nb_bins=nb_bins, device=device) 
            loss = ((prediction - target)**2).mean()
            print(loss.shape)
            print(loss)
            
            optimizer.zero_grad()
            loss.backward() # computes gradients using the loss
            optimizer.step() # updates the values of the network parameters
            training_loss.append(loss.item())
        
        img,_,psnr = testing(
                            model=model, o=o, d=d,
                            tn=tn, tf=tf, nb_bins=nb_bins, chunk_size=10,
                            H=400, W=400,target=prediction,white_bckgr=True
                            )
        print("Training PSNR:",psnr)    
        scheduler.step()
        
    torch.save(model, 'model_nerf.pth')
    model.to(device)
        
    return training_loss

def online_training(model, optimizer, scheduler, tn, tf, nb_bins, cum_training_loss, data_loader, device='cpu'):

    for batch in tqdm(data_loader):
        o = batch[:, :3].to(device)
        d = batch[:, 3:6].to(device)
        
        target = batch[:, 6:].to(device)
        
        prediction = rendering(model, o, d, tn, tf, nb_bins=nb_bins, device=device)
        
        loss = ((prediction - target)**2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cum_training_loss.append(loss.item())
        
    scheduler.step()
    
    torch.save(model, 'nerfs/online_model_coral.pth')
    model.to(device)
        
    return cum_training_loss

def mse2psnr(mse):
    return 20 * np.log10(1 / np.sqrt(mse))

@torch.no_grad()
def testing(model, o, d, tn, tf, nb_bins=100, chunk_size=10, H=400, W=400, target=None, white_bckgr=True, interp=None):
    '''
    MODIFIED FOR INTERPOLATION
    Renders an image from the test dataset using the trained machine learning model.
    
    In: ...
    Returns: 
        - The rendered image as a variable if target=None
        - The rendered image, mse, and psnr of the image if target/ground truth image is provided
    '''
    
    o = o.chunk(chunk_size)
    d = d.chunk(chunk_size)
    
    img = []
    for o_batch, d_batch in zip(o,d):
        img_batch = rendering(model=model,rays_o=o_batch,rays_d=d_batch,tn=tn,tf=tf,interp=interp,
                              nb_bins=nb_bins,device=o_batch.device)
        img.append(img_batch) # [N,3]
        
    img = torch.cat(img)
    img = img.reshape(H,W,3).cpu().numpy()
    
    if target is not None:
        
        mse = ((img-target)**2).mean()
        
        return img, mse, mse2psnr(mse)
    
    else:
        return img
