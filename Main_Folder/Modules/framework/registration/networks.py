# network objects for registration
from torch import nn
import torch.nn.functional as F
import torch
import logging
from typing import Any



class HomographyNet(nn.Module):
    
    def __init__(self, device: torch.device | None = None):
        '''
        imposing the parameters for the matrix of affine trasformation as some of a nn, 
        whose layer depends on the matrix expo respecting the resolution level
        
        PARAMETERS:
        -----------

        device : torch.device
            the device to use for the model, default is cuda if available, otherwise cpu
        B : torch.Tensor
            the basis of generator of the affine transformation, it is a tensor of shape (6, 3, 3) 
            where each slice B[i] is a base generator
        v1 : torch.nn.Parameter
            the parameters for the affine transformation at the highest resolution level, it is a tensor of shape
            (6, 1, 1) and it is initialized as zeros, it is a learnable parameter that will be updated during training
        vL : torch.nn.Parameter
            the parameters for the affine transformation at the lower resolution levels, it is a tensor of shape
            (6, 1, 1) and it is initialized as zeros, it is a learnable parameter that will be updated during training

        FORWARD INPUT: 
        --------------

        s : int
            the resolution level, it is an integer that can take values from 0 to 6, 
            where 0 is the highest resolution level and 6 is the lowest resolution level, it is used to decide which parameters to use for the affine transformation


        RETURNS:
        --------

        H : torch.Tensor
            the homography matrix, it is a tensor of shape (3, 3)
            obtained by matrix exponentiation of C

        '''
        super(HomographyNet, self).__init__()
        # affine transform basis matrices
        self.logs = logging.getLogger(f'{__name__}.{type(self).__name__}')
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.B = torch.zeros(6,3,3).to(self.device)
        self.B[0,0,2] = 1.0
        self.B[1,1,2] = 1.0
        self.B[2,0,1] = 1.0 
        self.B[3,1,0] = 1.0 
        self.B[4,0,0], self.B[4,1,1] = 1.0, -1.0
        self.B[5,1,1], self.B[5,2,2] = -1.0, 1.0

        #inizializzatione  parametri - base for initialize the affine matrix transformation as ones for affine
        self.v1 = torch.nn.Parameter(torch.zeros(6,1,1).to(device), requires_grad=True)
        self.vL = torch.nn.Parameter(torch.zeros(6,1,1).to(device), requires_grad=True)

    def get_param_groups(self,
                         lr : float | dict[str, float],
                         )->list[dict[str, Any]]:
        '''
        Get the list of dicts to extract parameter and their lr

        Args:
            lr (float | dict[str, float]): the lr for model.parameters if it's a float, or for specific parameters otherwise

        Returns:
            list[dict[str, Any]]: _description_
        '''
        
        if isinstance(lr, (float, int)):
            self.logs.info(f'using the list of parameters: {list(dict(self.named_parameters()).keys())} with lr: {lr}')
            return [{'params' : self.parameters(), 'lr': float(lr)}]
        else:
            params_dict =  dict(self.named_parameters())
            param_groups = []
            for name, lr_val in lr.items():

                if not name in params_dict:
                    self.logs.error(f"Attribute {name} not found in {type(self).__name__}")
                    continue

                param_groups.append({'params': [params_dict[name]], 'lr': lr_val})
                
            if not param_groups:
                self.logs.warning(f'No valid parameters found in the provided lr dictionary: {lr}')
        return param_groups

    def forward(self, s): # depending the resolution the matrix expo define the approximation for the homography matrix
        C = torch.sum(self.B*self.vL,0)
        if s==0: # caso in cui ci si trovi nel più alto livello di risoluzione vine eaggiunto un parametro correttivo v1
            C += torch.sum(self.B*self.v1,0) 
        A = torch.eye(3).to(self.device)
        H = A
        for i in torch.arange(1,10): # prova torch.matrixexp(C)
            A = torch.mm(A/i,C) 
            H = H + A
        return H
    


#===============
#    METRICS
#===============

class MINE(nn.Module): #https://arxiv.org/abs/1801.04062
    def __init__(self, nChannel: int, n_neurons:int=100):
        
        '''
        Creation of MINE net for calculating Mutual Information lower bound: Ml_lb
        - fc1 1st layer for blended rappresentation of reference and test
            (it gets the doubled inputs of reference and test image color channel)
        -fc2 2nd layer increases non linearity of the nn: elaboration single hidden layer
        -fc3 3rd layer elaborate a rapresentative number for each coupled (x,y): it performa calculation for mutual information


        Args:
            nChannel (int): number of color channels
            n_neurons (int): number of neurons in the hidden layers
        '''
        super(MINE, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logs = logging.getLogger(f'{__name__}.{type(self).__name__}')
        self.nChannel = nChannel
        self.fc1 = nn.Linear(2*self.nChannel, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, 1)
        self.bsize = 1 #NOTE  number of permutation for assesting MI_lb, (may be sufficient)
    
    def get_param_groups(self,
                         lr : float | dict[str, float],
                         )->list[dict[str, Any]]:
        '''
        Get the list of dicts to extract parameter and their lr

        Args:
            lr (float | dict[str, float]): the lr for model.parameters if it's a float, or for specific parameters otherwise

        Returns:
            list[dict[str, Any]]: _description_
        '''
        
        if isinstance(lr, (float, int)):
            self.logs.info(f'using the list of parameters: {list(dict(self.named_parameters()).keys())} with lr: {lr}')
            return [{'params' : self.parameters(), 'lr': float(lr)}]
        else:
            params_dict =  dict(self.named_parameters())
            param_groups = []
            for name, lr_val in lr.items():

                if not name in params_dict:
                    self.logs.error(f"Attribute {name} not found in {type(self).__name__}")
                    continue

                param_groups.append({'params': [params_dict[name]], 'lr': lr_val})
                
            if not param_groups:
                self.logs.warning(f'No valid parameters found in the provided lr dictionary: {lr}')
        return param_groups


    def forward(self, x, ind):
        '''
        Calculate a T_score(x, y) for each bsize(permutation)
        z1: representing the matching couple scored: T(x_i, y_i) expecting to be high
        z2: representing the unmatching couple scored: T(x_i, y_j) expecting to be low when i!=j indeeed (log(exp(t)) t->0 --> 1 )


        Args:
            x (torch.Tensor): the input tensor of shape (H, W, 2*nChannels) where nChannel is the number of color channels, H and W are the height and width of the images
            ind (torch.Tensor): the indices available for the coupled images: (HxWxS) where S is the sampling ratio (0:1)
                                if u want to modify the batch of samples u have to modify this tensor: **they are the batch of point used for metric calculation**

        Returns:
            Ml_lb:  Mutual information lower bound
            z1, z2:  for visualization purpose
        '''
        x = x.view(x.size()[0]*x.size()[1],x.size()[2]) # FIXME change in x = x.reshape(-1, 2*self.nChannel) or equivalently with x=x.contiguous().view(x.size()[0]*x.size()[1],x.size()[2])
        MI_lb=0.0
        for i in range(self.bsize):
            ind_perm = ind[torch.randperm(len(ind), device= ind.device)]
            while torch.equal(ind_perm, ind): #NOTE loop while aggiunto, prova ablation study con questo loop
                perm = ind[torch.randperm(len(ind), device=ind.device)]
                ind_perm = ind[perm]

            z1 = self.fc3(F.relu(self.fc2(F.relu(self.fc1(x[ind,:])))))
            z2 = self.fc3(F.relu(self.fc2(F.relu(self.fc1(torch.cat((x[ind,0:self.nChannel],x[ind_perm,self.nChannel:2*self.nChannel]),1))))))
            MI_lb += torch.mean(z1) - torch.log(torch.mean(torch.exp(z2)))

        return MI_lb/self.bsize, z1, z2