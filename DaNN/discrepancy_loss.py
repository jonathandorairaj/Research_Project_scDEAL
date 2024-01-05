#!/usr/bin/env python
# encoding: utf-8

import torch

min_var_est = 1e-8

def discrepancy_loss(mu_src,log_var_src,mu_tar,log_var_tar):
    #std_src, mu_src = torch.std_mean(x_src, keepdim=True, dim=0)
    #std_tar, mu_tar = torch.std_mean(x_tar, keepdim=True, dim=0)
    
    mu_a = mu_src
    var_a = torch.diag(torch.exp(log_var_src))
    
    #print(f"mu_a: {mu_a}")
    #print(f"var_a: {var_a}")

    mu_b = mu_tar
    var_b = torch.diag(torch.exp(log_var_tar))

    #print(f"mu_b: {mu_b}")
    #print(f"var_b: {var_b}")
    #mu_a1 = mu_a.view(mu_a.size(0), 1, -1)
    #mu_a2 = mu_a.view(1, mu_a.size(0), -1)
    
    mu_a1 = mu_a.view(mu_a.size(0),1,-1)
    mu_a2 = mu_a.view(1,mu_a.size(0),-1)
    var_a1 = var_a.view(var_a.size(0),1,-1)
    var_a2 = var_a.view(1,var_a.size(0),-1)

    mu_b1 = mu_b.view(mu_b.size(0),1,-1)
    mu_b2 = mu_b.view(1,mu_b.size(0),-1)
    var_b1 = var_b.view(var_b.size(0),1,-1)
    var_b2 = var_b.view(1,var_b.size(0),-1)
    
    vaa = torch.sum(torch.div(torch.exp(torch.mul(torch.div(torch.pow(mu_a1-mu_a2,2),var_a1+var_a2),-0.5)),torch.sqrt(var_a1+var_a2)))
    vab = torch.sum(torch.div(torch.exp(torch.mul(torch.div(torch.pow(mu_a1-mu_b2,2),var_a1+var_b2),-0.5)),torch.sqrt(var_a1+var_b2)))
    vbb = torch.sum(torch.div(torch.exp(torch.mul(torch.div(torch.pow(mu_b1-mu_b2,2),var_b1+var_b2),-0.5)),torch.sqrt(var_b1+var_b2)))
    
    loss = vaa+vbb-torch.mul(vab,2.0)
    
    return loss