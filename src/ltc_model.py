# -*- coding: utf-8 -*-
"""
ltc_model.py: Definition of the Liquid Time-Constant (LTC) Network / Liquid Neural Network (LNN).
Reference: Hasani et al., Liquid Time-constant Networks, AAAI 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LTCCell(nn.Module):
    """
    Liquid Time-Constant (LTC) neuron cell with explicit Euler numerical discretization.
    
    Governing ODE:
        dh/dt = -[1/tau + f(x, h)] * h + f(x, h) * A
        where:
            f(x, h) = sigmoid(W_x * x + W_h * h + b)
            tau = softplus(raw_tau) + 1e-3  (learnable unit-specific time constant)
            A is a learnable bias vector
    
    Discretized update rule (fused explicit forward Euler):
        h_{k+1} = (h_k + dt * f * A) / (1.0 + dt * (1/tau + f))
    """
    def __init__(self, in_dim=6, hidden=112, out_dim=2, steps=5, dt=1.0, learnable_tau=True):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.steps = steps
        self.dt = dt
        
        # Linear projection from input space to hidden state (constant external drive)
        self.W_x = nn.Linear(in_dim, hidden, bias=False)
        # Recurrent connections
        self.W_h = nn.Linear(hidden, hidden, bias=False)
        # Hidden bias
        self.b = nn.Parameter(torch.zeros(hidden))
        # Parameterized time-constant (softplus(0.5413) ≈ 1.0)
        self.raw_tau = nn.Parameter(torch.full((hidden,), 0.5413), requires_grad=learnable_tau)
        # Target activation vector A
        self.A = nn.Parameter(torch.ones(hidden))
        # Linear readout layer
        self.out = nn.Linear(hidden, out_dim)

    def tau(self):
        """Returns the strictly positive effective time constants."""
        return F.softplus(self.raw_tau) + 1e-3

    def forward(self, x, return_trajectory=False):
        """
        Forward propagation across numerical ODE integration steps.
        
        Parameters:
            x (torch.Tensor): Input batch of shape (batch_size, in_dim)
            return_trajectory (bool): Whether to return all intermediate hidden states
            
        Returns:
            logits (torch.Tensor): Shape (batch_size, out_dim)
            trajectory (list of torch.Tensor, optional): If return_trajectory is True
        """
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden, device=x.device)
        tau = self.tau()
        
        trajectory = [h] if return_trajectory else None
        
        for _ in range(self.steps):
            f = torch.sigmoid(self.W_x(x) + self.W_h(h) + self.b)
            h = (h + self.dt * f * self.A) / (1.0 + self.dt * (1.0 / tau + f))
            if return_trajectory:
                trajectory.append(h)
                
        logits = self.out(h)
        if return_trajectory:
            return logits, trajectory
        return logits
