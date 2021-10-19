import numpy as np
import torch
import scipy.optimize
from utils import *
import pdb

class bcolors:
    MAGENTA = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    YELLOW = '\033[93m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RED = '\033[91m'
    WHITE = '\033[97m'
    BLACK = '\033[90m'
    DEFAULT = '\033[99m'


def conjugate_gradients(Avp_f, b, nsteps, rdotr_tol=1e-10):
    x = zeros(b.size(), device=b.device)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        Avp = Avp_f(p)
        alpha = rdotr / torch.dot(p, Avp)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < rdotr_tol:
            break
    return x


def line_search(model, f, x, fullstep, expected_improve_full, max_backtracks=10, accept_ratio=0.1):
    fval = f(True).item()

    for stepfrac in [.5**x for x in range(max_backtracks)]:
        x_new = x + stepfrac * fullstep
        set_flat_params_to(model, x_new)
        fval_new = f(True).item()
        actual_improve = fval - fval_new
        expected_improve = expected_improve_full * stepfrac
        ratio = actual_improve / expected_improve

        if ratio > accept_ratio:
            return True, x_new
    return False, x


def cpo_step(env_name, policy_net, value_net, states, actions, returns, advantages, cost_advantages, constraint_value, d_k, max_kl, damping, l2_reg, use_fim=True):

    """update critic"""

    def get_value_loss(flat_params):
        set_flat_params_to(value_net, tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)
        values_pred = value_net(states)
        value_loss = (values_pred - returns).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        value_loss.backward()
        return value_loss.item(), get_flat_grad_from(value_net.parameters()).cpu().numpy()
    
    #pdb.set_trace()
    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss,
                                                            get_flat_params_from(value_net).detach().cpu().numpy(),
                                                            maxiter=25)
    v_loss,_ = get_value_loss(get_flat_params_from(value_net).detach().cpu().numpy())
    set_flat_params_to(value_net, tensor(flat_params))

    """update policy"""
    with torch.no_grad():
        fixed_log_probs = policy_net.get_log_prob(states, actions)
    """define the loss function for Objective"""
    def get_loss(volatile=False):
        with torch.set_grad_enabled(not volatile):
            log_probs = policy_net.get_log_prob(states, actions)
            action_loss = -advantages * torch.exp(log_probs - fixed_log_probs)
            return action_loss.mean()
        
    """define the loss function for Constraint"""
    def get_cost_loss(volatile=False):
        with torch.set_grad_enabled(not volatile):
            log_probs = policy_net.get_log_prob(states, actions)
            cost_loss = cost_advantages * torch.exp(log_probs - fixed_log_probs)
            return cost_loss.mean()      

    """use fisher information matrix for Hessian*vector"""
    def Fvp_fim(v):
        M, mu, info = policy_net.get_fim(states)
        #pdb.set_trace()
        mu = mu.view(-1)
        filter_input_ids = set() if policy_net.is_disc_action else set([info['std_id']])

        t = ones(mu.size(), requires_grad=True, device=mu.device)
        mu_t = (mu * t).sum()
        Jt = compute_flat_grad(mu_t, policy_net.parameters(), filter_input_ids=filter_input_ids, create_graph=True)
        Jtv = (Jt * v).sum()
        Jv = torch.autograd.grad(Jtv, t)[0]
        MJv = M * Jv.detach()
        mu_MJv = (MJv * mu).sum()
        JTMJv = compute_flat_grad(mu_MJv, policy_net.parameters(), filter_input_ids=filter_input_ids).detach()
        JTMJv /= states.shape[0]
        if not policy_net.is_disc_action:
            std_index = info['std_index']
            JTMJv[std_index: std_index + M.shape[0]] += 2 * v[std_index: std_index + M.shape[0]]
        return JTMJv + v * damping

    """directly compute Hessian*vector from KL"""
    def Fvp_direct(v):
        kl = policy_net.get_kl(states)
        kl = kl.mean()

        grads = torch.autograd.grad(kl, policy_net.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * v).sum()
        grads = torch.autograd.grad(kl_v, policy_net.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach()

        return flat_grad_grad_kl + v * damping
    
    #define f_a(\lambda) and f_b(\lambda)
    def f_a_lambda(lamda):
        a = ((r**2)/s - q)/(2*lamda)
        b = lamda*((cc**2)/s - max_kl)/2
        c = - (r*cc)/s
        return a+b+c
    
    def f_b_lambda(lamda):
        a = -(q/lamda + lamda*max_kl)/2
        return a   
    
    Fvp = Fvp_fim if use_fim else Fvp_direct
    
    # Obtain objective gradient and step direction
    loss = get_loss()
    grads = torch.autograd.grad(loss, policy_net.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach() #g  
    grad_norm = False
    if grad_norm == True:
        loss_grad = loss_grad/torch.norm(loss_grad)
    stepdir = conjugate_gradients(Fvp, -loss_grad, 10) #(H^-1)*g   
    if grad_norm == True:
        stepdir = stepdir/torch.norm(stepdir)
    
    #Obtain constraint gradient and step direction
    if env_name == "CartPole-v0" or env_name == "CartPole-v1" or env_name == "MountainCar-v0":
        agent_data = torch.cat([states, actions.unsqueeze(-1)], 1) #Arrange agent and expert data in required format 
    else:
        agent_data = torch.cat([states, actions], 1)
    
    cost_loss = get_cost_loss()
    #print('cost_loss', cost_loss)
    cost_grads = torch.autograd.grad(cost_loss, policy_net.parameters(), allow_unused=True)
    cost_loss_grad = torch.cat([grad.view(-1) for grad in cost_grads]).detach() #a
    cost_loss_grad = cost_loss_grad/torch.norm(cost_loss_grad)
    cost_stepdir = conjugate_gradients(Fvp, -cost_loss_grad, 10) #(H^-1)*a
    #print('PG norm {}\t cost norm {}'.format(torch.norm(stepdir).item(), torch.norm(cost_stepdir).item()))
    cost_stepdir = cost_stepdir/torch.norm(cost_stepdir)
    
    # Define q, r, s
    p = -cost_loss_grad.dot(stepdir) #a^T.H^-1.g
    q = -loss_grad.dot(stepdir) #g^T.H^-1.g
    r = loss_grad.dot(cost_stepdir) #g^T.H^-1.a
    s = -cost_loss_grad.dot(cost_stepdir) #a^T.H^-1.a 

    d_k = tensor(d_k).to(constraint_value.dtype).to(constraint_value.device)
    cc = constraint_value - d_k # c would be positive for most part of the training
    lamda = 2*max_kl
        
    #find optimal lambda_a and lambda_b
    A = torch.sqrt((q - (r**2)/s)/(max_kl - (cc**2)/s))
    B = torch.sqrt(q/max_kl)
    if cc>0:
        opt_lam_a = torch.max(r/cc,A)
        opt_lam_b = torch.max(0*A,torch.min(B,r/cc))
    else: 
        opt_lam_b = torch.max(r/cc,B)
        opt_lam_a = torch.max(0*A,torch.min(A,r/cc))
    
    #find values of optimal lambdas 
    opt_f_a = f_a_lambda(opt_lam_a)
    opt_f_b = f_b_lambda(opt_lam_b)
    
    if opt_f_a > opt_f_b:
        opt_lambda = opt_lam_a
    else:
        opt_lambda = opt_lam_b
            
    #find optimal nu
    nu = (opt_lambda*cc - r)/s
    if nu>0:
        opt_nu = nu 
    else:
        opt_nu = 0
        
    """ find optimal step direction """
    # check for feasibility
    if ((cc**2)/s - max_kl) > 0 and cc>0:
        print('INFEASIBLE !!!!')
        #opt_stepdir = -torch.sqrt(2*max_kl/s).unsqueeze(-1)*Fvp(cost_stepdir)
        opt_stepdir = torch.sqrt(2*max_kl/s)*Fvp(cost_stepdir)
    else: 
        #opt_grad = -(loss_grad + opt_nu*cost_loss_grad)/opt_lambda
        opt_stepdir = (stepdir - opt_nu*cost_stepdir)/opt_lambda
        #opt_stepdir = conjugate_gradients(Fvp, -opt_grad, 10)
    
    #print(f"{bcolors.OKBLUE} nu by lambda {opt_nu/opt_lambda},\t lambda {1/opt_lambda}{bcolors.ENDC}")
    """
    #find the maximum step length
    xhx = opt_stepdir.dot(Fvp(opt_stepdir))
    beta_1 = -cc/(cost_loss_grad.dot(opt_stepdir))
    beta_2 = torch.sqrt(max_kl / xhx)
    
    if beta_1 < beta_2:
        beta_star = beta_1
    else: 
        beta_star = beta_2
       
    # perform line search
    #fullstep = beta_star*opt_stepdir
    prev_params = get_flat_params_from(policy_net)
    fullstep = opt_stepdir
    expected_improve = -loss_grad.dot(fullstep)
    success, new_params = line_search(policy_net, get_loss, prev_params, fullstep, expected_improve)
    set_flat_params_to(policy_net, new_params)
    """
    # trying without line search
    prev_params = get_flat_params_from(policy_net)
    new_params = prev_params + opt_stepdir
    set_flat_params_to(policy_net, new_params)
    
    return v_loss, loss, cost_loss
