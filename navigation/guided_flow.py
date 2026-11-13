"""Experimental batched Flow sampling with decoded-trajectory soft guidance.

No planner replacement or hardware release is implied. It can still output an
unsafe trajectory; a separately validated hard checker/stop path is required.
"""
import torch


def decode(model,state):
    if model.latent:return model.vae.decode(state.flatten(1)*model.latent_std+model.latent_mean)
    return state*model.scale+model.center


def sample_guided(model,environment,goal_condition,candidates=16,steps=5,
                  cost=None,strength=.1,max_gradient_norm=1.,generator=None):
    """Share one perception result; execute one batch per ODE step, not total.

    Evaluate cost on a one-step endpoint estimate; detach the velocity field
    and differentiate only through the trajectory decoder and cost. Network
    parameters receive no optimizer updates or accumulated gradients.
    """
    if len(environment)!=1 or len(goal_condition) not in (1,candidates):raise ValueError('Expected one observation and one or K goal conditions')
    if candidates<1 or steps<1:raise ValueError('Expected positive candidates and steps')
    device=environment.device;env=environment.expand(candidates,-1,-1);goal=goal_condition.expand(candidates,-1)
    state=torch.randn(candidates,model.n,model.ch,device=device,dtype=environment.dtype,generator=generator)
    for k in range(steps):
        fraction=k/steps
        with torch.no_grad():velocity=model.velocity(state,torch.full((candidates,),fraction,device=device,dtype=environment.dtype),env,goal)
        if cost is not None and strength>0:
            with torch.enable_grad():
                current=state.detach().requires_grad_(True)
                endpoint=decode(model,current+(1-fraction)*velocity.detach())
                gradient=torch.autograd.grad(cost(endpoint).sum(),current)[0]
                gradient=torch.nan_to_num(gradient)
                norm=torch.linalg.vector_norm(gradient.flatten(1),dim=1).clamp_min(1e-9)
                gradient=gradient*torch.clamp(max_gradient_norm/norm,max=1.)[:,None,None]
            state=(state+velocity/steps-strength*gradient/steps).detach()
        else:state=(state+velocity/steps).detach()
    with torch.no_grad():
        trajectory=decode(model,state)
        angle=trajectory[:,:,2:];norm=torch.linalg.vector_norm(angle,dim=-1,keepdim=True)
        heading=torch.where(norm>1e-6,angle/norm.clamp_min(1e-6),angle.new_tensor([0.,1.]))
        trajectory=torch.cat([trajectory[:,:,:2],heading],-1)
        scores=cost(trajectory) if cost is not None else trajectory.new_zeros(candidates)
    return trajectory,scores
