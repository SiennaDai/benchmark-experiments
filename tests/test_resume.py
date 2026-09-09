import copy
import random
import numpy as np
import torch
from data.frozen_tokens import DeterministicSampler
from experiment_io import restore_rng,rng_state
from optim.adamw_reference import ReferenceAdamW

def test_optimizer_sampler_and_rng_resume_exact():
    def setup():
        random.seed(7);np.random.seed(7);torch.manual_seed(7);p=torch.nn.Parameter(torch.tensor([1.,-2.]));return p,ReferenceAdamW([p],lr=.01),DeterministicSampler(13,5,True)
    a,oa,sa=setup(); trajectory=[]
    for step in range(20):
        ids=sa.take(2);a.grad=torch.tensor([ids[0]/10,ids[1]/10]);oa.step();trajectory.append((ids,a.detach().clone()))
        if step==6: checkpoint={"p":a.detach().clone(),"o":copy.deepcopy(oa.state_dict()),"s":copy.deepcopy(sa.state_dict()),"r":rng_state(False)}
    b,ob,sb=setup();b.data.copy_(checkpoint["p"]);ob.load_state_dict(checkpoint["o"]);sb.load_state_dict(checkpoint["s"]);restore_rng(checkpoint["r"])
    for step in range(7,20):
        ids=sb.take(2);b.grad=torch.tensor([ids[0]/10,ids[1]/10]);ob.step();assert ids==trajectory[step][0] and torch.equal(b,trajectory[step][1])
