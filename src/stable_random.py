import numpy as np
import sys

def stable_random(size,seed,dist='normal'):
    rand=eval('np.random.'+dist)
    np.random.seed(seed)
    #if size==[]:
    if len(size)<=1:
        return rand(size=size)
    else:
        head=size[0]
        tail=size[1:]
        res=[]
        for i in range(0,head):
            res.append(stable_random(tail,np.random.randint(2**32-1),dist))
        return np.stack(res)
