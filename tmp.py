import torch
import numpy as np
import pdb

class Objective_Functions():
       
    def evaluate(self, x):        
        n = x.shape[1]
        
        f1 = 1 - torch.exp(-torch.sum((x - 1 / np.sqrt(n))**2, axis = 1))
        f2 = 1 - torch.exp(-torch.sum((x + 1 / np.sqrt(n))**2, axis = 1))
     
        objs = torch.stack([f1,f2]).T
        
        return objs

class Scalar():
    def __init__(self):
        self.p = 0.001
        self.e = 0.1
        self.z = torch.Tensor([0.5, 0.5])
        self.weight = torch.Tensor([0.5, 0.5])

    def aug_tch(self, f_x):
        tch = torch.max(self.weight * (f_x - (self.z - torch.abs(self.z * self.e))), axis = 1)[0]
        ws = torch.sum(self.weight * f_x, axis = 1)
        # pdb.set_trace()

        return (tch + self.p * ws).unsqueeze(1)

def objective(x):
    obj_func = Objective_Functions()
    scalar = Scalar()
    f_x = obj_func.evaluate(x)
    tch_f_x = scalar.aug_tch(f_x).squeeze().unsqueeze(0)
    # print(tch_f_x)
    return tch_f_x

def groundtruth(dim):
    obj_func = Objective_Functions()
    num_sample = 10000*dim
    x = np.linspace(-1, 1, num_sample)
    x = np.repeat(np.expand_dims(x, axis=1), dim, axis = 1)
    x = torch.from_numpy(x)
    truth = obj_func.evaluate(x)
    scalar = Scalar()
    ref_vec = scalar.aug_tch(truth)
    return ref_vec


# obj = Objective_Functions()
# n_obj =2
# x = np.linspace(-1, 1, 100)                 # [100, ]
# x = torch.from_numpy(x).unsqueeze(1)        # [100, 1]
# truth = obj.evaluate(x)                      # [100, 2]
# # pdb.set_trace()

# scalar = Scalar()
# ref_vec = scalar.aug_tch(truth)
# pdb.set_trace()
front = groundtruth(10)
print(torch.min(front)," - ", torch.argmin(front))
