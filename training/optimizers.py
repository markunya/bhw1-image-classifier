from utils.class_registry import ClassRegistry
from torch.optim import Adam
from torch.optim import SGD

optimizers_registry = ClassRegistry()

@optimizers_registry.add_to_registry(name="adam")
class Adam_(Adam):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0, amsgrad=False):
        super().__init__(params, lr=lr, betas=(beta1,beta2), eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

@optimizers_registry.add_to_registry(name="sgd")
class SGD_(SGD):
    def __init__(self, params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super().__init__(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
