from utils.class_registry import ClassRegistry
from torch.optim import Adam
from torch.optim import SGD
from torch.optim import AdamW

optimizers_registry = ClassRegistry()

@optimizers_registry.add_to_registry(name="adam")
class Adam_(Adam):
    def __init__(self, params, lr=0.0001, beta1=0.9, beta2=0.999):
        super().__init__(params, lr=lr, betas=(beta1,beta2))

@optimizers_registry.add_to_registry(name="sgd")
class SGD_(SGD):
    def __init__(self, params, lr=0.01, momentum=0.9, dampening=0, weight_decay=0, nesterov=False):
        super().__init__(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)

@optimizers_registry.add_to_registry(name='adamW')
class AdamW_(AdamW):
    def __init__(self, params, lr=0.0001, beta1=0.9, beta2=0.999):
        super().__init__(params, lr=lr, betas=(beta1,beta2))
