from utils.class_registry import ClassRegistry
from torch.optim.lr_scheduler import ReduceLROnPlateau

schedulers_registry = ClassRegistry()

@schedulers_registry.add_to_registry(name='reduce_on_plateau')
class ReduceOnPlateau(ReduceLROnPlateau):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)