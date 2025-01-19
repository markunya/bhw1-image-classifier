from typing import Literal
from utils.class_registry import ClassRegistry
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import ExponentialLR

schedulers_registry = ClassRegistry()

@schedulers_registry.add_to_registry(name='reduce_on_plateau')
class ReduceOnPlateau(ReduceLROnPlateau):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)

@schedulers_registry.add_to_registry(name='multi_step')
class MultiStep(MultiStepLR):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)

@schedulers_registry.add_to_registry(name='exponential')
class Exponential(ExponentialLR):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)

@schedulers_registry.add_to_registry(name='cyclic')
class Cyclic(LambdaLR):
    def __init__(
        self,
        optimizer,
        period: int,
        jump_factor: float,
        lr_interval: tuple[float, float],
        mode: Literal['linear', 'convex', 'concave'] = 'linear',
        last_epoch: int = -1
    ):
        """
        Cyclic LR Scheduler:
        
        :param optimizer: Оптимизатор, для которого применяется шедулер.
        :param period: Период цикла в эпохах.
        :param jump_factor: Множитель для lr в начале цикла.
        :param lr_interval: Кортеж (min_factor, max_factor), задающий диапазон множителей после jump_factor.
        :param mode: Функция убывания ('linear', 'convex', 'concave').
        :param last_epoch: Номер последней эпохи (по умолчанию -1).
        """
        self.period = period
        self.jump_factor = jump_factor
        self.max_lr, self.min_lr = lr_interval
        self.mode = mode
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, epoch: int) -> float:
        if epoch == 0:
            return 1
        cycle_position = epoch % self.period

        if cycle_position == 0:
            return self.jump_factor
        else:
            progress = cycle_position / self.period
            factor = (1 - ((epoch + 1) % self.period) / self.period) / (1 - progress)
            if self.mode == 'linear':
                pass
            if self.mode == 'convex':
                factor = factor ** 2
            elif self.mode == 'concave':
                factor = factor ** 0.5
            else:
                raise ValueError(f"Invalid mode '{self.mode}'. Must be 'linear', 'convex', or 'concave'.")
            
            return factor
