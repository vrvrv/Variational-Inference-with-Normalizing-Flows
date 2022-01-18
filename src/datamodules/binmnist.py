from typing import Optional, Union, Any
from pl_bolts.datamodules import BinaryMNISTDataModule

class BinMNISTDataModule(BinaryMNISTDataModule):
    def __init__(self,
            data_dir: Optional[str] = None,
            val_split: Union[int, float] = 0.2,
            num_workers: int = 0,
            normalize: bool = False,
            batch_size: int = 32,
            seed: int = 42,
            shuffle: bool = True,
            pin_memory: bool = True,
            drop_last: bool = False,
            *args: Any,
            **kwargs: Any
        ):
        super(BinMNISTDataModule, self).__init__(
            data_dir,
            val_split,
            num_workers,
            normalize,
            batch_size,
            seed,
            shuffle,
            pin_memory,
            drop_last,
            *args,
            ** kwargs
        )