from typing import Any, Callable, Optional, Tuple, List

from PIL import Image
from torchvision import datasets

class HandUpDownDataSet(datasets.VisionDataset):
    def __init__(
        self,
        file_list: List[str],
        targets: List[int],
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.my_files = file_list
        self.targets = targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        file_path = self.my_files[index]
        img = Image.open(file_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[index]

    def __len__(self) -> int:
        return len(self.targets)