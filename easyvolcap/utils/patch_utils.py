import torch
import numpy as np
from typing import List, Union


def sample_patch(H: int, W: int,
                 patch_size: Union[int, List[int]],
                 exact: bool = False
                 ) -> List[int]:
    """ Sample a patch from an image of size H x W

    Args:
        H (int): height of the image
        W (int): width of the image
        patch_size (int or List[int]): size of the patch to be sampled. If int, the patch is square
        exact (bool): If True, the patch is sampled exactly at the center of the image

    Returns:
        x, y, w, h (int): coordinates of the patch
    """
    # Convert patch_size to a list
    if isinstance(patch_size, int): patch_size = [patch_size, patch_size]
    Hp, Wp = patch_size

    if exact:
        # Try to sample the exact size of the patch
        x = 0 if W - Wp <= 0 else np.random.randint(0, W - Wp + 1)
        y = 0 if H - Hp <= 0 else np.random.randint(0, H - Hp + 1)
        w = min(W, Wp)
        h = min(H, Hp)
    else:
        # Sample a patch of with more flexibility, may be smaller than the patch_size
        at_least = min(Hp, Wp) // 4
        x = np.random.randint(-Wp + 1 + at_least, W - at_least)  # left edge
        y = np.random.randint(-Hp + 1 + at_least, H - at_least)  # top edge
        r = min(W, Wp + x)
        d = min(H, Hp + y)
        x = max(0, x)
        y = max(0, y)
        w = r - x
        h = d - y

    return x, y, w, h
