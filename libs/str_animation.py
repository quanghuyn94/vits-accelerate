import torch

def get_vram_usage_torch():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Chọn GPU đầu tiên
        vram_used = torch.cuda.memory_allocated(device) / (1024 ** 2)  # Chuyển đổi sang đơn vị MB
        return vram_used
    else:
        return None

class StrAnimator():
    def __init__(self, keys : list, reverse = False) -> None:
        self.keys : list = keys
        self.index = 0

        if reverse:
            tmp_keys = keys.copy()
            tmp_keys.reverse()
            self.keys.extend(tmp_keys)

    def next(self):
        if len(self.keys) <= self.index:
            self.index = 0

        back = self.keys[self.index]

        self.index = self.index + 1

        return back

class SystemDisplayAnimator(StrAnimator):
    def __init__(self, keys: list, reverse=False) -> None:
        super().__init__(keys, reverse)

        self.vram_used = f'GPU0: {get_vram_usage_torch():.2f}'
        self.keys.extend(([self.vram_used] * 3))