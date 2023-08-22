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

        
