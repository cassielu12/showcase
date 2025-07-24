class BaseTransform:
    def __init__(self, mode='random'):
        assert mode in ['random', 'fixed']
        self.mode = mode

    def __call__(self, img):
        raise NotImplementedError
