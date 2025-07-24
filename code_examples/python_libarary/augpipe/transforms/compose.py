import random

class ComposeTransform:
    def __init__(self, transforms, mode="fixed"):
        self.transforms = []
        for t in transforms:
            if isinstance(t, tuple):
                transform, p = t
            else:
                transform, p = t, 1.0
            self.transforms.append((transform, p))
        self.mode = mode

    def __call__(self, img, mask=None):
        for t, p in self.transforms:
            if hasattr(t, "mode"):
                t.mode = self.mode
            if random.random() < p:
                if mask is not None:
                    img, mask = t(img, mask)
                else:
                    img = t(img)
        return (img, mask) if mask is not None else img


class OneOfTransform:
    def __init__(self, transforms, mode="fixed"):
        self.transforms = []
        total_p = 0.0
        for t in transforms:
            if isinstance(t, tuple):
                transform, p = t
            else:
                transform, p = t, 1.0
            total_p += p
            self.transforms.append((transform, p))
        self.total_p = total_p
        self.mode = mode

    def __call__(self, img, mask=None):
        r = random.random() * self.total_p
        cumulative = 0.0
        for t, p in self.transforms:
            cumulative += p
            if r < cumulative:
                if hasattr(t, "mode"):
                    t.mode = self.mode
                if mask is not None:
                    return t(img, mask)
                else:
                    return t(img)
        return (img, mask) if mask is not None else img


class SometimesTransform:
    def __init__(self, transform, p=0.5, mode="fixed"):
        self.transform = transform
        self.p = p
        self.mode = mode

    def __call__(self, img, mask=None):
        if hasattr(self.transform, "mode"):
            self.transform.mode = self.mode
        if random.random() < self.p:
            if mask is not None:
                return self.transform(img, mask)
            else:
                return self.transform(img)
        return (img, mask) if mask is not None else img
