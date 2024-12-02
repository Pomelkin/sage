class IdentityCorruptor:
    def __init__(self):
        pass

    def corrupt(self, x):
        return x

    def __call__(self, x, tokenizer):
        return self.corrupt(x), '', []
