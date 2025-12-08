from .identity import IdentityCorruptor
from .randomchar import RandomCharAug, RandomCharPuncAug, PuncAug

corruptors_names = {
    "identity": IdentityCorruptor,
    "randomcharaug_0.05": lambda: RandomCharAug([1, 2, 2, 1, 1, 1, 1],
                                                {'replace': 1, 'add': 2, 'swap': 3},
                                                0.05, 1, 1000, 2),
    "randomcharaug_0.1": lambda: RandomCharAug([1, 2, 2, 1, 1, 1, 1],
                                               {'replace': 1, 'add': 2, 'swap': 3},
                                               0.1, 1, 1000, 2, lang='rus'),
    "randomcharaug_0.2": lambda: RandomCharAug([1, 2, 2, 1, 1, 1, 1],
                                               {'replace': 1, 'add': 2, 'swap': 3},
                                               0.2, 1, 1000, 2),
    'ru_punc_randomcharaug_0.2': lambda: RandomCharPuncAug([1, 2, 2, 1, 1, 1, 1],
                                                           {'replace': 1, 'add': 2, 'swap': 3},
                                                           0.2, 1, 1000, 2,
                                                           0.4, lang='rus'),
    'en_punc_randomcharaug_0.2': lambda: RandomCharPuncAug([1, 2, 2, 1, 1, 1, 1],
                                                           {'replace': 1, 'add': 2, 'swap': 3},
                                                           0.2, 1, 1000, 2,
                                                           0.4, lang='eng'),
    'punc_0.2': lambda: PuncAug([1, 2, 2, 1, 1, 1, 1],
                                {'replace': 1, 'add': 2, 'swap': 3},
                                0.2, 1, 1000, 2, 0.5)
}
