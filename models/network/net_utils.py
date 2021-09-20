"""positional encoding"""
import torch


class PosEncoder():
    def __init__(self, number_frequencies, include_identity):
        freq_bands = torch.pow(2, torch.linspace(0., number_frequencies - 1, number_frequencies))
        self.embed_fns = []
        self.output_dim = 0
        self.number_frequencies = number_frequencies
        self.include_identity = include_identity
        if include_identity:
            self.embed_fns.append(lambda x: x)
            self.output_dim += 1
        if number_frequencies > 0:
            for freq in freq_bands:
                for periodic_fn in [torch.sin, torch.cos]:
                    self.embed_fns.append(lambda x, periodic_fn=periodic_fn, freq=freq: periodic_fn(x * freq))
                    self.output_dim += 1

    def encode(self, coordinate):
        return torch.cat([fn(coordinate) for fn in self.embed_fns], -1)

