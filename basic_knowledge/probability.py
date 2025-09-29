import torch
from torch.distributions import multinomial

# fair_probs = torch.ones([6])/6
# counts = multinomial.Multinomial(1000,fair_probs).sample()
# print(counts/1000)

# print(dir(torch.distributions))