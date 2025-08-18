import torch
from torch.distributions import multinomial
import matplotlib.pyplot as plt

fair_probs = torch.ones([6]) / 6
# print(fair_probs)
# print(multinomial.Multinomial(10, fair_probs).sample())
# print(multinomial.Multinomial(10000, fair_probs).sample()/10000)


counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
# print(cum_counts.shape, cum_counts)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
# print(estimates.shape, estimates)
plt.figure(figsize=(6, 4.5))
for i in range(6):
    plt.plot(estimates[:, i].numpy(), label=("P(die=" + str(i + 1) + ")"))
plt.legend()
plt.show()
