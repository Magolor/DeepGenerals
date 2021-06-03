import torch.distributions as dist
import seaborn as sns
import matplotlib.pyplot as plt
import torch

sampler = dist.Gumbel(loc =0 ,scale=1)
data = sampler.sample((1000,))

sns.distplot(data.numpy(),kde=False)
plt.show()