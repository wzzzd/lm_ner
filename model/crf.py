
import torch
from torchcrf import CRF



num_tags = 5  # number of tags is 5
model = CRF(num_tags)


seq_length = 3                                                      # maximum sequence length in a batch
batch_size = 2                                                      # number of samples in the batch
emissions = torch.randn(seq_length, batch_size, num_tags)
tags = torch.tensor([[0, 1], [2, 4], [3, 1]], dtype=torch.long)     # (seq_length, batch_size)
loglikelihood = model(emissions, tags)
print(loglikelihood)

mask = torch.tensor([[1, 1], [1, 1], [1, 0]], dtype=torch.uint8)
loglikelihood = model(emissions, tags, mask=mask)
print(loglikelihood)

label = model.decode(emissions)
print(label)


tags1 = torch.tensor([[0, 1]], dtype=torch.long) 
tags2 = torch.tensor([[5, 10]], dtype=torch.long)
print((tags1,)+(tags2,))
print(tags1, tags2)



a = torch.tensor([[[1,2,3,4,5],[6,7,8,9,10]],[[11,12,31,4,15],[6,17,8,91,10]]])
# aa = a[:,1:-1]
# print(a)
# print(aa)
# print(torch.tensor(a, dtype=torch.uint8))
print(a.size())
print(a)
print(torch.argmax(a, dim=2))


