import torch



K=10
B=30
labeling_matrix = torch.arange(K).unsqueeze(1) + torch.arange(B).unsqueeze(0)
print(f"{labeling_matrix.shape=}") 
print(f"{labeling_matrix=}")