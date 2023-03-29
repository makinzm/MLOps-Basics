import torch

torch.use_deterministic_algorithms(True)

try:
    print(
        torch.bmm(torch.randn(2, 2, 2).to_sparse().cuda(), torch.randn(2, 2, 2).cuda())
    )
except RuntimeError as e:
    print(e)
    print(
        torch.bmm(torch.randn(2, 2, 2).to_sparse(), torch.randn(2, 2, 2))
    )
finally:
    print("---------")

## There is no Reproducibility on CPU

torch.manual_seed(42)

print(
    torch.bmm(torch.randn(2, 2, 2).to_sparse(), torch.randn(2, 2, 2))
)

## There is Reproducibility on CPU
