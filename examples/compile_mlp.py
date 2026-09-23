import json

import torch
from torch import nn

import forgeml


def main() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.GELU(approximate="tanh"),
        nn.Linear(128, 32),
    ).eval()
    x = torch.randn(8, 64)
    compiled = forgeml.compile(model, (x,))
    expected = model(x)
    actual = compiled(x)
    torch.testing.assert_close(actual, expected)
    print(json.dumps(compiled.explain(), indent=2))


if __name__ == "__main__":
    main()
