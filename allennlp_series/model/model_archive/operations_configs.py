import torch


operations_all = []
for a in [-1.0, 0.0, 1.0]:
    for b in [-1.0, 0.0, 1.0]:
        for c in [-1.0, 0.0, 1.0]:
            operations_all.append(torch.tensor([a, b, c]))
operations_configa = [
    torch.tensor([-1.0, 1.0, 0.0]),
    torch.tensor([0.0, 1.0, 0.0]),
    torch.tensor([0.3, 0.3, 0.3]),
    torch.tensor([0.0, -1.0, 1.0]),
    torch.tensor([0.0, 0.0, 0.0]),
]
# operations_configb = [
#         torch.tensor([-1.0, 1., 0.]),
#         torch.tensor([0., 1., 0.]),
#         torch.tensor([0.3, 0.3, 0.3 ]),
#         torch.tensor([0., -1., 1.]),
#         torch.tensor([0., 0., 0.])
#     ]
operations_configb = [
    torch.tensor([-1.0, 1.0, 0.0]),
    torch.tensor([0.0, 1.0, 0.0]),
    torch.tensor([0.0, -1.0, 1.0]),
    torch.tensor([0.0, 0.0, 0.0]),
    torch.tensor([0.5, -1.0, 0.5]),  # double der
]


vert_two_operations_configa = [
    torch.tensor([1.0, -1.0]),
    torch.tensor(
        [
            0.5,
            0.5,
        ]
    ),
    torch.tensor([-1.0, 1.0]),
]
