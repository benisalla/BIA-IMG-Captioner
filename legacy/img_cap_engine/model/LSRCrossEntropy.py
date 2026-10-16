import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class LSRCrossEntropy(torch.nn.Module):
    """
    Label Smoothing Cross Entropy Loss is a modification of the standard cross-entropy loss function
    that smooths the labels to improve generalization and reduce overfitting. It is commonly used in
    classification tasks to make the model less confident about the exact class labels.

    Attributes:
    -----------
    eps : float
        The smoothing parameter, a small value between 0 and 1, that controls the degree of smoothing applied
        to the target labels. Default is 0.1.
    device : str
        The device on which to perform the computation (e.g., "cpu", "cuda"). Default is "cpu".
    """

    def __init__(self, eps=0.1, device="cpu"):
        super(LSRCrossEntropy, self).__init__()
        self.eps = eps
        self.device = device

    def forward(self, x, y, lens):
        lens = lens.cpu()
        x = pack_padded_sequence(
            input=x, lengths=lens, batch_first=True, enforce_sorted=False
        ).data.to(self.device)
        y = pack_padded_sequence(
            input=y, lengths=lens, batch_first=True, enforce_sorted=False
        ).data.to(self.device)

        tv = (
            torch.zeros_like(x)
            .scatter(dim=1, index=y.unsqueeze(1), value=1.0)
            .to(self.device)
        )
        tv = tv * (1.0 - self.eps) + self.eps / tv.size(1)

        loss = (-1 * tv * F.log_softmax(x, dim=1)).sum(dim=1)
        loss = torch.mean(loss)

        return loss
