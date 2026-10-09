"""VGG-16 adapted for 32x32 CIFAR-10 images."""

from __future__ import annotations

from torch import Tensor, nn


VGG16_CONFIG = ((64, 2), (128, 2), (256, 3), (512, 3), (512, 3))


class VGG16CIFAR(nn.Module):
    """VGG-16 with global-average pooling and an optional identity shortcut.

    The plain and residual variants share every convolution and classifier.
    A residual add is used only when the input and output tensor shapes match,
    so it introduces no projection layers or trainable parameters.
    """

    def __init__(self, num_classes: int = 10, residual: bool = False, batch_norm: bool = False):
        super().__init__()
        self.residual = residual
        self.stages = nn.ModuleList()
        in_channels = 3

        for out_channels, conv_count in VGG16_CONFIG:
            stage = nn.ModuleList()
            for _ in range(conv_count):
                stage.append(
                    nn.ModuleDict(
                        {
                            "conv": nn.Conv2d(
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                padding=1,
                                bias=not batch_norm,
                            ),
                            "norm": nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
                        }
                    )
                )
                in_channels = out_channels
            self.stages.append(stage)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        for stage in self.stages:
            for layer in stage:
                identity = x
                x = layer["norm"](layer["conv"](x))
                if self.residual and identity.shape == x.shape:
                    x = x + identity
                x = nn.functional.relu(x, inplace=True)
            x = self.pool(x)

        x = nn.functional.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
