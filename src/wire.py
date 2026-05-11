"""
MIT License

Copyright (c) 2022 Vishwanath Saragadam

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from models import BaseMLP


class ComplexGaborLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega0=10.0,
        sigma0=40.0,
        trainable=False,
    ):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        self.in_features = in_features

        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat

        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        return torch.exp(1j * omega - scale.abs().square())


class Wire(BaseMLP):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        hidden_layers,
        first_omega_0=20.0,
        hidden_omega_0=20.0,
        scale=10.0,
    ):
        # hidden_omega_0 = first_omega_0
        super().__init__()
        self._spectral_parameters = (first_omega_0, scale)
        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin = ComplexGaborLayer

        # Since complex numbers are two real numbers, reduce the number of hidden parameters by 2
        hidden_features = int(hidden_features / np.sqrt(2))
        dtype = torch.cfloat
        self.complex = True

        self._net = []
        self._net.append(
            self.nonlin(
                in_features,
                hidden_features,
                is_first=True,
                omega0=first_omega_0,
                sigma0=scale,
            )
        )

        for i in range(hidden_layers):
            self._net.append(
                self.nonlin(
                    hidden_features, hidden_features, omega0=hidden_omega_0, sigma0=10
                )
            )

        final_linear = nn.Linear(hidden_features, out_features, dtype=dtype)
        self._net.append(final_linear)

        class RealLayer(nn.Module):
            def forward(self, x):
                return x.real / 2 + 0.5

        self._net.append(RealLayer())
        self._net = nn.Sequential(*self._net)

    @property
    def net(self) -> nn.Sequential:
        return self._net

    @property
    def spectral_parameters(self) -> tuple:
        return self._spectral_parameters
