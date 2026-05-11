"""
Copyright (C) 2025 Adam Kania, Tomasz Dądela, Maciej Rut, Przemysław Spurek, GlaxoSmithKline plc

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the “Software”), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import argparse

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import center_of_mass
from torchvision.transforms.v2.functional import to_dtype, to_image


def get_periodogram(y: torch.Tensor):
    assert np.array(y.shape).argmin() == 0, (
            f"First dim should be for channels, but found shape: {y.shape}"
        )
    fft = torch.fft.fft2(y)
    fft = torch.abs(fft)
    mask = torch.ones_like(fft)
    mask[:, 0, 0] = 0
    fft /= torch.max(mask * fft)
    periodogram = fft.square()
    return periodogram


def get_spectrum(y: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    if y.ndim == 3:
        y = y.mean(dim=0)
    assert y.ndim == 2
    n = y.shape[0]
    n2 = y.shape[1]
    y = torch.fft.fftshift(y)
    idx = torch.arange(-(n // 2), (n + 1) // 2)
    idx2 = torch.arange(-(n2 // 2), (n2 + 1) // 2)
    grid_y, grid_x = torch.meshgrid(idx, idx2, indexing="ij")
    dist = torch.sqrt(grid_x**2 + grid_y**2).round().to(torch.int64)
    max_radius = dist[0][0]
    result = torch.zeros(max_radius + 1, dtype=y.dtype)
    result.scatter_add_(0, dist.view(-1), y.view(-1))
    result[0] = 0
    if normalize:
        result = result / result.max()
    return result


def get_frequency_from_tensor(img: torch.Tensor, threshold: float | str) -> int:
    assert threshold == "center_of_mass" or 0 < threshold < 1
    fft = get_periodogram(img)
    spectrum = get_spectrum(fft)

    if threshold == "center_of_mass":
        return round(center_of_mass(spectrum.detach().cpu().numpy())[0])

    cs = torch.cumsum(spectrum, dim=0)
    tot = cs[-1]
    freq_mask = cs >= threshold * tot
    max_freq = torch.where(freq_mask)[0][0].item()
    return max_freq


def get_frequency(img_path, threshold):
    img = Image.open(img_path)
    img_tensor = to_dtype(to_image(img), dtype=torch.float32, scale=True)
    return get_frequency_from_tensor(img_tensor, threshold)


def main():
    parser = argparse.ArgumentParser(
        description="Compute SEC of a given image."
    )
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    args = parser.parse_args()

    print(f"Calculating SEC for image: {args.image_path}...")
    target_freq = get_frequency(args.image_path, "center_of_mass")
    print(f"SEC: {target_freq}")


if __name__ == "__main__":
    main()
