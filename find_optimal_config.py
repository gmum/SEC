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
import pandas as pd
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
        description="Evaluate image frequency and find the best network configuration."
    )
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument(
        "--size",
        type=str,
        choices=["S", "M", "L"],
        default="M",
        help="Network size (S=2, M=3, L=4 hidden layers). Default: M",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["siren", "finer", "relu", "wire", "all"],
        default="all",
        help="Network architecture. Default: all",
    )

    args = parser.parse_args()

    print(f"Calculating SEC for image: {args.image_path}...")
    target_freq = get_frequency(args.image_path, "center_of_mass")
    print(f"Calculated SEC: {target_freq}")

    try:
        df = pd.read_csv("data.csv")
    except FileNotFoundError:
        print("Error: 'data.csv' not found in the current directory.")
        return

    size_map = {"S": 2, "M": 3, "L": 4}
    target_layers = size_map[args.size]
    df_filtered = df[df["hidden_layers"] == target_layers].copy()

    # Filter by architecture
    if args.architecture != "all":
        df_filtered = df_filtered[df_filtered["architecture"] == args.architecture]

    if df_filtered.empty:
        print("No configurations found matching the selected size and architecture.")
        return

    # Filter entries with the most similar input_freq_standard_center_of_mass
    # We find the minimum absolute difference and keep all rows that match that minimum
    df_filtered = df_filtered.dropna(subset=["input_freq_standard_center_of_mass"])
    df_filtered["freq_diff"] = (
        df_filtered["input_freq_standard_center_of_mass"] - target_freq
    ).abs()
    min_diff = df_filtered["freq_diff"].min()

    df_closest = df_filtered[df_filtered["freq_diff"] == min_diff].copy()

    if df_closest.empty:
        print("No valid frequency matches found in the dataset.")
        return

    print(
        f"Found {len(df_closest)} configurations matching the closest frequency (Delta: {min_diff})."
    )

    # Calculate mean PSNR over seeds
    group_cols = ["architecture", "hidden_layers", "hidden_features", "sigma"]
    mean_psnr_df = df_closest.groupby(group_cols)["psnr"].mean().reset_index()

    # Show configuration with the highest PSNR
    best_config = mean_psnr_df.loc[mean_psnr_df["psnr"].idxmax()]

    architecture = best_config["architecture"]
    hidden_layers = int(best_config["hidden_layers"])
    hidden_features = int(best_config["hidden_features"])
    sigma = best_config["sigma"]
    datasets = f'["{args.image_path}"]'
    psnr = best_config["psnr"]

    print("\n" + "=" * 50)
    print("BEST CONFIGURATION (Highest Mean PSNR)")
    print("=" * 50)
    print(f"Architecture    : {architecture}")
    print(f"Hidden Layers   : {hidden_layers}")
    print(f"Hidden Features : {hidden_features}")
    print(f"Sigma/omega_0   : {sigma}")
    print(f"Mean PSNR       : {psnr:.4f}")
    print("=" * 50)
    print(
        f"python src/train.py --{datasets=} --conf=configs/{architecture}_base.yaml model.{'sigma' if architecture == 'relu' else 'omega_0'}={sigma} model.{hidden_layers=} model.{hidden_features=}"
    )
    print("=" * 50)


if __name__ == "__main__":
    main()
