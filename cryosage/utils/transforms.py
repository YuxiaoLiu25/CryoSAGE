# Adapted from CryoSTAR under Apache-2.0. See NOTICE and provenance.json.
from typing import Optional
import einops
import torch
import torch.nn.functional as F

def shift_coords(coords: torch.Tensor,
                 x_half_range: float,
                 y_half_range: float,
                 z_half_range: float,
                 Nx: int,
                 Ny: int,
                 Nz: Optional[int],
                 flip: bool = False):
    """
    Shifts the coordinates and puts the DC component at (0, 0, 0).
    
    If an image has an even size, such as 4, the `fftshift`ed freq-map looks like:
    `[-0.50, -0.25,  0.00,  0.25]`, which means the DC term locates at the bottom-right
    block. It has an offset compared with uniformly initialized coords:
    `[-0.50, -0.17,  0.17,  0.50]`. The offset is a left-shift of 0.17, i.e.,
    `half-range/(num_grids-1)`.

    This may cause errors when we apply a frequency-based operation to the image 
    in the fourier domain, such as translation. See `FourierGridTranslate` for details.

    See Also: https://pytorch.org/docs/stable/generated/torch.fft.fftshift.html

    Input:
        coords:     torch.Tensor (..., 3 or 2)
        x_range:    float, (max_x - min_x) / 2
        y_range:    float, (max_y - min_y) / 2
        z_range:    float, (max_z - min_z) / 2
        Nx:         int 
        Ny:         int
        Nz:         int
        flip:       bool

    Returns:
        coords: torch.Tensor (..., 3 or 2)
    """
    alpha = -1.
    if flip:  # "unshift" the coordinates.
        alpha = 1.

    if Nx % 2 == 0:
        x_shift = coords[..., 0] + alpha * x_half_range / (Nx - 1)
    else:
        x_shift = coords[..., 0]
    if Ny % 2 == 0:
        y_shift = coords[..., 1] + alpha * y_half_range / (Ny - 1)
    else:
        y_shift = coords[..., 1]
    if Nz is not None:
        if Nz % 2 == 0:
            z_shift = coords[..., 2] + alpha * z_half_range / (Nz - 1)
        else:
            z_shift = coords[..., 2]
        coords = torch.cat((x_shift.unsqueeze(-1), y_shift.unsqueeze(-1), z_shift.unsqueeze(-1)), dim=-1)
    else:
        coords = torch.cat((x_shift.unsqueeze(-1), y_shift.unsqueeze(-1)), dim=-1)
    return coords


class FourierGridTranslate(torch.nn.Module):
    """
        DFT's translation is:
            `f(x - x0, y - y0) <=> F(u, v) exp(-2 j \pi (x0 u + y0 v) / N )`
        where `x, y, u, v` all have a range of `N`, so `(x0 u + y0 v) / N \in (0, N)`

        Here we initialize the `u, v` coordinates between `(-0.5, 0.5)` so that the 
        range is 1, where the `1/N` term can be ignored.

        See also: https://dsp.stackexchange.com/questions/40228/translation-property-of-2-d-discrete-fourier-transform

        Important notes:
            If `N=4`, the coordinates u will be `[-0.5, -0.17, 0.17, 0.5]`, but the 
            `fft`ed image's frequency is `[-0.50, -0.25, 0.00, 0.25]`, so we have to 
            add some corrections:
                - right-shift `u` to be `[-0.50, -0.25, 0.00, 0.25]`
                - perform multiplication

    """

    def __init__(self, D, device=None) -> None:
        super().__init__()
        self.D = D
        # yapf: disable
        coords = torch.stack(torch.meshgrid([
            torch.linspace(-1.0, 1.0, self.D, device=device),
            torch.linspace(-1.0, 1.0, self.D, device=device)],
        indexing="ij"), dim=-1).reshape(-1, 2) / 2
        # yapf: enable
        coords = shift_coords(coords, 0.5, 0.5, None, self.D, self.D, None, False)
        self.register_buffer("coords", coords)

    def transform(self, images: torch.Tensor, trans: torch.Tensor):
        """
            The `images` are stored in `YX` mode, so the `trans` is also `YX`!

            Input:
                images: (B, NY, NX)
                trans:  (B, T,  2)

            Returns:
                images: (B, T,  NY, NX)
        """
        B, NY, NX = images.shape
        assert self.D == NY == NX
        assert images.shape[0] == trans.shape[0]
        images = einops.rearrange(images, "B NY NX -> B 1 (NY NX)")
        delta = trans @ self.coords.t() * -2j * torch.pi
        images_trans = torch.exp(delta) * images
        images_trans = einops.rearrange(images_trans, "B T (NY NX) -> B T NY NX", NY=self.D, NX=self.D)
        return images_trans


class SpatialGridTranslate(torch.nn.Module):

    def __init__(self, D, device=None) -> None:
        super().__init__()
        self.D = D
        # yapf: disable
        coords = torch.stack(torch.meshgrid([
            torch.linspace(-1.0, 1.0, self.D, device=device),
            torch.linspace(-1.0, 1.0, self.D, device=device)],
        indexing="ij"), dim=-1).reshape(-1, 2)
        # yapf: enable
        self.register_buffer("coords", coords)

    def transform(self, images: torch.Tensor, trans: torch.Tensor):
        """
            The `images` are stored in `YX` mode, so the `trans` is also `YX`!

            Supposing that D is 96, a point is at 0.0:
                - adding 48 should move it to the right corner which is 1.0
                    1.0 = 0.0 + 48 / (96 / 2)
                - adding 96(>48) should leave it at 0.0
                    0.0 = 0.0 + 96 / (96 / 2) - 2.0
                - adding -96(<48) should leave it at 0.0
                    0.0 = 0.0 - 96 / (96 / 2) + 2.0

            Input:
                images: (B, NY, NX)
                trans:  (B, T,  2)

            Returns:
                images: (B, T,  NY, NX)
        """
        B, NY, NX = images.shape
        assert self.D == NY == NX
        assert images.shape[0] == trans.shape[0]

        grid = einops.rearrange(self.coords, "N C2 -> 1 1 N C2") - \
            einops.rearrange(trans, "B T C2 -> B T 1 C2") * 2 / self.D
        grid = grid.flip(-1)  # convert the first axis from slow-axis to fast-axis
        grid[grid >= 1] -= 2
        grid[grid <= -1] += 2
        grid.clamp_(-1.0, 1.0)

        sampled = F.grid_sample(einops.rearrange(images, "B NY NX -> B 1 NY NX"), grid, align_corners=True)

        sampled = einops.rearrange(sampled, "B 1 T (NY NX) -> B T NY NX", NX=NX, NY=NY)
        return sampled

