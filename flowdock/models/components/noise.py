import numpy as np
import rootutils
import torch
from beartype.typing import Any, Dict, List, Optional, Tuple, Union

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowdock.models.components.transforms import LatentCoordinateConverter
from flowdock.utils import RankedLogger
from flowdock.utils.model_utils import segment_mean

MODEL_BATCH = Dict[str, Any]

log = RankedLogger(__name__, rank_zero_only=True)


class DiffusionSDE:
    """Diffusion SDE class.

    Adapted from: https://github.com/HannesStark/FlowSite
    """

    def __init__(self, sigma: torch.Tensor, tau_factor: float = 5.0):
        """Initialize the Diffusion SDE class."""
        self.lamb = 1 / sigma**2
        self.tau_factor = tau_factor

    def var(self, t: torch.Tensor) -> torch.Tensor:
        """Calculate the variance of the diffusion SDE."""
        return (1 - torch.exp(-self.lamb * t)) / self.lamb

    def max_t(self) -> float:
        """Calculate the maximum time of the diffusion SDE."""
        return self.tau_factor / self.lamb

    def mu_factor(self, t: torch.Tensor) -> torch.Tensor:
        """Calculate the mu factor of the diffusion SDE."""
        return torch.exp(-self.lamb * t / 2)


class HarmonicSDE:
    """Harmonic SDE class.

    Adapted from: https://github.com/HannesStark/FlowSite
    """

    def __init__(self, J: Optional[torch.Tensor] = None, diagonalize: bool = True):
        """Initialize the Harmonic SDE class."""
        self.l_index = 1
        self.use_cuda = False
        if not diagonalize:
            return
        if J is not None:
            self.D, self.P = np.linalg.eigh(J)
            self.N = self.D.size

    @staticmethod
    def diagonalize(
        N,
        ptr: torch.Tensor,
        edges: Optional[List[Tuple[int, int]]] = None,
        antiedges: Optional[List[Tuple[int, int]]] = None,
        a=1,
        b=0.3,
        lamb: Optional[torch.Tensor] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Diagonalize using the Harmonic SDE."""
        device = device or ptr.device
        J = torch.zeros((N, N), device=device)
        if edges is None:
            for i, j in zip(np.arange(N - 1), np.arange(1, N)):
                J[i, i] += a
                J[j, j] += a
                J[i, j] = J[j, i] = -a
        else:
            for i, j in edges:
                J[i, i] += a
                J[j, j] += a
                J[i, j] = J[j, i] = -a
        if antiedges is not None:
            for i, j in antiedges:
                J[i, i] -= b
                J[j, j] -= b
                J[i, j] = J[j, i] = b
        if edges is not None:
            J += torch.diag(lamb)

        Ds, Ps = [], []
        for start, end in zip(ptr[:-1], ptr[1:]):
            D, P = torch.linalg.eigh(J[start:end, start:end])
            D_ = D
            if edges is None:
                D_inv = 1 / D
                D_inv[0] = 0
                D_ = D_inv
            Ds.append(D_)
            Ps.append(P)
        return torch.cat(Ds), torch.block_diag(*Ps)

    def eigens(self, t):
        """Calculate the eigenvalues of `sigma_t` using the Harmonic SDE."""
        np_ = torch if self.use_cuda else np
        D = 1 / self.D * (1 - np_.exp(-t * self.D))
        t = torch.tensor(t, device="cuda").float() if self.use_cuda else t
        return np_.where(D != 0, D, t)

    def conditional(self, mask, x2):
        """Calculate the conditional distribution using the Harmonic SDE."""
        J_11 = self.J[~mask][:, ~mask]
        J_12 = self.J[~mask][:, mask]
        h = -J_12 @ x2
        mu = np.linalg.inv(J_11) @ h
        D, P = np.linalg.eigh(J_11)
        z = np.random.randn(*mu.shape)
        return (P / D**0.5) @ z + mu

    def A(self, t, invT=False):
        """Calculate the matrix `A` using the Harmonic SDE."""
        D = self.eigens(t)
        A = self.P * (D**0.5)
        if not invT:
            return A
        AinvT = self.P / (D**0.5)
        return A, AinvT

    def Sigma_inv(self, t):
        """Calculate the inverse of the covariance matrix `Sigma` using the Harmonic SDE."""
        D = 1 / self.eigens(t)
        return (self.P * D) @ self.P.T

    def Sigma(self, t):
        """Calculate the covariance matrix `Sigma` using the Harmonic SDE."""
        D = self.eigens(t)
        return (self.P * D) @ self.P.T

    @property
    def J(self):
        """Return the matrix `J`."""
        return (self.P * self.D) @ self.P.T

    def rmsd(self, t):
        """Calculate the root mean square deviation using the Harmonic SDE."""
        l_index = self.l_index
        D = 1 / self.D * (1 - np.exp(-t * self.D))
        return np.sqrt(3 * D[l_index:].mean())

    def sample(self, t, x=None, score=False, k=None, center=True, adj=False):
        """Sample from the Harmonic SDE."""
        l_index = self.l_index
        np_ = torch if self.use_cuda else np
        if x is None:
            if self.use_cuda:
                x = torch.zeros((self.N, 3), device="cuda").float()
            else:
                x = np.zeros((self.N, 3))
        if t == 0:
            return x
        z = (
            np.random.randn(self.N, 3)
            if not self.use_cuda
            else torch.randn(self.N, 3, device="cuda").float()
        )
        D = self.eigens(t)
        xx = self.P.T @ x
        if center:
            z[0] = 0
            xx[0] = 0
        if k:
            z[k + l_index :] = 0
            xx[k + l_index :] = 0

        out = np_.exp(-t * self.D / 2)[:, None] * xx + np_.sqrt(D)[:, None] * z

        if score:
            score = -(1 / np_.sqrt(D))[:, None] * z
            if adj:
                score = score + self.D[:, None] * out
            return self.P @ out, self.P @ score
        return self.P @ out

    def score_norm(self, t, k=None, adj=False):
        """Calculate the score norm using the Harmonic SDE."""
        if k == 0:
            return 0
        l_index = self.l_index
        np_ = torch if self.use_cuda else np
        k = k or self.N - 1
        D = 1 / self.eigens(t)
        if adj:
            D = D * np_.exp(-self.D * t)
        return (D[l_index : k + l_index].sum() / self.N) ** 0.5

    def inject(self, t, modes):
        """Inject noise along the given modes using the Harmonic SDE."""
        z = (
            np.random.randn(self.N, 3)
            if not self.use_cuda
            else torch.randn(self.N, 3, device="cuda").float()
        )
        z[~modes] = 0
        A = self.A(t, invT=False)
        return A @ z

    def score(self, x0, xt, t):
        """Calculate the score of the diffusion kernel using the Harmonic SDE."""
        Sigma_inv = self.Sigma_inv(t)
        mu_t = (self.P * np.exp(-t * self.D / 2)) @ (self.P.T @ x0)
        return Sigma_inv @ (mu_t - xt)

    def project(self, X, k, center=False):
        """Project onto the first `k` nonzero modes using the Harmonic SDE."""
        l_index = self.l_index
        D = self.P.T @ X
        D[k + l_index :] = 0
        if center:
            D[0] = 0
        return self.P @ D

    def unproject(self, X, mask, k, return_Pinv=False):
        """Find the vector along the first k nonzero modes whose mask is closest to `X`"""
        l_index = self.l_index
        PP = self.P[mask, : k + l_index]
        Pinv = np.linalg.pinv(PP)
        out = self.P[:, : k + l_index] @ Pinv @ X
        if return_Pinv:
            return out, Pinv
        return out

    def energy(self, X):
        """Calculate the energy using the Harmonic SDE."""
        l_index = self.l_index
        return (self.D[:, None] * (self.P.T @ X) ** 2).sum(-1)[l_index:] / 2

    @property
    def free_energy(self):
        """Calculate the free energy using the Harmonic SDE."""
        l_index = self.l_index
        return 3 * np.log(self.D[l_index:]).sum() / 2

    def KL_H(self, t):
        """Calculate the Kullback-Leibler divergence using the Harmonic SDE."""
        l_index = self.l_index
        D = self.D[l_index:]
        return -3 * 0.5 * (np.log(1 - np.exp(-D * t)) + np.exp(-D * t)).sum(0)


def sample_gaussian_prior(
    x0: torch.Tensor,
    latent_converter: LatentCoordinateConverter,
    sigma: float,
    x0_sigma: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample noise from a Gaussian prior distribution.

    :param x0: ground-truth tensor
    :param latent_converter: The latent coordinate converter
    :param sigma: standard deviation of the Gaussian noise
    :param x0_sigma: standard deviation of the Gaussian noise for the ground-truth tensor
    :return: tuple of ground-truth and predicted tensors with additive Gaussian prior noise
    """
    prior = torch.randn_like(x0)
    x_int_0 = x0 + prior * x0_sigma  # add small Gaussian noise to the ground-truth tensor
    (
        x1_ca_lat,
        x1_cother_lat,
        x1_lig_lat,
    ) = torch.split(
        prior * sigma,
        [
            latent_converter._n_res_per_sample,
            latent_converter._n_cother_per_sample,
            latent_converter._n_ligha_per_sample,
        ],
        dim=1,
    )
    x_int_1 = torch.cat(
        [
            x1_ca_lat,
            x1_cother_lat,
            x1_lig_lat,
        ],
        dim=1,
    )
    return x_int_0, x_int_1


def sample_protein_harmonic_prior(
    protein_ca_x0: torch.Tensor,
    protein_cother_x0: torch.Tensor,
    batch: MODEL_BATCH,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample protein noise from a harmonic prior distribution.
    Adapted from: https://github.com/bjing2016/alphaflow

    Note that this function represents non-Ca atoms as Gaussian noise
    centered around each harmonically-noised Ca atom.

    :param protein_ca_x0: ground-truth protein Ca-atom tensor
    :param protein_cother_x0: ground-truth protein other-atom tensor
    :param batch: A batch dictionary
    :return: tuple of harmonic protein Ca atom noise and Gaussian protein other atom noise
    """
    indexer = batch["indexer"]
    metadata = batch["metadata"]
    protein_bid = indexer["gather_idx_a_structid"]
    protein_num_nodes = protein_ca_x0.size(0) * protein_ca_x0.size(1)
    ptr = torch.cumsum(torch.bincount(protein_bid), dim=0)
    ptr = torch.cat((torch.tensor([0], device=protein_bid.device), ptr))
    try:
        D_inv, P = HarmonicSDE.diagonalize(
            protein_num_nodes,
            ptr,
            a=3 / (3.8**2),
        )
    except Exception as e:
        log.error(
            f"Failed to call HarmonicSDE.diagonalize() for protein(s) {metadata['sample_ID_per_sample']} due to: {e}"
        )
        raise e
    noise = torch.randn((protein_num_nodes, 3), device=protein_ca_x0.device)
    harmonic_ca_noise = P @ (torch.sqrt(D_inv)[:, None] * noise)
    gaussian_cother_noise = (
        torch.randn_like(protein_cother_x0.flatten(0, 1))
        + harmonic_ca_noise[indexer["gather_idx_a_cotherid"]]
    )
    return (
        harmonic_ca_noise.view(protein_ca_x0.size()).contiguous(),
        gaussian_cother_noise.view(protein_cother_x0.size()).contiguous(),
    )


def sample_ligand_harmonic_prior(
    lig_x0: torch.Tensor, protein_ca_x0: torch.Tensor, batch: MODEL_BATCH, sigma: float = 1.0
) -> torch.Tensor:
    """
    Sample ligand noise from a harmonic prior distribution.
    Adapted from: https://github.com/HannesStark/FlowSite

    :param lig_x0: ground-truth ligand tensor
    :param protein_x0: ground-truth protein Ca-atom tensor
    :param batch: A batch dictionary
    :param sigma: standard deviation of the harmonic noise
    :return: tensor of harmonic noise
    """
    indexer = batch["indexer"]
    metadata = batch["metadata"]
    lig_num_nodes = lig_x0.size(0) * lig_x0.size(1)
    num_molid_per_sample = max(metadata["num_molid_per_sample"])
    # NOTE: here, we distinguish each ligand chain in a complex for harmonic chain sampling
    lig_bid = indexer["gather_idx_i_molid"]
    protein_sigma = (
        segment_mean(
            torch.square(protein_ca_x0).flatten(0, 1),
            indexer["gather_idx_a_structid"],
            metadata["num_structid"],
        ).mean(dim=-1)
        ** 0.5
    ).repeat_interleave(num_molid_per_sample)
    sde = DiffusionSDE(protein_sigma * sigma)
    edges = torch.stack(
        (
            indexer["gather_idx_ij_i"],
            indexer["gather_idx_ij_j"],
        )
    )
    edges = edges[:, edges[0] < edges[1]]  # de-duplicate edges
    ptr = torch.cumsum(torch.bincount(lig_bid), dim=0)
    ptr = torch.cat((torch.tensor([0], device=lig_bid.device), ptr))
    try:
        D, P = HarmonicSDE.diagonalize(
            lig_num_nodes,
            ptr,
            edges=edges.T,
            lamb=sde.lamb[lig_bid],
        )
    except Exception as e:
        log.error(
            f"Failed to call HarmonicSDE.diagonalize() for ligand(s) {metadata['sample_ID_per_sample']} due to: {e}"
        )
        raise e
    noise = torch.randn((lig_num_nodes, 3), device=lig_x0.device)
    prior = P @ (noise / torch.sqrt(D)[:, None])
    return prior.view(lig_x0.size()).contiguous()


def sample_complex_harmonic_prior(
    x0: torch.Tensor,
    latent_converter: LatentCoordinateConverter,
    batch: MODEL_BATCH,
    x0_sigma: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample protein-ligand complex noise from a harmonic prior distribution.
    From: https://github.com/bjing2016/alphaflow

    :param x0: ground-truth tensor
    :param latent_converter: The latent coordinate converter
    :param batch: A batch dictionary
    :param x0_sigma: standard deviation of the Gaussian noise for the ground-truth tensor
    :return: tuple of ground-truth and predicted tensors with additive Gaussian and harmonic prior noise, respectively
    """
    ca_lat, cother_lat, lig_lat = x0.split(
        [
            latent_converter._n_res_per_sample,
            latent_converter._n_cother_per_sample,
            latent_converter._n_ligha_per_sample,
        ],
        dim=1,
    )
    harmonic_ca_lat, gaussian_cother_lat = sample_protein_harmonic_prior(
        ca_lat,
        cother_lat,
        batch,
    )
    harmonic_lig_lat = sample_ligand_harmonic_prior(lig_lat, harmonic_ca_lat, batch)
    x1 = torch.cat(
        [
            # NOTE: the following normalization steps assume that `self.latent_model == "default"`
            harmonic_ca_lat / latent_converter.ca_scale,
            gaussian_cother_lat / latent_converter.other_scale,
            harmonic_lig_lat / latent_converter.other_scale,
        ],
        dim=1,
    )
    gaussian_prior = torch.randn_like(x0)
    return x0 + gaussian_prior * x0_sigma, x1


def sample_esmfold_prior(
    x0: torch.Tensor, x1: torch.Tensor, sigma: float, x0_sigma: float = 1e-4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample noise from an ESMFold prior distribution.

    :param x0: ground-truth tensor
    :param x1: predicted tensor
    :param sigma: standard deviation of the ESMFold prior's additive Gaussian noise
    :param x0_sigma: standard deviation of the Gaussian noise for the ground-truth tensor
    :return: tuple of ground-truth and predicted tensors with additive Gaussian prior noise
    """
    prior_noise = torch.randn_like(x0)
    return x0 + prior_noise * x0_sigma, x1 + prior_noise * sigma


# =============================================================================
# KPFM Prior Sampling
# =============================================================================

def sample_kpfm_prior(
    x0: torch.Tensor,
    kin_system: Any,
    dof_cache: Any,
    latent_converter: Any,
    translation_std: float = 5.0,
    rotation_std: float = 0.5,
    x0_sigma: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, Any, Any]:
    """
    Sample from KPFM prior distribution in DOF space.
    
    This is the core KPFM prior that samples in DOF space and reconstructs
    coordinates using forward kinematics, ensuring all geometric constraints
    (bond lengths, angles) are maintained.
    
    Args:
        x0: [B, N_latent, 3] ground-truth latent coordinates (holo structure)
        kin_system: KinematicSystem object
        dof_cache: CachedDOFData with target DOF and torsion definitions
        latent_converter: LatentCoordinateConverter for coordinate transforms
        translation_std: Std for ligand translation prior (Angstroms)
        rotation_std: Std for ligand rotation prior (radians)
        x0_sigma: Small noise to add to ground-truth
    
    Returns:
        noisy_x0: Ground-truth with small additive noise
        x1: Prior sample reconstructed via FK
        q0: Target DOF state
        q1: Prior DOF state
    """
    from flowdock.data.components.dof_utils import sample_dof_prior, DOFState
    from flowdock.models.components.kinematics import ForwardKinematics
    
    batch_size = x0.shape[0]
    device = x0.device
    dtype = x0.dtype
    
    # Add small noise to ground-truth
    gaussian_noise = torch.randn_like(x0)
    noisy_x0 = x0 + gaussian_noise * x0_sigma
    
    # Get target DOF state from cache
    q0 = dof_cache.target_dof
    if q0.translation.shape[0] == 1 and batch_size > 1:
        # Expand to batch
        q0 = DOFState(
            translation=q0.translation.expand(batch_size, -1).clone(),
            rotation=q0.rotation.expand(batch_size, -1).clone(),
            ligand_torsions=q0.ligand_torsions.expand(batch_size, -1).clone(),
            sidechain_torsions=q0.sidechain_torsions.expand(batch_size, -1).clone()
        )
    q0 = q0.to(device)
    
    # Sample prior DOF state
    q1 = sample_dof_prior(
        batch_size=batch_size,
        kin_system=kin_system,
        translation_std=translation_std,
        rotation_std=rotation_std,
        device=device,
        dtype=dtype
    )
    
    # Reconstruct prior coordinates via forward kinematics
    fk = ForwardKinematics()
    
    # Get reference coordinates from kinematic system
    ref_coords = kin_system.reference_coords
    if ref_coords.dim() == 2:
        ref_coords = ref_coords.unsqueeze(0)
    if ref_coords.shape[0] == 1 and batch_size > 1:
        ref_coords = ref_coords.expand(batch_size, -1, -1)
    
    # Apply FK to get prior coordinates
    x1_coords = fk(q1, kin_system, ref_coords)  # [B, N, 3]
    
    # Convert back to latent space (normalize by scales)
    # NOTE: This assumes DefaultPLCoordinateConverter
    x1 = coords_to_latent(x1_coords, kin_system, latent_converter)
    
    return noisy_x0, x1, q0, q1


def coords_to_latent(
    coords: torch.Tensor,
    kin_system: Any,
    latent_converter: Any
) -> torch.Tensor:
    """
    Convert flat coordinates to latent representation.
    
    Args:
        coords: [B, N, 3] Cartesian coordinates
        kin_system: KinematicSystem with masks
        latent_converter: LatentCoordinateConverter with scales
    
    Returns:
        latent: [B, N_latent, 3] latent coordinates (normalized)
    """
    batch_size = coords.shape[0]
    device = coords.device
    
    lig_mask = kin_system.ligand_mask
    prot_mask = kin_system.protein_mask
    
    # Separate ligand and protein
    lig_coords = coords[:, lig_mask]  # [B, N_lig, 3]
    prot_coords = coords[:, prot_mask]  # [B, N_prot, 3]
    
    # Compute centroids for normalization
    if lig_coords.shape[1] > 0:
        centroid = lig_coords.mean(dim=1, keepdim=True)  # [B, 1, 3]
    else:
        centroid = prot_coords.mean(dim=1, keepdim=True)
    
    # Center coordinates
    lig_coords_centered = lig_coords - centroid
    prot_coords_centered = prot_coords - centroid
    
    # Normalize by scales
    ca_scale = latent_converter.ca_scale
    other_scale = latent_converter.other_scale
    
    # For simplicity, treat ligand as "other" scale
    lig_latent = lig_coords_centered / other_scale
    prot_latent = prot_coords_centered / ca_scale  # Simplified
    
    # Concatenate in expected order
    latent = torch.cat([prot_latent, lig_latent], dim=1)
    
    return latent


def sample_kpfm_interpolated(
    q0: Any,
    q1: Any,
    t: torch.Tensor,
    kin_system: Any,
    latent_converter: Any
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute KPFM interpolated state and target velocity.
    
    This is used during training to get:
    1. x_t: Interpolated coordinates at time t
    2. v_target: Target velocity in atom space (J @ dq_target)
    
    Args:
        q0: Target DOF state (holo)
        q1: Prior DOF state
        t: [B, 1] interpolation time
        kin_system: KinematicSystem
        latent_converter: LatentCoordinateConverter
    
    Returns:
        x_t: [B, N_latent, 3] interpolated latent coordinates
        v_target: [B, 3N, 1] target velocity in atom space
    """
    from flowdock.data.components.dof_utils import geodesic_interpolate, geodesic_velocity
    from flowdock.models.components.kinematics import (
        ForwardKinematics, SparseJacobianBuilder
    )
    
    batch_size = q0.translation.shape[0]
    device = q0.translation.device
    
    # Interpolate in DOF space
    q_t = geodesic_interpolate(q0, q1, t)
    
    # Forward kinematics to get coordinates
    fk = ForwardKinematics()
    ref_coords = kin_system.reference_coords
    if ref_coords.dim() == 2:
        ref_coords = ref_coords.unsqueeze(0).expand(batch_size, -1, -1)
    
    x_t = fk(q_t, kin_system, ref_coords)
    
    # Compute target DOF velocity: dq = (q0 - q_t) / (1 - t)
    # (Note: q0 is target/holo, we're going from prior q1 at t=1 to q0 at t=0)
    dq_target = geodesic_velocity(q_t, q0, dt=(1 - t.squeeze(-1)).clamp(min=1e-6))
    
    # Build Jacobian at current state
    jacobian_builder = SparseJacobianBuilder(use_sparse=False)
    J = jacobian_builder(x_t, kin_system, return_sparse=False)  # [B, 3N, M]
    
    # Target velocity in atom space: v = J @ dq
    dq_flat = dq_target.to_flat().unsqueeze(-1)  # [B, M, 1]
    v_target = torch.bmm(J, dq_flat)  # [B, 3N, 1]
    
    # Convert x_t to latent
    x_t_latent = coords_to_latent(x_t, kin_system, latent_converter)
    
    return x_t_latent, v_target
