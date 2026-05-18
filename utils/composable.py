"""
Composable Neuron Wrappers - Nöron Birleştirme Yardımcıları

Bu modül INNATE nöronlarını birleştirmek için wrapper sınıfları sağlar.
PyTorch'un nn.Sequential, nn.ModuleList benzeri ama fluid-specific.

ÖNEMLİ: Bu wrapper'lar FluidState/FluidState3D döndüren nöronlar içindir.
         Advection, Vorticity gibi tuple (RHS) döndüren nöronlar için
         INNATE.step() veya TimeMarcher kullanın.

Kullanım:
    from innate import INNATE, TimeMarcher, DataInjector
    from utils import Sequential, Parallel, Residual, MultiBranch, Branch

    # Sequential: FluidState döndüren nöronlar için
    # (TimeMarcher, DataInjector, Boundary, vb.)

    # Parallel: Birden fazla modeli paralel çalıştırıp birleştir
    ensemble = Parallel(model1, model2, fusion='weighted')

    # MultiBranch: TGV3D için multi-branch ensemble
    model = MultiBranch([
        Branch([model1], name='branch1'),
        Branch([model2], name='branch2'),
    ])

NOT: Bu wrapper'lar hem 2D (FluidState) hem 3D (FluidState3D) destekler.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Callable, Union, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from innate import FluidState, FluidState3D


def _get_fluid_state_classes():
    """Lazy import for FluidState and FluidState3D to avoid circular imports."""
    from innate import FluidState, FluidState3D
    return FluidState, FluidState3D


def _is_fluid_state(obj) -> bool:
    """Check if object is a FluidState (2D or 3D)."""
    return hasattr(obj, 'u') and hasattr(obj, 'v') and hasattr(obj, 't')


def _is_3d_state(obj) -> bool:
    """Check if object is a FluidState3D (has w component)."""
    return hasattr(obj, 'w') and hasattr(obj, 'omega_x')


def _update_state_velocities_2d(state, u_new: torch.Tensor, v_new: torch.Tensor,
                                 vorticity_new: Optional[torch.Tensor] = None):
    """
    2D FluidState'in u, v ve opsiyonel vorticity alanlarını güncelle.
    Yeni bir FluidState döndürür (immutable pattern).
    """
    FluidState, _ = _get_fluid_state_classes()
    return FluidState(
        u=u_new,
        v=v_new,
        p=state.p if hasattr(state, 'p') else torch.zeros_like(u_new),
        vorticity=vorticity_new if vorticity_new is not None else state.vorticity,
        t=state.t
    )


def _update_state_velocities_3d(state, u_new: torch.Tensor, v_new: torch.Tensor, w_new: torch.Tensor,
                                 omega_x_new: Optional[torch.Tensor] = None,
                                 omega_y_new: Optional[torch.Tensor] = None,
                                 omega_z_new: Optional[torch.Tensor] = None):
    """
    3D FluidState3D'nin u, v, w ve opsiyonel omega alanlarını güncelle.
    Yeni bir FluidState3D döndürür (immutable pattern).
    """
    _, FluidState3D = _get_fluid_state_classes()
    return FluidState3D(
        u=u_new,
        v=v_new,
        w=w_new,
        p=state.p if hasattr(state, 'p') else torch.zeros_like(u_new),
        omega_x=omega_x_new if omega_x_new is not None else state.omega_x,
        omega_y=omega_y_new if omega_y_new is not None else state.omega_y,
        omega_z=omega_z_new if omega_z_new is not None else state.omega_z,
        t=state.t
    )


def _broadcast_weights(weights: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    """
    Weights'i tensor shape'ine göre broadcast et.

    Regular weights [num]:
        2D: [num] -> [num, 1, 1, 1] for [num, B, H, W]
        3D: [num] -> [num, 1, 1, 1, 1] for [num, B, Nx, Ny, Nz]

    Attention weights [num, B]:
        2D: [num, B] -> [num, B, 1, 1] for [num, B, H, W]
        3D: [num, B] -> [num, B, 1, 1, 1] for [num, B, Nx, Ny, Nz]
    """
    # Kaç tane 1 eklememiz gerekiyor?
    # tensor.dim() - weights.dim() = eksik boyut sayısı
    num_unsqueeze = tensor.dim() - weights.dim()
    for _ in range(num_unsqueeze):
        weights = weights.unsqueeze(-1)
    return weights


class Sequential(nn.Module):
    """
    Nöronları sıralı olarak bağlar.

    SADECE FluidState/FluidState3D döndüren nöronlar için!

    NOT: Advection, Vorticity gibi tuple (RHS/türev) döndüren nöronlar
    bu wrapper ile KULLANILAMAZ. Bu nöronlar için INNATE.step() veya
    TimeMarcher kullanın.

    Args:
        *neurons: FluidState döndüren INNATE nöronları (sıralı)

    Örnek:
        # Doğru kullanım - FluidState döndüren nöronlar:
        seq = Sequential(
            Projection(64),
            TimeMarcher(64)
        )
        output = seq(state)

        # YANLIŞ - tuple döndüren nöronlar:
        # seq = Sequential(Advection(64), Vorticity(64))  # ÇALIŞMAZ
    """

    def __init__(self, *neurons: nn.Module):
        super().__init__()
        self.neurons = nn.ModuleList(neurons)

    def forward(self, state, **kwargs):
        """Her nöronu sırayla uygula."""
        for neuron in self.neurons:
            result = neuron(state, **kwargs) if kwargs else neuron(state)

            # Sadece FluidState döndürenleri state olarak kullan
            if _is_fluid_state(result):
                state = result
            # Tuple çıktılar (Advection, Vorticity, vb.) desteklenmez
            # Bu nöronlar RHS (türev) döndürür, yeni velocity değil!

        return state

    def extra_repr(self) -> str:
        return f"num_neurons={len(self.neurons)}"


class Parallel(nn.Module):
    """
    Nöronları paralel çalıştırıp sonuçları birleştirir.

    Tüm nöronlar aynı girdiyi alır, çıktılar fusion ile birleştirilir.

    Args:
        *neurons: INNATE nöronları (paralel)
        fusion: Birleştirme yöntemi ('mean', 'sum', 'weighted', 'attention')
        learnable_weights: fusion='weighted' için öğrenilebilir ağırlıklar

    Örnek:
        par = Parallel(
            Advection(64),
            Vorticity(64),
            fusion='weighted'
        )
        output = par(state)
    """

    def __init__(
        self,
        *neurons: nn.Module,
        fusion: str = 'mean',
        learnable_weights: bool = True,
        resolution: int = 64
    ):
        super().__init__()
        self.neurons = nn.ModuleList(neurons)
        self.fusion = fusion
        self.num_neurons = len(neurons)
        self.resolution = resolution

        if fusion == 'weighted' and learnable_weights:
            # Öğrenilebilir ağırlıklar (softmax ile normalize edilecek)
            self.weights = nn.Parameter(torch.ones(self.num_neurons))
        elif fusion == 'attention':
            # Attention: hem attention projector hem de fallback weights oluştur
            # Query: global velocity magnitude -> per-neuron attention scores
            self.attention_proj = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, self.num_neurons)
            )
            # Fallback için base weights
            self.weights = nn.Parameter(torch.ones(self.num_neurons))

    def forward(self, state, **kwargs):
        """Tüm nöronları paralel çalıştır ve birleştir."""
        outputs = []
        for neuron in self.neurons:
            result = neuron(state, **kwargs) if kwargs else neuron(state)
            outputs.append(result)

        # Fusion
        if self.fusion == 'mean':
            return self._fuse_mean(outputs)
        elif self.fusion == 'sum':
            return self._fuse_sum(outputs)
        elif self.fusion == 'weighted':
            return self._fuse_weighted(outputs)
        elif self.fusion == 'attention':
            return self._fuse_attention(outputs, state)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion}")

    def _fuse_mean(self, outputs):
        """Basit ortalama."""
        # Tuple çıktılar için her elemanın ortalaması
        if isinstance(outputs[0], tuple):
            n = len(outputs[0])
            return tuple(
                sum(o[i] for o in outputs) / len(outputs)
                for i in range(n)
            )
        return sum(outputs) / len(outputs)

    def _fuse_sum(self, outputs):
        """Toplam."""
        if isinstance(outputs[0], tuple):
            n = len(outputs[0])
            return tuple(sum(o[i] for o in outputs) for i in range(n))
        return sum(outputs)

    def _fuse_weighted(self, outputs):
        """Öğrenilebilir ağırlıklı birleştirme."""
        weights = torch.softmax(self.weights, dim=0)

        if isinstance(outputs[0], tuple):
            n = len(outputs[0])
            return tuple(
                sum(w * o[i] for w, o in zip(weights, outputs))
                for i in range(n)
            )
        return sum(w * o for w, o in zip(weights, outputs))

    def _fuse_attention(self, outputs, state):
        """Attention-based fusion (state-dependent weights). 2D/3D uyumlu."""
        # Velocity magnitude'dan state-dependent attention weights hesapla
        # 2D için mean(dim=(-2,-1)), 3D için mean(dim=(-3,-2,-1))
        if _is_3d_state(state):
            vel_mag = state.velocity_magnitude().mean(dim=(-3, -2, -1))  # [B]
        else:
            vel_mag = state.velocity_magnitude().mean(dim=(-2, -1))  # [B]

        # [B] -> [B, 1] for linear layer
        if vel_mag.dim() == 0:
            vel_mag = vel_mag.unsqueeze(0).unsqueeze(0)  # [] -> [1, 1]
        elif vel_mag.dim() == 1:
            vel_mag = vel_mag.unsqueeze(-1)  # [B] -> [B, 1]

        # Attention scores: [B, num_neurons]
        attn_logits = self.attention_proj(vel_mag)
        # Base weights ile birleştir
        combined_logits = attn_logits + self.weights.unsqueeze(0)
        attn_weights = torch.softmax(combined_logits, dim=-1)  # [B, num_neurons]

        # Weighted sum of outputs
        if isinstance(outputs[0], tuple):
            n = len(outputs[0])
            fused = []
            for i in range(n):
                # Stack tensors: [num_neurons, B, H, W] or [num_neurons, B, Nx, Ny, Nz]
                stacked = torch.stack([o[i] for o in outputs], dim=0)
                # Dynamic broadcast for 2D/3D
                w = _broadcast_weights(attn_weights.t(), stacked)
                weighted = (stacked * w).sum(dim=0)
                fused.append(weighted)
            return tuple(fused)
        else:
            # FluidState output
            stacked_u = torch.stack([o.u for o in outputs], dim=0)
            w = _broadcast_weights(attn_weights.t(), stacked_u)
            u_fused = (stacked_u * w).sum(dim=0)

            stacked_v = torch.stack([o.v for o in outputs], dim=0)
            v_fused = (stacked_v * w).sum(dim=0)

            # 3D check
            if _is_3d_state(outputs[0]):
                stacked_w_vel = torch.stack([o.w for o in outputs], dim=0)
                w_vel_fused = (stacked_w_vel * w).sum(dim=0)

                # Omega fusion
                omega_x_stacked = torch.stack([o.omega_x for o in outputs], dim=0)
                omega_y_stacked = torch.stack([o.omega_y for o in outputs], dim=0)
                omega_z_stacked = torch.stack([o.omega_z for o in outputs], dim=0)
                omega_x_fused = (omega_x_stacked * w).sum(dim=0)
                omega_y_fused = (omega_y_stacked * w).sum(dim=0)
                omega_z_fused = (omega_z_stacked * w).sum(dim=0)

                return _update_state_velocities_3d(
                    outputs[0], u_fused, v_fused, w_vel_fused,
                    omega_x_fused, omega_y_fused, omega_z_fused
                )
            else:
                # Vorticity fusion (2D)
                vort_stacked = torch.stack([o.vorticity for o in outputs], dim=0)
                vort_fused = (vort_stacked * w).sum(dim=0)
                return _update_state_velocities_2d(outputs[0], u_fused, v_fused, vort_fused)

    def extra_repr(self) -> str:
        w_str = ""
        if hasattr(self, 'weights'):
            w = torch.softmax(self.weights, dim=0)
            w_str = f", weights=[{', '.join(f'{x:.2f}' for x in w.tolist())}]"
        return f"num_neurons={self.num_neurons}, fusion={self.fusion}{w_str}"


class Residual(nn.Module):
    """
    Skip connection wrapper.

    output = input + scale * neuron(input)

    Args:
        neuron: Ana nöron
        scale: Residual ölçeği (öğrenilebilir veya sabit)
        learnable_scale: scale'i öğrenilebilir yap

    Örnek:
        block = Residual(Advection(64), scale=0.1)
        output = block(state)  # state + 0.1 * advection(state)
    """

    def __init__(
        self,
        neuron: nn.Module,
        scale: float = 1.0,
        learnable_scale: bool = False
    ):
        super().__init__()
        self.neuron = neuron

        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(scale))
        else:
            self.register_buffer('scale', torch.tensor(scale))

    def forward(self, state, **kwargs):
        """Residual connection uygula."""
        result = self.neuron(state, **kwargs) if kwargs else self.neuron(state)

        # FluidState/FluidState3D için residual (interpolation)
        # output = (1 - scale) * input + scale * result
        # = input + scale * (result - input)
        if _is_fluid_state(result):
            u_new = state.u + self.scale * (result.u - state.u)
            v_new = state.v + self.scale * (result.v - state.v)

            # 3D check
            if _is_3d_state(result):
                w_new = state.w + self.scale * (result.w - state.w)
                return _update_state_velocities_3d(state, u_new, v_new, w_new)
            else:
                return _update_state_velocities_2d(state, u_new, v_new)

        # Tuple çıktılar için (u_delta, v_delta[, w_delta]) - delta semantics
        if isinstance(result, tuple) and len(result) >= 2:
            u_delta, v_delta = result[0], result[1]
            u_new = state.u + self.scale * u_delta
            v_new = state.v + self.scale * v_delta

            # 3D tuple: (u, v, w, ...)
            if len(result) >= 3 and _is_3d_state(state):
                w_delta = result[2]
                w_new = state.w + self.scale * w_delta
                return (u_new, v_new, w_new) + result[3:] if len(result) > 3 else (u_new, v_new, w_new)

            return (u_new, v_new) + result[2:] if len(result) > 2 else (u_new, v_new)

        return result

    def extra_repr(self) -> str:
        return f"scale={self.scale.item():.4f}"


class Branch(nn.Module):
    """
    Multi-branch yapısı için tek bir branch.

    Sequential + activation + optional normalization.

    SADECE FluidState/FluidState3D döndüren nöronlar/modeller için!
    Advection, Vorticity gibi tuple döndüren nöronlar desteklenmez.

    Args:
        neurons: FluidState döndüren nöronlar/modeller
        activation: Ara aktivasyon ('none', 'tanh', 'softplus')
        name: Branch adı (debug için)

    Örnek:
        # INNATE modeli veya FluidState döndüren nöronlarla:
        branch = Branch(
            [TimeMarcher(64), DataInjector(64)],
            activation='tanh',
            name='time_branch'
        )
    """

    def __init__(
        self,
        neurons: List[nn.Module],
        activation: str = 'none',
        name: str = 'branch'
    ):
        super().__init__()
        self.name = name
        self.neurons = nn.ModuleList(neurons)

        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'softplus':
            self.activation = nn.functional.softplus
        else:
            self.activation = None

    def forward(self, state, **kwargs):
        """Branch'i çalıştır."""
        for neuron in self.neurons:
            result = neuron(state, **kwargs) if kwargs else neuron(state)

            # Aktivasyon (velocity fields için)
            if self.activation is not None and isinstance(result, tuple):
                result = tuple(self.activation(r) for r in result)

            if hasattr(result, 'u'):
                state = result

        return state

    def extra_repr(self) -> str:
        act = self.activation.__name__ if self.activation else 'none'
        return f"name={self.name}, neurons={len(self.neurons)}, activation={act}"


class MultiBranch(nn.Module):
    """
    Multi-branch ensemble - TGV3D için ana yapı.

    Birden fazla branch'i paralel çalıştırıp fusion ile birleştirir.
    Quantum-inspired superposition: softmax ağırlıkları.

    Args:
        branches: Branch listesi
        fusion: Birleştirme yöntemi

    Örnek:
        model = MultiBranch([
            Branch([Advection(64), Projection(64)], name='adv'),
            Branch([Vorticity(64), Projection(64)], name='vort'),
        ])
    """

    def __init__(
        self,
        branches: List[Branch],
        fusion: str = 'weighted'
    ):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.num_branches = len(branches)

        # Öğrenilebilir branch ağırlıkları
        self.branch_logits = nn.Parameter(torch.zeros(self.num_branches))

    @property
    def branch_weights(self) -> torch.Tensor:
        """Softmax normalize edilmiş branch ağırlıkları."""
        return torch.softmax(self.branch_logits, dim=0)

    def forward(self, state, **kwargs):
        """Tüm branch'leri çalıştır ve birleştir. 2D/3D uyumlu."""
        outputs = []
        for branch in self.branches:
            result = branch(state, **kwargs) if kwargs else branch(state)
            outputs.append(result)

        # Weighted fusion
        weights = self.branch_weights  # [num_branches]

        # FluidState/FluidState3D outputs için weighted fusion
        if outputs and _is_fluid_state(outputs[0]):
            # Stack velocities: [num_branches, B, H, W] or [num_branches, B, Nx, Ny, Nz]
            u_stacked = torch.stack([o.u for o in outputs], dim=0)
            v_stacked = torch.stack([o.v for o in outputs], dim=0)

            # Dynamic broadcast for 2D/3D
            w = _broadcast_weights(weights, u_stacked)

            # Weighted sum - velocities
            u_fused = (u_stacked * w).sum(dim=0)
            v_fused = (v_stacked * w).sum(dim=0)

            # 3D check
            if _is_3d_state(outputs[0]):
                w_vel_stacked = torch.stack([o.w for o in outputs], dim=0)
                w_vel_fused = (w_vel_stacked * w).sum(dim=0)

                # Omega fusion
                omega_x_stacked = torch.stack([o.omega_x for o in outputs], dim=0)
                omega_y_stacked = torch.stack([o.omega_y for o in outputs], dim=0)
                omega_z_stacked = torch.stack([o.omega_z for o in outputs], dim=0)
                omega_x_fused = (omega_x_stacked * w).sum(dim=0)
                omega_y_fused = (omega_y_stacked * w).sum(dim=0)
                omega_z_fused = (omega_z_stacked * w).sum(dim=0)

                return _update_state_velocities_3d(
                    outputs[0], u_fused, v_fused, w_vel_fused,
                    omega_x_fused, omega_y_fused, omega_z_fused
                )
            else:
                # Vorticity fusion (2D)
                vort_stacked = torch.stack([o.vorticity for o in outputs], dim=0)
                vort_fused = (vort_stacked * w).sum(dim=0)
                return _update_state_velocities_2d(outputs[0], u_fused, v_fused, vort_fused)

        # Tuple outputs için weighted fusion
        elif outputs and isinstance(outputs[0], tuple):
            n = len(outputs[0])
            fused = []
            for i in range(n):
                stacked = torch.stack([o[i] for o in outputs], dim=0)
                w = _broadcast_weights(weights, stacked)
                fused.append((stacked * w).sum(dim=0))
            return tuple(fused)

        # Fallback - tek branch varsa direkt döndür
        return outputs[0] if outputs else state

    def extra_repr(self) -> str:
        weights = self.branch_weights.tolist()
        names = [b.name for b in self.branches]
        w_str = ', '.join(f'{n}:{w:.2f}' for n, w in zip(names, weights))
        return f"branches=[{w_str}]"
