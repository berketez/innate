"""
Ortak test altyapısı.

İçerir:
    - device: Otomatik cihaz algılama (CUDA/MPS/CPU)
    - dns_solver: SciPy pseudo-spectral DNS referans çözücü
    - trainer_base: Ortak eğitim sınıfı
    - visualizer: Görselleştirme araçları
"""

# Lazy imports to avoid circular dependencies
def get_device():
    from .device import get_device as _get_device
    return _get_device()

def device_info():
    from .device import device_info as _device_info
    return _device_info()

# Direct imports
try:
    from .device import DEVICE
except ImportError:
    DEVICE = None

__all__ = [
    'get_device',
    'device_info', 
    'DEVICE',
    'PseudoSpectralDNS',
    'PseudoSpectralDNS3D',
    'BenchmarkTrainer',
    'ValidationMetrics',
    'FlowVisualizer',
]

def __getattr__(name):
    """Lazy loading for heavy modules."""
    if name == 'PseudoSpectralDNS':
        from .dns_solver import PseudoSpectralDNS
        return PseudoSpectralDNS
    elif name == 'PseudoSpectralDNS3D':
        from .dns_solver import PseudoSpectralDNS3D
        return PseudoSpectralDNS3D
    elif name == 'BenchmarkTrainer':
        from .trainer_base import BenchmarkTrainer
        return BenchmarkTrainer
    elif name == 'ValidationMetrics':
        from .trainer_base import ValidationMetrics
        return ValidationMetrics
    elif name == 'FlowVisualizer':
        from .visualizer import FlowVisualizer
        return FlowVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

