"""
Otomatik Cihaz Algılama Modülü

CUDA → MPS → CPU öncelik sırasıyla en uygun cihazı seçer.
Tüm test scriptleri bu modülü kullanmalıdır.

Kullanım:
    from tests.common.device import DEVICE, get_device, device_info
    
    # Global DEVICE kullan
    tensor = torch.randn(64, 64, device=DEVICE)
    
    # Veya dinamik al
    device = get_device()
    
    # Cihaz bilgisi
    print(device_info())
"""

import torch
import platform
import sys
from typing import Dict, Any


def get_device() -> torch.device:
    """
    Otomatik cihaz seçimi: CUDA → MPS → CPU
    
    Returns:
        En uygun torch.device
    
    Öncelik:
        1. CUDA (NVIDIA GPU) - Varsa ve kullanılabilirse
        2. MPS (Apple Silicon) - macOS M1/M2/M3 için
        3. CPU - Fallback
    """
    # CUDA kontrolü
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # MPS kontrolü (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS FFT desteğini kontrol et
        try:
            test_tensor = torch.randn(4, 4, device='mps')
            _ = torch.fft.fft2(test_tensor)
            return torch.device("mps")
        except Exception:
            # FFT desteklenmiyorsa CPU'ya düş
            pass
    
    return torch.device("cpu")


def device_info() -> Dict[str, Any]:
    """
    Cihaz hakkında detaylı bilgi döndür.
    
    Returns:
        Dict: Cihaz bilgileri
    """
    device = get_device()
    info = {
        'device': str(device),
        'device_type': device.type,
        'python_version': sys.version.split()[0],
        'pytorch_version': torch.__version__,
        'platform': platform.system(),
        'platform_version': platform.version(),
        'processor': platform.processor(),
    }
    
    if device.type == 'cuda':
        info.update({
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_count': torch.cuda.device_count(),
            'gpu_memory_total': f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB",
        })
    elif device.type == 'mps':
        info.update({
            'mps_available': torch.backends.mps.is_available(),
            'mps_built': torch.backends.mps.is_built(),
        })
    
    return info


def print_device_info() -> None:
    """Cihaz bilgilerini güzel formatta yazdır."""
    info = device_info()
    
    print("=" * 60)
    print("CIHAZ BILGILERI")
    print("=" * 60)
    
    print(f"  Device:          {info['device']}")
    print(f"  PyTorch:         {info['pytorch_version']}")
    print(f"  Python:          {info['python_version']}")
    print(f"  Platform:        {info['platform']}")
    
    if info['device_type'] == 'cuda':
        print(f"  GPU:             {info['gpu_name']}")
        print(f"  GPU Memory:      {info['gpu_memory_total']}")
        print(f"  CUDA:            {info['cuda_version']}")
    elif info['device_type'] == 'mps':
        print(f"  Apple Silicon:   MPS Enabled")
    else:
        print(f"  Processor:       {info['processor']}")
    
    print("=" * 60)


def setup_device_optimizations() -> None:
    """
    Cihaza özel optimizasyonları uygula.
    
    Bu fonksiyon test scriptlerinin başında çağrılmalı.
    """
    device = get_device()
    
    # Default dtype (MPS ve genel uyumluluk için float32)
    torch.set_default_dtype(torch.float32)
    
    if device.type == 'cuda':
        # CUDA optimizasyonları
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    elif device.type == 'mps':
        # MPS için özel ayarlar (gerekirse)
        pass


# Global DEVICE - import edildiğinde otomatik ayarlanır
DEVICE = get_device()


if __name__ == "__main__":
    print_device_info()

