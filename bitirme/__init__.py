"""
INNATE3D for Taylor-Green Vortex 3D

SADECE INNATE nöronlarından oluşan neural operator.
HİÇBİR CNN veya MLP yok!

Kullanım:
    from bitirme import create_model, train

    # Model oluştur
    model = create_model(resolution=32, nu=0.001)

    # Eğit
    from bitirme.train import train, CONFIG
    model, state = train(CONFIG)

    # Değerlendir
    from bitirme.evaluate import evaluate_model
    results = evaluate_model(model, device)
"""

from .model import INNATE3D_TGV, create_model, FluidState3D

__all__ = [
    'INNATE3D_TGV',
    'create_model',
    'FluidState3D',
]
