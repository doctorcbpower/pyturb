from .measure_turbulence_fft import MeasureVelocityFieldFFT
from .measure_turbulence_wavelets import MeasureVelocityFieldWavelets
from .measure_turbulence_kinematics import MeasureVelocityFieldKinematics

class MeasureVelocityField:
    def __init__(self):
        pass
    def fft(self, *args, verbose=False, **kwargs):
        if verbose:
            print("Using FFT approach") 
        return MeasureVelocityFieldFFT(*args, **kwargs)
    def wavelets(self, *args, verbose=False, **kwargs):
        if verbose:
            print("Using wavelet approach")
        return MeasureVelocityFieldWavelets(*args, **kwargs)
    def kinematics(self,*args,verbose=False,**kwargs):
        if verbose:
            print("Using kinematics method")
        return MeasureVelocityFieldKinematics(*args,**kwargs)