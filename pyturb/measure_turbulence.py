from .measure_turbulence_fft import MeasureVelocityFieldFFT
from .measure_turbulence_wavelets import MeasureVelocityFieldWavelets
from .measure_turbulence_kinematics import MeasureVelocityFieldKinematics

class MeasureVelocityField():
    def __init__(self):
        pass
    def fft(self,verbose=False):
        if verbose==True:
            print("Using FFT approach") 
        return MeasureVelocityFieldFFT()
    def wavelets(self,verbose=False):
        if verbose==True:
            print("Using wavelet approach")
        return MeasureVelocityFieldWavelets()
    def kinematics(self,*args,verbose=False,**kwargs):
        if verbose==True:
            print("Using kinematics method")
        return MeasureVelocityFieldKinematics(*args,**kwargs)
