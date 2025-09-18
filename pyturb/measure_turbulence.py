from .measure_turbulence_fft import MeasureVelocityFieldFFT
from .measure_turbulence_wavelets import *
from .measure_turbulence_kinematics import *

class MeasureVelocityField():
    def __init__(self):
        pass
    def fft(self):
        print("Using FFT approach")
        return MeasureVelocityFieldFFT()
    def wavelets(self):
        print("Using wavelet method")
        raise("Option not implemented")
        return
    def kinematics(self):
        print("Using kinematics method")
        return