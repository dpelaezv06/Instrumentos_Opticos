import numpy as np
import matplotlib.pyplot as plt

# Parámetros
grid_size = 1024  # Tamaño de la malla
radius = 50       # Radio de la abertura circular
wavelength = 500e-9  # Longitud de onda (en metros)
pixel_size = 10e-6   # Tamaño de cada pixel en el plano espacial

# Dominio espacial
x = np.linspace(-grid_size//2, grid_size//2, grid_size) * pixel_size
y = np.linspace(-grid_size//2, grid_size//2, grid_size) * pixel_size
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# Abertura circular
aperture = np.zeros((grid_size, grid_size))
aperture[R <= radius * pixel_size] = 1

# Visualización de la abertura
plt.figure(figsize=(6, 6))
plt.title("Abertura circular")
plt.imshow(aperture, extent=[x.min(), x.max(), y.min(), y.max()], cmap='gray')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.colorbar(label="Transmisión")
plt.show()

# Transformada de Fourier
fft_aperture = np.fft.fftshift(np.fft.fft2(aperture))  # FFT 2D con cambio de cuadrantes
intensity = np.abs(fft_aperture)**2  # Intensidad del espectro angular

# Coordenadas en el dominio angular
kx = np.fft.fftshift(np.fft.fftfreq(grid_size, d=pixel_size)) * 2 * np.pi
ky = np.fft.fftshift(np.fft.fftfreq(grid_size, d=pixel_size)) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky)
theta = np.arcsin(np.sqrt(KX**2 + KY**2) * wavelength / (2 * np.pi))  # Ángulo de difracción

# Visualización del espectro angular
plt.figure(figsize=(6, 6))
plt.title("Espectro angular (difracción)")
plt.imshow(intensity, extent=[kx.min(), kx.max(), ky.min(), ky.max()], cmap='inferno', norm=plt.Normalize(0, np.percentile(intensity, 99)))
plt.xlabel("kx (1/m)")
plt.ylabel("ky (1/m)")
plt.colorbar(label="Intensidad")
plt.show()
