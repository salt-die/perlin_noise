"""
Slightly modified code from:
https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy

which seems to be pythonized code from:
https://flafla2.github.io/2014/08/09/perlinnoise.html
"""
import numpy as np

def perlin(shape, frequency=5):
    width = np.linspace(0, frequency, shape[0], endpoint=False)
    height = np.linspace(0, frequency, shape[1], endpoint=False)
    x, y = np.meshgrid(width, height)
    # permutation table
    p = np.arange(256,dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    #integer part
    x_int = x.astype(int)
    y_int = y.astype(int)
    #fraction part
    x_frac = x - x_int
    y_frac = y - y_int
    #ease transitions with sigmoid-type function
    fade_x = fade(x_frac)
    fade_y = fade(y_frac)
    # noise components
    n00 = gradient(p[p[x_int] + y_int], x_frac, y_frac)
    n01 = gradient(p[p[x_int] + y_int + 1], x_frac, y_frac - 1)
    n11 = gradient(p[p[x_int + 1] + y_int + 1], x_frac - 1, y_frac - 1)
    n10 = gradient(p[p[x_int + 1] + y_int], x_frac - 1, y_frac)
    # combine noises
    x1 = lerp(n00, n10, fade_x)
    x2 = lerp(n01, n11, fade_x)
    return lerp(x1, x2, fade_y)

def lerp(a, b, x):
    return a + x * (b - a)

def fade(t):
    t_squared = t**2 #Time saver
    return (6 * t_squared - 15 * t + 10) * t * t_squared

def gradient(h, x, y):
    vectors = np.array([[ 0,  1],
                        [ 0, -1],
                        [ 1,  0],
                        [-1,  0]])
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y

def octave_perlin(x, y, octaves=3, persistence=2):
    """
    Fractal perlin noise.
    """
    total = 0
    frequency = 1
    amplitude = 1
    max_value = 0
    for _ in range(octaves):
        total += perlin(x * frequency, y * frequency) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= 2
    return total / max_value


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.axis('off')
    plt.imshow(perlin([100, 100]), origin='upper')
