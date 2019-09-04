"""
This is a hack-y version of the perlin noise algorithm.  There's no hash
function, we calculate random normalized vectors on-the-fly instead. These
random vectors should at least reduce directional artifacts -- we do suffer
extra computational overhead. We interpolate with nd.gaussian_filter, which is
an interpolation through convolution.

The goal is simply to see if I could mimick perlin noise with an image filter.
"""

import numpy as np
from scipy import ndimage as nd

def perlin(shape, frequency=15):
    shape = np.array(shape)
    zoom = shape // frequency  #Resolution of our grid of random vectors
    #Padding accounts for when frequency doesn't evenly divide the shape.
    pad = -shape % zoom

    #Our internal coordinates
    width = np.linspace(0, frequency, shape[1], endpoint=False)
    height = np.linspace(0, frequency, shape[0], endpoint=False)
    coords = np.dstack(np.meshgrid(width, height)) % 1

    #Our grid of random normalized directional vectors
    angles = np.random.random((shape + pad) // zoom) * 2 * np.pi
    random_vectors = np.dstack([np.cos(angles)**2, np.sin(angles)**2])
    # We use the Kronecker product of random_vectors with an array of ones to
    # repeat the grid points for each internal coordinate. That is, if we have
    # a grid of vectors:
    #
    #      [[a, b],
    #       [c, d]]
    #
    # and internal coordinates:
    #
    #      [[[x1, y1], [x2, y1]],
    #       [[x1. y2], [x2, y2]]]
    #
    # The Kronecker product will give us:
    #
    #       [[a, a, b, b],
    #        [a, a, b, b],
    #        [c, c, d, d],
    #        [c, c, d, d]]
    grid = np.kron(random_vectors,
                   np.ones([zoom[0], zoom[1], 1]))[pad[0]:,pad[1]:,]

    #Dot product of the internal coords with grid of random vectors
    noise = np.einsum('ijk, ijk -> ij', grid, coords)

    #Interpolate the noise with a filter
    return nd.gaussian_filter(noise, sigma=zoom, mode='wrap')

def octave_perlin(shape, octaves=5, persistence=2):
    """
    Fractal perlin noise.
    """
    shape = np.array(shape)
    total = 0
    frequency = 1
    amplitude = 1 / frequency
    max_value = 0
    for _ in range(octaves):
        total += perlin(shape, frequency) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= 2
    return total / max_value

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.axis('off')
    plt.imshow(octave_perlin([200, 200]))