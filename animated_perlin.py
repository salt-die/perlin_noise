"""
A demonstration of how we can animate the perlin noise using our random vector
generation.
"""

import numpy as np
from scipy import ndimage as nd

def perlin(shape, offset, frequency=15, seed=1):
    """
    Seed and offset added so we can animate.
    """
    shape = np.array(shape)
    zoom = shape // frequency
    pad = -shape % zoom
    samples = (shape + pad) // zoom
    width = np.linspace(0, frequency, shape[1], endpoint=False)
    height = np.linspace(0, frequency, shape[0], endpoint=False)
    coords = np.dstack(np.meshgrid(width, height)) % 1
    np.random.seed(seed) #Seed so we can animate
    #Each vector rotates in random direction
    signs = 2 * np.random.randint(0,  2, samples) - 1
    np.random.seed(seed)
    angles = np.random.random(samples) * 2 * np.pi + offset * signs
    random_vectors = np.dstack([np.cos(angles)**2, np.sin(angles)**2])
    grid = np.kron(random_vectors,
                   np.ones([zoom[0], zoom[1], 1]))[pad[0]:,pad[1]:,]
    noise = np.einsum('ijk, ijk -> ij', grid, coords)
    return nd.gaussian_filter(noise, sigma=zoom, mode='wrap')

def octave_perlin(shape, offset, octaves=5, persistence=2, seed=1):
    """
    Seed and offset added so we can animate.
    """
    shape = np.array(shape)
    total = 0
    frequency = 1
    amplitude = 1 / frequency
    max_value = 0
    for _ in range(octaves):
        total += perlin(shape, offset, frequency) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= 2
    return total / max_value

if __name__ == "__main__":
    path='/home/salt/Documents/Python/perlin/frames/' #Use your own path!
    #Frames
    import matplotlib.pyplot as plt
    plt.axis('off')
    for i, val in enumerate(np.linspace(0, 2 * np.pi, 100,)):
        #plt.imsave(path + f'Frame{i:03d}.png', perlin([200, 200], val))
        plt.imsave(path + f'Frame{i:03d}.png', octave_perlin([200, 200], val))

    #Stitch together
    import os
    import imageio
    images = [imageio.imread(os.path.join(path, file_name))
              for file_name in sorted(os.listdir(path))
              if file_name.endswith('.png')]
    imageio.mimsave(path + 'animated_perlin.gif', images, duration=.1)
