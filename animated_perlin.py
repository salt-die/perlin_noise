import numpy as np
from scipy import ndimage as nd

def perlin(shape, angles, frequency=15):
    shape = np.array(shape)
    zoom = shape // frequency
    pad = -shape % zoom
    width = np.linspace(0, frequency, shape[1], endpoint=False)
    height = np.linspace(0, frequency, shape[0], endpoint=False)
    coords = np.dstack(np.meshgrid(width, height)) % 1
    random_vectors = np.dstack([np.cos(angles)**2, np.sin(angles)**2])
    grid = np.kron(random_vectors,
                   np.ones([zoom[0], zoom[1], 1]))[pad[0]:,pad[1]:,]
    noise = np.einsum('ijk, ijk -> ij', grid, coords)
    return nd.gaussian_filter(noise, sigma=zoom, mode='wrap')

if __name__ == "__main__":
    path='/home/salt/Documents/Python/perlin/frames/'
    shape = np.array([200, 200])
    frequency = 15
    zoom = shape // frequency
    pad = -shape % zoom
    angles = np.random.random((shape + pad) // zoom) * 2 * np.pi

    #Frames
    import matplotlib.pyplot as plt
    plt.axis('off')
    for i, val in enumerate(np.linspace(0, 2 * np.pi, 100,)):
        plt.imsave(path + f'Frame{i:03d}.png',
                   perlin(shape, (angles + val) % (2 * np.pi), frequency))

    #Stitch together
    import os
    import imageio
    images = []
    for file_name in sorted(os.listdir(path)):
        if file_name.endswith('.png'):
            file_path = os.path.join(path, file_name)
            images.append(imageio.imread(file_path))
            imageio.mimsave(path + 'animated_perlin.gif', images, duration=.1)





