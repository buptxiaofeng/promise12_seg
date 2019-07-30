import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure, morphology

def plot_3d(image, threshold = -300):
    p = image.transpose(2, 1, 0)
    verts, faces = measure.marching_cubes(p, threshold)
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111, projection = "3d")
    mesh = Poly3DCollection(verts[faces], alpha = 0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()

