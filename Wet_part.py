import scipy.io
import numpy as np
import scipy.misc
from matplotlib import pyplot as plt
from PIL import Image
from scipy import fftpack

#### functions ####

def save_as_img(PSF, index, name, my_format):
    PSF_256 = PSF * 255
    im_PSF = Image.fromarray(PSF_256)
    im_PSF = im_PSF.convert('RGB')
    name_PSF = name + str(index)
    im_PSF.save(name_PSF + my_format)

def normalize(mat):
    return mat / mat.max()

def calculate_psnr(im1, im2):
  MSE = np.mean((im1 - im2) ** 2)
  max_value = im1.max()
  PSNR = 20 * np.log10(max_value / (np.sqrt(MSE)))
  return PSNR


# question 1.1 #


img = plt.imread("DIPSourceHW1.jpg")
trajectories = "100_motion_paths.mat"
my_format = ".PNG"
mat = scipy.io.loadmat(trajectories)

img_np = np.array(img)
img_np = img_np[:, :, 0]
img_np = normalize(img_np)
num_points = mat["X"].shape[1]
num_trajectories = mat["X"].shape[0]


# plotting all trajectories, uncomment for plotting #

# for trajectory in range(num_trajectories):
#     plt.plot(mat["X"][trajectory], mat["Y"][trajectory])
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.show()


# question 1.2 + 1.3 #

MY_IMG = fftpack.fft2(img_np)
dim_0 = img.shape[0]
dim_1 = img.shape[1]

half_picture = (dim_0 / 2, dim_1 / 2)  # for shifting the psf graph to center
RATIO = 3  # multiplying X,Y values by ratio in order to receive a bigger handshake movement

PSF = np.zeros((int(img.shape[0]), int(img.shape[1])))

gs = []
Gs = []

for trajectory in range(num_trajectories):
    for point in range(num_points):
        curr_point = (RATIO * mat["X"][trajectory][point], RATIO * mat["Y"][trajectory][point])
        curr_point = (curr_point[0] + half_picture[0], curr_point[1] + half_picture[1])
        round_curr_point = int(round(curr_point[0])), int(round(curr_point[1]))
        PSF[dim_1 - round_curr_point[1],round_curr_point[0]] += 1

    save_as_img(PSF, trajectory, "PSF_", my_format)
    PSF = PSF / PSF.sum()
    H = fftpack.fft2(PSF)
    G = MY_IMG * H
    g = fftpack.ifft2(G).real
    g = fftpack.ifftshift(g)
    g = normalize(g)
    gs.append(g)
    Gs.append(G)
    save_as_img(g, trajectory, "g_", my_format)



# part 2 #

magnitudes = []
for i in range(num_trajectories):
    G_mag = abs(Gs[i]) ** 30
    magnitudes.append(G_mag)

## uncomment if we want to use huristic of picking the max frequency ##
# max_G = np.zeros(((256,256)))
# all_mags = np.zeros((num_trajectories, 256, 256))
# for frame in range(num_trajectories):
#   all_mags[frame] = magnitudes[frame]



PSNRs = []
F =np.zeros(((dim_0,dim_1)),dtype = 'complex128')
for num_traj in range(num_trajectories):

    ## uncomment if we want to use huristic of picking the max frequency ##
    # max_G = np.argmax(all_mags[0:num_traj + 1],axis=0)
    # for i in range(256):
    #     for j in range(256):
    #         F[i, j] = Gs[max_G[i, j]][i, j]

    sum_of_mags = np.zeros((dim_0, dim_1))
    for k in range(num_traj+1):
        sum_of_mags = sum_of_mags + magnitudes[k]

    Gs_weighted = []
    for i in range(num_traj + 1):
        G_weighted = (magnitudes[i] * Gs[i]) / sum_of_mags
        Gs_weighted.append(G_weighted)

    for curr_num_traj in range(num_traj + 1):
        F = F + Gs_weighted[curr_num_traj]

    f = fftpack.ifft2(F)
    f = fftpack.fftshift(f)
    f = np.array(f)
    f = f / f.max()
    f = f.real

    save_as_img(f, num_traj, "f_", my_format)
    curr_PSNR = calculate_psnr(img_np, f)
    PSNRs.append(curr_PSNR)


plt.plot(PSNRs)
plt.xlabel("Number of frames")
plt.ylabel("PSNR value")
plt.savefig('PSNR.png')