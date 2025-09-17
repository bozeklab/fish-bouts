import h5py
import numpy as np
import numpy.ma as ma # masked arrays

with h5py.File("Datasets/JM_data/filtered_jmpool_kin.h5") as f:
    print(list(f.keys()))
    phi_smooth_allcond = ma.array(f['orientation_smooth']) #(463, 11651, 175)
    lengths_all = np.array(f['MetaData/lengths_data'], dtype=int) # (463,)
    maxL = np.max(lengths_all) # =11651
with h5py.File('Datasets/JM_data/kmeans_labels_K5_N1200_s8684.h5', 'r') as f:
    labels_fish = ma.array(f['labels_fish'],dtype=int) #(463, 11651)

labels_all = ma.concatenate(labels_fish, axis=0) # (463,)

phi_smooth_allcond[phi_smooth_allcond == 0] = ma.masked # (463, 11651, 175)

print(f"{ma.argmin(phi_smooth_allcond)=}, {ma.argmax(phi_smooth_allcond)=}")
print(f"{phi_smooth_allcond.shape=}")
# Kinematics calculation
# the line below is redundant since maxL is the current length of the second dimension
# phi_smooth_allcond = phi_smooth_allcond[:,:maxL,:] #(463, 11651, 175) #only take the first maxL bouts

phis_all = ma.concatenate(phi_smooth_allcond,axis=0) # (5394413, 175)
delphi = (ma.abs(phis_all[1:,0] - phis_all[:-1,0]))*(180/np.pi) # (5394412,) for each fsh and each bout the dfference n ortneaton between ths bout and the next one at 0th frame

print(f"{phi_smooth_allcond[0, 0, :]=}")
print(f"{phi_smooth_allcond[0, 1, 0]=}")


psi = ma.zeros(phis_all.shape[0])
psi[:-1] = delphi
psi[-1] = ma.masked

psi = psi.reshape(phi_smooth_allcond.shape[0],phi_smooth_allcond.shape[1])
print(psi.shape)

K=5
meanKphi_fish = [ma.abs(psi[:,k:k+K]).mean(axis=1) for k in range(len(psi[0])-K)]
meanKphi_fish = ma.vstack(meanKphi_fish).T

meanKphi = -1*ma.ones((psi.shape[0],psi.shape[1]))
meanKphi[:,2:-3] = meanKphi_fish
print(meanKphi.shape)

meanKphi[meanKphi==-1] = ma.masked
meanKphi_all = ma.hstack(meanKphi)
phi_labels = np.asarray([ma.mean(meanKphi_all[labels_all==kl]) for kl in range(1200)])