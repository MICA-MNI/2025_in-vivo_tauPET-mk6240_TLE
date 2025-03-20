#!/usr/bin/env python
# coding: utf-8

# # In-vivo evidence for increased tau deposition in temporal lobe epilepsy
# 
# ### Content
# 1. Participants
# 1.  *Figure 1:* Mapping of tau uptake in TLE
# 1.  *Figure 2:* Network contextualization
# 1.  *Figure 3:* Relationship to clinical and cognitive variables
# 1. Supplementary figures
# 

# In[1]:


# Libraries
import os
import requests
import tempfile
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
from IPython.display import display
from scipy.stats import ttest_ind
from scipy.interpolate import griddata
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import MixedEffect, FixedEffect
from brainstat.datasets import fetch_mask, fetch_template_surface
from brainspace.mesh.mesh_io import read_surface
from brainspace.plotting import plot_hemispheres
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Load utilities functions from utils.py
from utils import *


# ### Load data

# In[2]:


osf_path='/home/bic/rcruces/Desktop'

# Load the data frame
df = pd.read_csv(f'{osf_path}/18F-MK6240_database.csv')

# Load processed 18F-MK6240 PET data | matrix:{vertices x subjects}
pet_raw = np.load(f'{osf_path}/surf-fsLR-32k_desc-GroupData_smooth-10mm_pvc-probGM_ref-cerebellarGM_trc-18Fmk6240_pet.npy')

# Load the cortical thickness data | matrix:{vertices x subjects}
cth_raw = np.load(f'{osf_path}/surf-fsLR-32k_desc-GroupData_smooth-20mm_thickness.npy')

# Load fsLR-32 surface
fslr32k_lh, fslr32k_rh = fetch_template_surface("fslr32k", join=False)
fslr32k = fetch_template_surface("fslr32k", join=True)

# Load fsLR-32k middle wall mask
fslr32k_mask = fetch_mask("fslr32k")


# ### Sort the matrices into ipsilateral/contralateral relative to seizure focus

# In[3]:


# fsLR-32k length
n_64k = pet_raw.shape[1]
n_32k = int(n_64k/2)

# Flip R >> L 18F-mk6240
mk_ipsi, mk_contra = flip_mat(pet_raw[:,0:n_32k], pet_raw[:,n_32k:n_64k], df['lateralization'].values, flip='R')

# Flip R >> L Thickness
th_ipsi, th_contra = flip_mat(cth_raw[:,0:n_32k], cth_raw[:,n_32k:n_64k], df['lateralization'].values, flip='R')

# Merge ipsi and contra
tauMK6240 = np.concatenate((mk_ipsi, mk_contra), axis=1)
thickness = np.concatenate((th_ipsi, th_contra), axis=1)


# ## Participants
# 

# In[4]:


# Create a cross-tabulation of 'group' and 'mk6240.session'
cross_tab = pd.crosstab(df['group'], df['mk6240.session'], margins=False)

# Print the result in a nice pandas format
display(cross_tab)


# ------------------------
# ## **Figure 1:** Mapping of tau uptake in TLE
# ### A | Mean Tau 18F-MK6240 uptake by group

# In[5]:


# plot data
plot_ctx_groups(fslr32k_lh, fslr32k_rh, tauMK6240, df , color_range=(0.5, 2), Save=False, Col='inferno', mask=fslr32k_mask, scale=2)


# ### B.1 | Tau 18F-MK6240 uptake group difference

# In[6]:


# CREATE the mixed effects model p> 0.01 (p>0.005 per tail) 
slm = mem_groups(Data=tauMK6240, df=df, Surf=fslr32k, mask=fslr32k_mask, Cthr=0.01, mem=True,  Pcorr="rft")

# Plot t-values
plot_ctx_slm(fslr32k_lh, fslr32k_rh, slm, color_range=(-3,3), Str='t-values', Save=False, Col="RdBu_r", mask=fslr32k_mask, scale=1 )


# In[7]:


# p>0.025 Plot P-values
plot_ctx_pval(fslr32k_lh, fslr32k_rh, slm, Str='p-values', Save=False, Col="inferno_r", Thr=0.01, scale=1)


# ### B.2 | Mean SUVR values for significant regions in each hemisphere by group

# In[8]:


# significant values
thr=0.01
pvalues = np.copy(slm.P["pval"]["C"])

# Binarize the significant values
pvalues_bin = np.where(pvalues > thr, 0, 1)

# Split the binarized pvalues in left and right hemispheres
pvalues_bin_ipsi = pvalues_bin[0:n_32k]
pvalues_bin_contra = pvalues_bin[n_32k:n_64k]

# Calculate cmk.cntr for thr_l == 1 and assign it to cmk_dfm
df['mk6240.sig.ipsi'] = np.mean(mk_ipsi[:, pvalues_bin_ipsi == 1], axis=1)

# Calculate cmk.cntr for thr_r == 1 and assign it to cmk_dfm
df['mk6240.sig.contra'] = np.mean(mk_contra[:, pvalues_bin_contra == 1], axis=1)

# Save the significant values on a csv
#df.to_csv(f'{osf_path}/18F-MK6240_database.csv', sep=',', encoding='utf-8', index=False, header=True)


# In[9]:


# Define the features and titles
features = ['mk6240.sig.ipsi', 'mk6240.sig.contra']
titles = ['Ipsilateral', 'Contralateral']

# Create a figure with two subplots
plt.figure(figsize=(8, 4))  # Adjust the width to fit two plots

# Loop over the features and titles to create the plots
for i, (feature, title) in enumerate(zip(features, titles)):
    plt.subplot(1, 2, i + 1)  # 1 row, 2 columns, i+1 to select the subplot
    
    # Create violin plot with colored outlines and semi-transparent fill
    sns.violinplot(
        x="group", y=feature, data=df, order=["Patient", "Healthy"], linewidth=0.75, dodge=False,
        hue="group", fill=True, palette={"Patient": "#ff5555", "Healthy": "#666666"}, linecolor="white"
    )

    # Add jittered points (stripplot) with matching colors
    sns.stripplot(
        x="group", y=feature, data=df, jitter=True, order=["Patient", "Healthy"],
        palette={"Patient": "#ff5555", "Healthy": "#666666"},
        alpha=0.7, dodge=False, edgecolor="white", linewidth=1
    )

    # Set title and labels
    plt.title(title, fontsize=14)
    plt.xlabel("Group", fontsize=12)
    plt.ylabel("Mean SUVR", fontsize=12)

    # Set a lighter gray background (Gray 25: #dcdcdc)
    plt.gca().set_facecolor("#E5E5E5")

    # Perform t-test
    patients = df[df["group"] == "Patient"][feature]
    healthy = df[df["group"] == "Healthy"][feature]
    t_stat, p_value = ttest_ind(patients, healthy, equal_var=False)

    # Print t-test results
    print(f"{title} t-value: {t_stat:.3f}")
    print(f"{title} p-value: {p_value:.5f}")

# Show the plot
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


# ### C.1 | Subject-Wise Distribution of Cortical MK6240 Uptake
# ### C.1.a | Ridgeplot of TLE subjects

# In[10]:


# Slice for the 'Patient' group
tauMK6240_tle = tauMK6240[df['group'] == 'Patient', :]

# Remove the medial wall from the data
tauMK6240_tle_masked = tauMK6240_tle[:, fslr32k_mask]

# Now plot using the filtered data
plot_ridgeplot(tauMK6240_tle_masked,
               title="F18-MK6240 distribution in TLE", 
               Range=(0.4, 2.2), Cmap='inferno', Vline=1.5)


# ### C.1.b | Ridgeplot of Healthy controls

# In[11]:


# Filter mk_matched for the 'Healthy' group
tauMK6240_control = tauMK6240[df['group'] == 'Healthy', :]

# Apply the thr_sig mask to select the appropriate columns
tauMK6240_control_masked = tauMK6240_control[:, fslr32k_mask]

# Now plot using the filtered data
plot_ridgeplot(tauMK6240_control_masked,
               title="MK6240 distribution in HC", 
               Range=(0.4, 2.2), Cmap='inferno', Vline=1.5)


# ### C.2   | Individual examples of of Cortical [$^{18}F$]-MK6240 Uptake
# 
# ### C.2.a | TLE subjects

# In[12]:


# Low 18F-mk6240
# AGE: 33yrs, DURATION 15yrs, ONSET: 18yrs | px09_01
# AGE: 52yrs, DURATION  2yrs, ONSET: 50yrs | px04_02

# High 18F-mk6240
# AGE: 45yrs, DURATION 38yrs, ONSET: 7yrs  | px10_01
# AGE: 25yrs, DURATION  8yrs, ONSET: 17yrs | px11_02
individual_tle = [
    tauMK6240[df.index[df['id'] == 'px09_01'][0],:],
    tauMK6240[df.index[df['id'] == 'px04_02'][0],:],
    tauMK6240[df.index[df['id'] == 'px10_01'][0],:],
    tauMK6240[df.index[df['id'] == 'px11_02'][0],:] ]

# Plot subjects
plot_hemispheres(fslr32k_lh, fslr32k_rh, array_name=individual_tle, size=(900, 1000),
                 color_bar='bottom', zoom=1.25, embed_nb=True,
                 interactive=False, share='both',
                 nan_color=(0, 0, 0, 1), color_range=(0.5, 2),
                 cmap='inferno', transparent_bg=True, 
                 label_text=['TLE-low1', 'TLE-low2','TLE-high1', 'TLE-high2'],
                 screenshot=False, scale=1)


# ### C.2.b | Healthy controls

# In[13]:


# Subset of subjects HC
individual_hc = [
    tauMK6240[df.index[df['id'] == 'hc10_01'][0],:],
    tauMK6240[df.index[df['id'] == 'hc01_02'][0],:],
    tauMK6240[df.index[df['id'] == 'hc02_01'][0],:],
    tauMK6240[df.index[df['id'] == 'hc11_01'][0],:]
          ]

# Plot subjects
plot_hemispheres(fslr32k_lh, fslr32k_rh, array_name=individual_hc, size=(900, 1000),
                 color_bar='bottom', zoom=1.25, embed_nb=True,
                 interactive=False, share='both',
                 nan_color=(0, 0, 0, 1), color_range=(0.5, 2.2),
                 cmap='inferno', transparent_bg=True, 
                 label_text=['HC-low1', 'HC-low2', 'HC-high1', 'HC-high2'],
                 screenshot=False, scale=1)


# ----------------
# ## **Figure 2:** Network contextualization

# In[14]:


# Load Functional connectome (FC)
fc_5k_mean = np.load(f'{osf_path}/surf-fsLR-5k_desc-GroupMean_FC.npy')

# Mean columns FC
fc_5k_Colmean = np.mean(fc_5k_mean, axis=0)

# Load weighted Structural connectome (SC)
scw_5k_mean = np.load(f'{osf_path}/surf-fsLR-5k_desc-GroupMean_SCw.npy')

# Mean columns SC
scw_5k_Colmean = np.mean(scw_5k_mean, axis=0)

# Load fsLR-5k
fslr5k_lh = fetch_surface('fsLR-5k.L.inflated.surf.gii')
fslr5k_rh = fetch_surface('fsLR-5k.R.inflated.surf.gii')

# Load fsLR-5k mask
fslr5k_mask_lh = fetch_surface('fsLR-5k.L.mask.shape.gii', is_surf=False)
fslr5k_mask_rh =  fetch_surface('fsLR-5k.R.mask.shape.gii', is_surf=False)
fslr5k_mask_bin = np.concatenate((fslr5k_mask_lh, fslr5k_mask_rh), axis=0)

# Labels and boolean mask
fslr5k_mask = fslr5k_mask_bin != 0

# Create a tval map variable
mk6240_tval_map = slm.t[0]

# Remove NaN from tmap
mk6240_tval_map[np.isnan(mk6240_tval_map)] = 0

# Resample mk6240 group difference T-val map from fslr32k to fslr5k
mk6240_tval_map_5k = metric_resample(mk6240_tval_map)


# ### Tau MK6240 Group Differences (tvalues)

# In[15]:


plot_brain_network(fslr5k_lh, fslr5k_rh, fslr5k_mask, fc_5k_mean, node_val=mk6240_tval_map_5k, 
                   node_colmap='RdBu_r', edge_alpha=0, sparcity=0.975, node_colrange=[-4, 4], dpi=100)


# ### A | Relationship Between Tau MK6240 Group Differences and *Mean Group Connectome*

# In[16]:


plot_network_correlation(Xval = [fc_5k_Colmean[fslr5k_mask], scw_5k_Colmean[fslr5k_mask]],
    Yval = mk6240_tval_map_5k[fslr5k_mask],
    Xnames = ['Functional Connectome', 'Structural Connectome'])


# ### Spin permutation to assess significance
# 
# > Alexander-Bloch, A. F., Shou, H., Liu, S., Satterthwaite, T. D., Glahn, D. C., Shinohara, R. T., ... & Raznahan, A. (2018). On testing for spatial correspondence between maps of human brain structure and function. Neuroimage, 178, 540-551.
# 
# > Vos de Wael, R., Benkarim, O., Paquola, C., Lariviere, S., Royer, J., Tavakol, S., ... & Bernhardt, B. C. (2020). BrainSpace: a toolbox for the analysis of macroscale gradients in neuroimaging and connectomics datasets. Commun Biol 3, 103.

# In[17]:


get_ipython().run_cell_magic('time', '', "from brainspace.null_models import SpinPermutations\n\n# Load spheres fsLR5k\nsphere_lh = fetch_surface('fsLR-5k.L.sphere.surf.gii')\nsphere_rh = fetch_surface('fsLR-5k.R.sphere.surf.gii')\n\n# Set the number of rotations\nn_rand = 10000\n\nsp = SpinPermutations(n_rep=n_rand, random_state=0)\nsp.fit(sphere_lh, points_rh=sphere_rh)")


# In[18]:


get_ipython().run_cell_magic('time', '', "plot_network_spintest([fc_5k_Colmean, scw_5k_Colmean], \n                      mk6240_tval_map_5k, sp, fslr5k_mask, Xnames=['mean_fc', 'mean_sc'])")


# ### Mean functional connectome as adjacency matrix

# In[19]:


# Slice the FC of the patients to remove the midwall
fc_5k_NO_midwall = fc_5k_mean[fslr5k_mask, :]
fc_5k_NO_midwall = fc_5k_NO_midwall[:, fslr5k_mask]

# Threshold based on the edges
A_edges = fc_5k_NO_midwall[np.tril_indices_from(fc_5k_NO_midwall, k=-1)]
A_qt = np.quantile(A_edges, [0.01, 0.99])

# Plot the group FC mean
plot_connectome(fc_5k_NO_midwall, col='bone_r', xlab='', ylab='', Title='Functional Connectome',
                vmin=A_qt[0], vmax=A_qt[1], figsize=(6,5))


# ### Mean functional connectome as network

# In[20]:


plot_brain_network(fslr5k_lh, fslr5k_rh, fslr5k_mask, fc_5k_mean, node_val=fc_5k_Colmean, 
                   node_colmap='bone_r', edge_alpha=0.4, sparcity=0.975, dpi=100 )


# ### Mean structural connectome as adjacency matrix

# In[21]:


# Slice the SC of the patients to remove the midwall
scw_5k_NO_midwall = scw_5k_mean[fslr5k_mask, :]
scw_5k_NO_midwall = scw_5k_NO_midwall[:, fslr5k_mask]

# log transfor for visualization
scw_5k_NO_midwall_log = np.log(scw_5k_NO_midwall+1)

# Threshold based on the edges
A_edges = scw_5k_NO_midwall_log[np.tril_indices_from(scw_5k_NO_midwall_log, k=-1)]
A_qt = np.quantile(A_edges, [0.01, 0.99])

# Plot the group SCw mean
plot_connectome(scw_5k_NO_midwall_log, col='bone_r', xlab='', ylab='', Title='Structural Connectome',
                vmin=A_qt[0], vmax=A_qt[1], figsize=(6,5))


# ### Mean structural connectome as network

# In[22]:


plot_brain_network(fslr5k_lh, fslr5k_rh, fslr5k_mask, scw_5k_mean, node_val=scw_5k_Colmean, 
                   node_colmap='bone_r', edge_alpha=0.3, sparcity=0.995, dpi=100)


# ### B | Relationship Between Tau MK6240 Group Differences and *Connectome weighted by the neighbours*
# 
# Connectome weighted by neighbours:
# 
# $$D_i=\frac{1}{N_i} \sum^{N_i}_{j\ne i,j=1} d_j \times FC_{ij}$$
# 
# > Shafiei, Golia, et al. "Spatial patterning of tissue volume loss in schizophrenia reflects brain network architecture." Biological psychiatry 87.8 (2020): 727-735.

# In[23]:


# Calculate the neighborhood_estimates
_, neig_fc = neighborhood_estimates(mk6240_tval_map_5k[fslr5k_mask], fc_5k_NO_midwall, method='spearman')

_,neig_sc = neighborhood_estimates(mk6240_tval_map_5k[fslr5k_mask], scw_5k_NO_midwall, method='spearman')


# In[24]:


plot_network_correlation(Xval = [neig_fc, neig_sc],
    Yval = mk6240_tval_map_5k[fslr5k_mask],
    Xnames = ['neighbors_FC', 'neighbors_SC'])


# ### Spin permutation to assess significance

# In[25]:


get_ipython().run_cell_magic('time', '', "plot_network_spintest([fc_5k_Colmean, scw_5k_Colmean], \n                      mk6240_tval_map_5k, sp, fslr5k_mask, Xnames=['neighbors_FC', 'neighbors_SC'])")


# ### Tau MK6240 weighted by the *Functional Connectome* neighbours

# In[26]:


neig_fc_5k = map_to_labels5k(neig_fc, fslr5k_mask)

plot_brain_network(fslr5k_lh, fslr5k_rh, fslr5k_mask, fc_5k_mean, node_val=neig_fc_5k, 
                   node_colmap='cmo.amp', edge_alpha=0.4, sparcity=0.975, dpi=100)


# ### Tau MK6240 weighted by the *Structural connectome* neighbours

# In[27]:


neig_sc_5k = map_to_labels5k(neig_sc, fslr5k_mask)

plot_brain_network(fslr5k_lh, fslr5k_rh, fslr5k_mask, scw_5k_mean, node_val=neig_sc_5k, 
                   node_colmap='cmo.amp', edge_alpha=0.3, sparcity=0.995, dpi=100)


# -----------------
# ## **Figure 3:** Relationship to clinical and cognitive variables
# 
# ### A | Tau MK6240 SUVR and clinical relationships

# ### Effect of age

# In[28]:


slm_surf(df, tauMK6240, feat='age.mk6240', neg_tail=False, cthr=0.05, scale=1)


# ### Effect of duration

# In[29]:


# Slice only Patients
idx = df['group'] == 'Patient'

slm_surf(df[idx], tauMK6240[idx,:], feat='duration', neg_tail=False, cthr=0.05, scale=1)


# ### B | Tau MK6240 SUVR and behavioral relationships

# ### Epitrack score

# In[30]:


slm_surf(df, feat='EpiTrack.Total.Score', Y=tauMK6240, neg_tail=True, cthr=0.05, scale=1)


# ### Episodic memory

# In[31]:


slm_surf(df, feat='Epi.Acc.D.per', Y=tauMK6240, neg_tail=True, cthr=0.05, scale=1)


# ### Semantic memory

# In[32]:


slm_surf(df, feat='task.semantic', Y=tauMK6240, neg_tail=True, cthr=0.05, alpha=0.5, scale=1)


# ---------------------
# # Supplementary figures

# ## Cortical map of abnormal MK6240 SUVR probability

# In[33]:


# Create a new matrix with values higher than 1.5 set to 1, rest set to 0
tauMK6240_bin = np.where(tauMK6240 > 1.5, 1, 0)

# plot data
plot_ctx_groups(fslr32k_lh, fslr32k_rh, tauMK6240_bin, df , color_range=(-0.75,0.75), 
                Save=False, Col='RdBu_r', mask=fslr32k_mask, scale=1)


# ## Group difference in cortical thickness

# In[34]:


# p>0.025 CREATE the mixed effects model
slm_th = mem_groups(Data=thickness, df=df, Surf=fslr32k, mask=fslr32k_mask, Cthr=0.01, mem=True,  Pcorr="rft")

# Plot t-values
plot_ctx_slm(fslr32k_lh, fslr32k_rh, slm_th, color_range=(-6,6), Str='t-values', Col="RdBu_r", 
             mask=fslr32k_mask, scale=1 )


# In[35]:


# p>0.025 Plot P-values
plot_ctx_pval(fslr32k_lh, fslr32k_rh, slm_th, Str='p-values', Col="inferno_r", Thr=0.01, scale=1)


# ## Group difference in MK6240 controlled by thickness

# In[36]:


# -------------------------------
# Cortical 18F-MK6240~Thickness
tauMK6240_res = controlVertex(X=thickness, Y=tauMK6240)

# p>0.025 CREATE the mixed effects model
slm_mkth = mem_groups(Data=tauMK6240_res, df=df, Surf=fslr32k, mask=fslr32k_mask, Cthr=0.01, mem=True,  Pcorr="rft")

# Plot t-values
plot_ctx_slm(fslr32k_lh, fslr32k_rh, slm_mkth, color_range=(-3,3), Str='t-values', Col="RdBu_r", 
             mask=fslr32k_mask, scale=1)


# In[37]:


# p>0.025 Plot P-values
plot_ctx_pval(fslr32k_lh, fslr32k_rh, slm_mkth, Str='p-values', Col="inferno_r", Thr=0.01, scale=1)


# ## Mean longitudinal MK6240 by subject per group
# 
# See `Rmarkdown` document
