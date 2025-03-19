# utils.py - Python Utilities

# Libraries
import glob
import nibabel as nib
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from brainspace.plotting import plot_hemispheres
from brainstat.stats.terms import MixedEffect, FixedEffect
from brainstat.stats.SLM import SLM
import numpy as np
import seaborn as sns
import cmocean
cmaps = cmocean.cm.cmap_d

# -----------------------------------------------------------------------------
# FLIP data Right to Left
def flip_mat(mat_l, mat_r, lat_vec, flip='R'):
    '''
    Flips Right to Left data matrices to generate
    new matrices IPSILATERAL and CONTRALATERAL
    relative to the seizure onset

    Parameters
    ----------
    mat_l : numpy array
        Matrix with the values associated to the LEFT structures [subject x feature].
    mat_r     : numpy array
        Matrix with the values associated to the RIGHT structures [subject x feature].
    lat_vec   : numpy array (strings)
        Array with the laterality coded as R and L (rigth and left respectively).
    flip      : string
    Returns
    -------
    ipsi, ctra
    '''
    print("Flipping data from Right to Left")

    # Create an  array for the results
    ipsi = np.ndarray.copy(mat_l)
    ctra =  np.ndarray.copy(mat_r)

    for i, Lat in enumerate(lat_vec):
        if Lat == flip:
            ipsi[i,:] = mat_r[i,:]
            ctra[i,:] = mat_l[i,:]

    return ipsi, ctra

# -----------------------------------------------------------------------------
# Match data
def match_data(data_array, df_dat, col='preproc'):
    """
    Matches data based on a specified column in a dataframe.

    Parameters:
    - data_array (numpy.ndarray): Left dataset.
    - df_dat (pandas.DataFrame): Dataframe containing the data to be matched.
    - col (str, optional): Column name to use for matching. Default is 'preproc'.

    Returns:
    - data_array (numpy.ndarray): Left dataset after matching.
    - df (pandas.DataFrame): Dataframe after matching and dropping unmatched rows.

    """
    
    # Get the indices of rows with NaN values in the specified column
    nan_indices = df_dat.index[df_dat[col].isna()].values
    
    # Remove rows with NaN values in the specified column
    df_dat = df_dat.dropna(subset=[col])

    # Drop corresponding entries from the data array
    data_array = np.delete(data_array, nan_indices, axis=0)

    # Get the indices of unmatched rows
    indx = df_dat[df_dat[col] == 0].index.values[:]

    # Check if any subjects are unmatched
    N = np.isin(df_dat.index.values, indx)

    # Drop unmatched rows from the dataframe
    df = df_dat.drop(index=indx, axis=1)

    # Drop unmatched subjects from the data array
    data_array = data_array[np.where(N != True)]

    return data_array, df

# -----------------------------------------------------------------------------
# Plot the surface of patients and controls
def plot_ctx_groups(surf_lh, surf_rh, mtx, df, color_range=(-4,4), Save=False, Col="cmo.dense", png_file="", group=['Healthy', 'Patient'],
                   nan_color=(0, 0, 0, 1), mask=None, scale=2):
    """
    Plot the surface of patients and controls based on the given matrix and dataframe.
    Parameters:
    - mtx (numpy.ndarray): The matrix containing the data.
    - df (pandas.DataFrame): The dataframe containing the group information.
    - color_range (tuple, optional): The range of values for the color bar. Default is (-4, 4).
    - Str (str, optional): The label text for the groups. Default is 'feature'.
    - Save (bool, optional): Whether to save the plot as an image. Default is False.
    - Col (str, optional): The color map to use for the plot. Default is 'cmo.dense'.
    - png_file (str, optional): The file name for the saved image. Default is an empty string.

    Returns:
    - f: The plot object.

    """
    grad = [None] * len(group)

    # Add Groups to list of arrays
    for i, n in enumerate(group):
        grad[i] = np.mean(mtx[df.group==group[i],:], axis=0)
        if mask is not None:
            # Remove the midwall vertices
            grad[i][mask == False] = np.nan

    # Plot the surface PATIENTS and CONTROLS
    f = plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(900, 500),
                         color_bar='bottom', zoom=1.25, embed_nb=True,
                         interactive=False, share='both',
                         nan_color=nan_color, color_range=color_range,
                         cmap=Col, transparent_bg=True, label_text=group,
                         screenshot=Save, filename=png_file, scale=scale)
    return(f)

# -----------------------------------------------------------------------------
# Create a mixed effects model
def mem_groups(Data, df, Surf, Pcorr=["fdr", "rft"], Cthr=0.025, mask=None, mem=True, group=['Healthy', 'Patient']):
    '''
    Generate a mixed effects model of a Surface
    Contras: [TLE - Controls]

    Parameters
    ----------
    regex : str
        Regular expression with the path(s) to the data to be uploaded.

    Returns
    -------
    Array, pandas.Dataframe
     '''
    # -----------------------------------------------------------------------------
    # terms
    term_grp = FixedEffect(df['group'])
    term_age = FixedEffect(df['age.mk6240'])
    term_sex = FixedEffect(df['sex'])
    term_subject = MixedEffect(df['participant_id'])

    # contrast  (Patient - Control)
    # 1: control, 2: patient
    contrast_grp = (df.group == group[1]).astype(int) - (df.group == group[0]).astype(int)

    # Model is is mixed Subject is set as random MIxed effect variable
    if mem == True:
        print("Y ~ group + age + sex + 1/subject")
        model = term_grp + term_sex + term_age  + term_subject
    else:
        model = term_grp + term_sex + term_age
        print("Y ~ group + age + sex")

    # fitting the model
    slm_mixed = SLM(
        model,
        contrast_grp,
        mask=mask,
        surf=Surf,
        correction=Pcorr,
        two_tailed=True,
        cluster_threshold=Cthr
    )
    slm_mixed.fit(Data)

    return slm_mixed

# -----------------------------------------------------------------------------
# Plot the results of the cortical SLM T-values
def plot_ctx_slm(surf_lh, surf_rh, slm, color_range=(-4,4), Str='slm', Save=False, Col="bwr", png_file='', mask=None, scale=2):
    '''
    Plots the results of cortical SLM
    MEM Y ~ Age + Sex + (1 | subject )
    '''

    # Mask od the midwall vertices
    if mask is not None:
        surf_data = slm.t[0]*mask
    else:
        surf_data = slm.t[0]

    # Plot t-values
    f = plot_hemispheres(surf_lh, surf_rh, array_name=surf_data, size=(900, 250), color_bar='bottom', zoom=1.25, embed_nb=True, interactive=False, share='both',
                             nan_color=(0, 0, 0, 1), cmap=Col, transparent_bg=True, label_text=[Str], color_range=color_range,
                             screenshot=Save, filename=png_file, scale=scale)
    return(f)

# -----------------------------------------------------------------------------
# Plot the results of the cortical SLM P-values
def plot_ctx_pval(surf_lh, surf_rh, slm, Str='slm', Save=False, Col="inferno", Thr=0.05, png_file='', scale=2):
    '''
    Plots the results of cortical SLM
    MEM Y ~ Age + Sex + (1 | subject )
    '''
    # Plot cluster p-values
    sig_pvalues = np.copy(slm.P["pval"]["C"])

    # Apply thresholding and create a new array with 0s and 1s
    plot_pvalues = np.where(sig_pvalues > Thr, 0, 1)

    f = plot_hemispheres(surf_lh, surf_rh, array_name=plot_pvalues, size=(900, 250), color_bar='bottom', zoom=1.25, embed_nb=True, interactive=False, share='both',
                             nan_color=(0, 0, 0, 1), cmap=Col, transparent_bg=True, label_text=[Str], color_range=(0, Thr),
                             screenshot=Save, filename=png_file, scale=scale)
    return(f)

# -----------------------------------------------------------------------------
# Control Vertex data by thickness
def controlVertex(X,Y):
    '''
    controlVertex(X,Y)
    generates column wide residuals.
    Input matrices should be in [ N x vertices ]
    Same results as using R function: residuals(lm(y~x))

    Parameters
    ----------
    X : numpy.array
        Data matrix .
    Y : numpy.array
        Data matrix to be controlled by X (predict).

    Returns
    -------
    Array: Y_corr

    Usage
    -----
    Cortical MK6240~Thickness
    cmkth = controlVertex(X=cth, Y=cmk)
     '''

    if Y.shape != X.shape:
        print("X and Y matrices MUST have the same shape")
        raise ValueError("X and Y matrices MUST have the same shape")
    else:
        # Create an empty array for the results
        Y_corr=np.empty([Y.shape[0], Y.shape[1]])

        for i in range(0, Y.shape[1]):
            x = X[:,i].reshape(-1,1)
            y = Y[:,i].reshape(-1,1)

            # Create linear regression object
            mod = LinearRegression()
            # Fit the data to the model
            slm = mod.fit(x, y)
            # Generate the predicte values
            predic = slm.predict(x)
            # predic = (np.dot(x, slm.coef_) + slm.intercept_) == slm.predict(x)

            # Residual
            residual = (y - predic)

            # Save the corrected Y matrix
            Y_corr[:,i] = residual[:,0]

        return Y_corr

# -----------------------------------------------------------------------------
def plot_connectome(mtx, Title='matrix plot', xlab='X', ylab='Y', col='rocket', vmin=None, vmax=None,
                   xticklabels='auto', yticklabels='auto',xrot=90, yrot=0, save_path=None):

    '''
    This optional function, only plots a connectome as a heatmap
    Parameters
    ----------
    mtx : np.array
    Returns
    -------
    '''
    f, ax = plt.subplots(figsize=(15,10))
    g = sns.heatmap(mtx, ax=ax, cmap=col, vmin=vmin, vmax=vmax, xticklabels=xticklabels, yticklabels=yticklabels)
    g.set_xlabel(xlab)
    g.set_ylabel(ylab)
    g.set_title(Title)
    # Rotate the x-axis labels
    # rotate tick labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xrot, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=yrot, ha='right')

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    pass

def plot_ridgeplot(matrix, matrix_df=None, Cmap='rocket', Range=(0.5, 2), Xlab="SUVR value", save_path=None, 
                  title=None, Vline=None, VlineCol='red'):
    """
    Parameters:
    - matrix: numpy array
        The matrix to be plotted.
    - matrix_df: pandas DataFrame, optional
        The DataFrame containing additional information about the matrix.
    - Cmap: str, optional
        The colormap to be used for the ridgeplot. Default is 'rocket'.
    - Range: tuple, optional
        The range of values for the x-axis. Default is (0.5, 2).
    - Xlab: str, optional
        The label for the x-axis. Default is "SUVR value".
    - save_path: str, optional
        The file path to save the plot. If not provided, the plot will be displayed.
    - title: str, optional
        The title of the plot.
    - Vline: float, optional
        Whether to plot the mean line for all distributions. Default is False.
    Returns:
    None

    Plot a ridgeplot of the given matrix.
    
    """
    if matrix_df is None:
        # Create a default DataFrame with placeholder values
        matrix_df = pd.DataFrame({'id': [f'{i+1}' for i in range(matrix.shape[0])]})
        print_labels = False
    else:
        print_labels = True

    mean_row_values = np.mean(matrix, axis=1)
    sorted_indices = np.argsort(mean_row_values)
    sorted_matrix = matrix[sorted_indices]
    sorted_id_x = matrix_df['id'].values[sorted_indices]

    ai = sorted_matrix.flatten()
    subject = np.array([])
    id_x = np.array([])

    for i in range(sorted_matrix.shape[0]):
        label = np.array([str(i+1) for j in range(sorted_matrix.shape[1])])
        subject = np.concatenate((subject, label))
        id_label = np.array([sorted_id_x[i] for j in range(sorted_matrix.shape[1])])
        id_x = np.concatenate((id_x, id_label))

    d = {'feature': ai,
         'subject': subject,
         'id_x': id_x
        }
    df = pd.DataFrame(d)

    f, axs = plt.subplots(nrows=sorted_matrix.shape[0], figsize=(3.468504*2.5, 2.220472*3.5), sharex=True, sharey=True)
    f.set_facecolor('none')

    x = np.linspace(Range[0], Range[1], 100)

    for i, ax in enumerate(axs, 1):
        sns.kdeplot(df[df["subject"]==str(i)]['feature'],
                    fill=True,
                    color="w",
                    alpha=0.25,
                    linewidth=1.5,
                    legend=False,
                    ax=ax)
        
        ax.set_xlim(Range[0], Range[1])
        
        im = ax.imshow(np.vstack([x, x]),
                       cmap=Cmap,
                       aspect="auto",
                       extent=[*ax.get_xlim(), *ax.get_ylim()]
                      )
        ax.collections
        path = ax.collections[0].get_paths()[0]
        patch = mpl.patches.PathPatch(path, transform=ax.transData)
        im.set_clip_path(patch)
           
        ax.spines[['left','right','bottom','top']].set_visible(False)
        
        if i != sorted_matrix.shape[0]:
            ax.tick_params(axis="x", length=0)
        else:
            ax.set_xlabel(Xlab)
            
        ax.set_yticks([])
        ax.set_ylabel("")
        
        ax.axhline(0, color="black")

        ax.set_facecolor("none")

    for i, ax in enumerate(axs):
        if i == sorted_matrix.shape[0] - 1:
            ax.set_xticks([Range[0], Range[1]])  # Set x-axis ticks for the bottom plot
        else:
            ax.set_xticks([])  # Remove x-axis ticks from other plots
        if print_labels:
            ax.text(0.05, 0.01, sorted_id_x[i], transform=ax.transAxes, fontsize=10, color='black', ha='left', va='bottom')

    # Calculate and add a single mean line for all subplots if mean_line is True
    if Vline is not None:
        # Check that Vline is numeric
        for ax in axs:
            ax.axvline(x=Vline, linestyle='dashed', color=VlineCol)

    plt.subplots_adjust(hspace=-0.8)
    
    if title:
        plt.suptitle(title, y=0.99, fontsize=16)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

def load_data(regex, surf='fslr32k'):
    '''
    load_hippdata(regex,)
    Loads a subject(s) using a path OR regular expression.

    Parameters
    ----------
    regex : str
        Regular expression with the path(s) to the data to be uploaded.
    surf  : str
        ['hippunfold', 'fslr32k', 'fs5']
    Returns
    -------
    tuple
        - numpy.ndarray: Array of loaded data
        - pandas.DataFrame: DataFrame of patient data
    '''
    # get the data
    files = sorted(glob.glob(regex))

    # Empty variable to load the data
    if surf == 'fslr32k':
        N = 32492
    elif surf == 'hippunfold':
        N = 7262
    else:
        raise ValueError("Unsupported surface type. Choose from ['hippunfold', 'fslr32k', 'fs5']")

    def load_fun(file):
        mat = nib.load(file).darrays[0].data
        return mat

    # load the data
    vx_val = np.empty([N,])  # Initialize with N rows
    for n, file in enumerate(files):
        vx_val = np.vstack((vx_val, load_fun(file)))

    vx_val = np.delete(vx_val, (0), axis=0)  # Remove the first empty row

    # Generate a dataframe of patients and controls
    data_tb = {
        'sub': [x.split('/')[-1].split('_')[0].split('sub-')[1] for x in files],
        'type': [x.split('/')[-1].split('_')[0].split('sub-')[1][0] for x in files],
        'ses': [x.split('/')[-1].split('_')[1].split('ses-')[1][1] for x in files],
        'id': [x.split('/')[-1].split('_')[0].split('sub-')[1] for x in files]
    }
    data_df = pd.DataFrame(data_tb)

    # Merge Subject ID and Session to obtain 'id'
    data_df['id'] = data_df[['sub', 'ses']].agg('_0'.join, axis=1)

    # Use the id as row Index
    data_df.index = data_df['id']

    return vx_val, data_df

# -----------------------------------------------------------------------------
# Function to generate the surface models of mk6240 vs Clinical and behavioral variables
def slm_surf(df, Y, feat='age.mk6240', neg_tail=False, cthr=0.05, alpha=0.3):
    """
    Run SLM analysis on the given feature with specified contrast direction and cluster threshold.
    
    Parameters:
    df (DataFrame): The input data containing the feature and participant IDs.
    feat (str): The feature column to analyze.
    Y (np.array): Surface data to fit the model.
    neg_tail (bool): If True, reverses the contrast.
    cthr (float): Cluster threshold for statistical correction.
    
    Returns:
    np.array: Processed surface data after applying statistical thresholds.
    """
    
    # Load fsLR-32k surface and mask
    from brainstat.datasets import fetch_mask, fetch_template_surface
    fslr32k = fetch_template_surface("fslr32k", join=True)
    surf_lh, surf_rh = fetch_template_surface("fslr32k", join=False)

    # fsLR-32k middle wall mask
    fslr32k_mask = fetch_mask("fslr32k")

    # Identify NaN indices in the feature column
    nan_idx = df[feat].isna()

    # Remove NaN rows from df and Y
    df_clean = df.loc[~nan_idx].copy()
    Y_clean = Y[~nan_idx]  # Assuming Y has the same length as df

    # Define fixed and mixed effects models
    term = FixedEffect(df_clean[feat])
    term_subject = MixedEffect(df_clean['participant_id'])
    
    # Set contrast direction and colormap based on neg_tail flag
    contrast = -df_clean[feat] if neg_tail else df_clean[feat]
    cmap = "RdBu" if neg_tail else "RdBu_r"
    
    # Define the full model
    model = term + term_subject
    
    # Run SLM analysis with the given parameters
    slm_feat = SLM(
        model,
        contrast,
        surf=fslr32k,
        mask=fslr32k_mask,
        correction="rft",
        cluster_threshold=cthr,
        two_tailed=False
    )
    slm_feat.fit(Y_clean)
    
    # Get the cluster p-values from the analysis
    sig_pvalues = np.copy(slm_feat.P["pval"]["C"])
    
    # Apply thresholding to create a binary mask of significant values
    surf_pvalues = np.where(sig_pvalues > cthr, alpha, 1)
    
    # Multiply the t-values by the mask and thresholded p-values
    surf_data = slm_feat.t[0] * fslr32k_mask * surf_pvalues
    
    # Ensure negative values are set to zero
    surf_data[surf_data < 0] = 0
    
    # Plot the results on brain hemispheres
    f = plot_hemispheres(
        surf_lh, surf_rh, array_name=surf_data, size=(900, 250), color_bar='bottom', 
        zoom=1.25, embed_nb=True, interactive=False, share='both', nan_color=(0, 0, 0, 1), 
        cmap=cmap, transparent_bg=True, label_text=[feat], color_range=(-3, 3), 
        screenshot=False, scale=2
    )

    return f
