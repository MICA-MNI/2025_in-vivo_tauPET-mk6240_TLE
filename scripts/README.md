# Scripts to reproduce *In-vivo findings of increased tau uptake in TLE*

## Table of Contents

1. [Directory file content](#directory-file-content)  
2. [Tau PET-mk6240 features](#tau-pet-mk6240-features)  
   - [Parameters](#parameters)  
   - [Unit reference](#unit-reference)  
   - [3D non motion corrected data](#3d-non-motion-corrected-data)  
   - [4D motion corrected data](#4d-motion-corrected-data)  
5. [Data processing](#data-processing)  
   - [Processing function](#processing-function)  
   - [Standardized uptake values](#standardized-uptake-values)  
6. [Data analysis](#data-analysis)  

 ## Directory file content
 
|  File         |  Description             |
|---------------|--------------------------|
| [`.sh`]       | PET-preprocessing script |
| [`.ipynb`]    | notebook                 |

# Tau PET-mk6240 features
The PET images are transformed to `NIFTI` from the `ECAT (.v)` files with `v1.0.20240202 GCC11.2.0`. This transformation keeps the data in nano Curies.

### Parameters
**Dimensions**: 256 x 256 x 207 
**Voxel size**: 1.21875 x 1.21875 x 1.21875  
**psfmm** : scanner PSF FWHM in mm
> psf FWHM is the full-width/half-max of the the point-spread function (PSF) of the scanner as measured in image space (also known as the burring function). The blurring function depends on the scanner and reconstruction method

### Unit reference  
> cc=cm3=mL
> 1 nCi = 37 Bq  
> Bq: Becquerels  
> nCi: nano Curies  

### 3D non motion corrected data
1. `TX256.v` Linear atenuation map, 4D_MC is corregistered to this image. Later this is the image that is use to calculate the affine registration between PET and MRi space.
   > This is a CT transmission scan
1. `EM_3D.v` 4 frames, 20 minutes of acquisition each one (Bq/cc).  
1. `EM_3D_AVG.v` average of the four frames (Bq/cc).  

### 4D motion corrected data  
1. `EM_4D_MC01.v` Filter Back Projection (FBP) image. It is the inversion of the radon transformation. 4 frames, 20 minutes of acquisition each one (nCi/cc).  
1. `EM_4D_MC01_AVG.v` FBP average of 4 four frames (nCi/cc units).  

# Data processing  
The functions to process the PET data can be found under the [mica-pet GitHub repository](https://github.com/rcruces/mica-pet). The processing workflow consists in two steps, a transformation from `ECAT` to `NIFTI`, and a set of registrations to the *T1nativepro* space and the to the surface.  

![alt text](./data/fig_methods.png)

1. `dcm2niix`
1. `fslmaths -Tmean` compute average of EM_4D_MC01.v  
1. PET to T1 registration  
 > - PET to native MRI space (`ANTs`)
 > - PET to Freesurface (surface) space (`bbreg`)

## Processing function
> `micapet_01.sh`

### TO DO
> ADD new PVC methos
> keep json from t1w image
> rename outputs and conform to PET BIDS

### Normalize PET and projects it to the surface
> `micapet_02.sh`

### PET data to hipUnfold surfaces
> `micapet_02.sh -hipunfold`

# Standarized uptake values
In summary this gives the following equation to calculate SUV at time t post injection:
>>![SUV](https://latex.codecogs.com/svg.image?{\color{White}&space;SUV(t)&space;=&space;\frac&space;{c_{img}(t)}&space;{ID&space;/&space;BW}})  
>**Cimg**=PET image  
>**ID**=Injected dose  
>**BW**=body weight  
>With the radioactivity measured from an image acquired at (or around) the time t, decay corrected to t=0 and expressed as volume concentration (e.g. MBq/mL), the injected dose ID at t=0 (e.g. in MBq), and the body weight BW (near the time of image acquisition) implicitly converted into the body volume assuming an average mass density of 1 g/mL.  

> SUVR (ratio): the injected activity, the body weight and the mass density that are all part of the SUV calculation, cancel:
>> ![SUVR](https://latex.codecogs.com/svg.image?{\color{White}&space;{\mathit{SUVR}}&space;=&space;\frac&space;{\mathit{SUV_{target}}}&space;{\mathit{SUV_{reference}}}&space;=&space;\frac&space;{\mathit{c_{img,target}}}&space;{\mathit{c_{img,reference}}}})

# Data analysis