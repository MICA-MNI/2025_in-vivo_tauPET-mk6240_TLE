# *In-vivo* evidence for increased tau deposition in temporal lobe epilepsy

### Table of Contents

1.  [Directory file content](#directory-file-content)
2.  [Tau PET 18F-mk6240 features](#tau-pet--18f-mk6240-features)
3.  [Data analysis](#data-analysis)

### Directory file content

| File                                                                                                                                                                      | Description                                                                   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| [`Fig-1_Tau-pet_18F-mk6240_SupFig.ipynb`](https://github.com/MICA-MNI/2025_in-vivo_tauPET-mk6240_TLE/blob/main/scripts/Fig-1_Tau-pet_18F-mk6240_SupFig.ipynb)             | PET-preprocessing script                                                      |
| [`Fig-2_network_contextualization.ipynb`](https://github.com/MICA-MNI/2025_in-vivo_tauPET-mk6240_TLE/blob/main/scripts/Fig-2_network_contextualization.ipynb)             | notebook                                                                      |
| [`Fig-3_clinical-cognitive_correlations.ipynb`](https://github.com/MICA-MNI/2025_in-vivo_tauPET-mk6240_TLE/blob/main/scripts/Fig-3_clinical-cognitive_correlations.ipynb) | notebook                                                                      |
| [`Increased_in-vivo_tau_in_TLE.py`](https://github.com/MICA-MNI/2025_in-vivo_tauPET-mk6240_TLE/blob/main/scripts/Increased_in-vivo_tau_in_TLE.py)                         | A Python script with all the                                                  |
| [`Increased_in-vivo_tau_in_TLE.Rmd`](https://github.com/MICA-MNI/2025_in-vivo_tauPET-mk6240_TLE/blob/main/scripts/Increased_in-vivo_tau_in_TLE.Rmd)                       | An R Markdown document includes statistical analysis and data visualization.  |
| [`utils.py`](https://github.com/MICA-MNI/2025_in-vivo_tauPET-mk6240_TLE/blob/main/scripts/utils.py)                                                                       | A Python utility script, containing all the helper functions for the analysis |

## Tau PET \| 18F-mk6240 features

The PET images are transformed to `NIFTI` from the `ECAT (.v)` files
with `v1.0.20240202 GCC11.2.0`. This transformation keeps the data in
nano Curies.

### Parameters

**Dimensions**: 256 x 256 x 207  
**Voxel size**: 1.21875 x 1.21875 x 1.21875  
**psfmm** : scanner PSF FWHM in mm

> psf FWHM is the full-width/half-max of the the point-spread function
> (PSF) of the scanner as measured in image space (also known as the
> burring function). The blurring function depends on the scanner and
> reconstruction method

### Unit reference

> cc=cm3=mL 1 nCi = 37 Bq  
> Bq: Becquerels  
> nCi: nano Curies

### 3D non motion corrected data

1.  `TX256.v` Linear atenuation map, 4D_MC is corregistered to this
    image. Later this is the image that is use to calculate the affine
    registration between PET and MRi space. \> This is a CT transmission
    scan
2.  `EM_3D.v` 4 frames, 20 minutes of acquisition each one (Bq/cc).
3.  `EM_3D_AVG.v` average of the four frames (Bq/cc).

### 4D motion corrected data

1.  `EM_4D_MC01.v` Filter Back Projection (FBP) image. It is the
    inversion of the radon transformation. 4 frames, 20 minutes of
    acquisition each one (nCi/cc).
2.  `EM_4D_MC01_AVG.v` FBP average of 4 four frames (nCi/cc units).

### Standardized uptake values

In summary this gives the following equation to calculate SUV at time
$t$ post injection:

> ![SUV](https://latex.codecogs.com/svg.image?%7B\color%7BWhite%7D&space;SUV(t)&space;=&space;\frac&space;%7Bc_%7Bimg%7D(t)%7D&space;%7BID&space;/&space;BW%7D%7D)  
> **Cimg**=PET image  
> **ID**=Injected dose  
> **BW**=body weight  
> With the radioactivity measured from an image acquired at (or around)
> the time t, decay corrected to t=0 and expressed as volume
> concentration (e.g. MBq/mL), the injected dose ID at t=0 (e.g. in
> MBq), and the body weight BW (near the time of image acquisition)
> implicitly converted into the body volume assuming an average mass
> density of 1 g/mL.

**SUVR (ratio)**: The injected activity, body weight, and mass density,
which are all components of the SUV calculation, cancel each other out.

> ![](https://latex.codecogs.com/svg.image?%7B\color%7BWhite%7D&space;%7B\mathit%7BSUVR%7D%7D&space;=&space;\frac&space;%7B\mathit%7BSUV_%7Btarget%7D%7D%7D&space;%7B\mathit%7BSUV_%7Breference%7D%7D%7D&space;=&space;\frac&space;%7B\mathit%7Bc_%7Bimg,target%7D%7D%7D&space;%7B\mathit%7Bc_%7Bimg,reference%7D%7D%7D%7D)

# Data analysis

## Database description

| Column name                    | Description                                                 |
|--------------------------------|-------------------------------------------------------------|
| `EpiTrack.Score.age.corrected` | Composite score of the EpiTrack battery.                    |
| `Epi.Acc.D.per`                | Accuracy in the delayed recall task.                        |
| `task.semantic`                | Accuracy in the semantic task.                              |
| `lateralization`               | Laterality of the seizure onset in TLE (R, L)               |
| `mCi.net`                      | Net injected dose of the radiotracer.                       |
| `mri.Tdiff`                    | Time difference between the MRI and the PET scan.           |
| `group`                        | Diagnostic group (Healthy, Patient).                        |
| `mk6240.session`               | Session of the PET scan (1,2).                              |
| `mk6240.sig.ipsi`              | Mean 18F-mk6240 SUVR in the ipsilateral significant areas   |
| `mk6240.sig.contra`            | Mean 18F-mk6240 SUVR in the contralateral significant areas |

### Setup the environment

### Get the data from the OSF repository

## Participants

<table style="NAborder-bottom: 0; color: black; " class="table">
<thead>
<tr>
<th style="text-align:left;">
Demographics
</th>
<th style="text-align:center;">
Healthy <br>N = 30
</th>
<th style="text-align:center;">
Patient <br>N = 28
</th>
<th style="text-align:center;">
Statistic
</th>
<th style="text-align:center;">
p-value
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
handedness
</td>
<td style="text-align:center;">
</td>
<td style="text-align:center;">
</td>
<td style="text-align:center;">
</td>
<td style="text-align:center;">
0.6
</td>
</tr>
<tr>
<td style="text-align:left;padding-left: 2em;" indentlevel="1">
L
</td>
<td style="text-align:center;">
3 (10%)
</td>
<td style="text-align:center;">
1 (3.6%)
</td>
<td style="text-align:center;">
</td>
<td style="text-align:center;">
</td>
</tr>
<tr>
<td style="text-align:left;padding-left: 2em;" indentlevel="1">
R
</td>
<td style="text-align:center;">
27 (90%)
</td>
<td style="text-align:center;">
27 (96%)
</td>
<td style="text-align:center;">
</td>
<td style="text-align:center;">
</td>
</tr>
<tr>
<td style="text-align:left;">
age
</td>
<td style="text-align:center;">
34.5±7.2
</td>
<td style="text-align:center;">
34.6±10.1
</td>
<td style="text-align:center;">
-0.062
</td>
<td style="text-align:center;">
\>0.9
</td>
</tr>
<tr>
<td style="text-align:left;">
sex
</td>
<td style="text-align:center;">
</td>
<td style="text-align:center;">
</td>
<td style="text-align:center;">
0.210
</td>
<td style="text-align:center;">
0.6
</td>
</tr>
<tr>
<td style="text-align:left;padding-left: 2em;" indentlevel="1">
F
</td>
<td style="text-align:center;">
8 (27%)
</td>
<td style="text-align:center;">
9 (32%)
</td>
<td style="text-align:center;">
</td>
<td style="text-align:center;">
</td>
</tr>
<tr>
<td style="text-align:left;padding-left: 2em;" indentlevel="1">
M
</td>
<td style="text-align:center;">
22 (73%)
</td>
<td style="text-align:center;">
19 (68%)
</td>
<td style="text-align:center;">
</td>
<td style="text-align:center;">
</td>
</tr>
<tr>
<td style="text-align:left;">
mk6240.session
</td>
<td style="text-align:center;">
</td>
<td style="text-align:center;">
</td>
<td style="text-align:center;">
1.07
</td>
<td style="text-align:center;">
0.3
</td>
</tr>
<tr>
<td style="text-align:left;padding-left: 2em;" indentlevel="1">
1
</td>
<td style="text-align:center;">
23 (77%)
</td>
<td style="text-align:center;">
18 (64%)
</td>
<td style="text-align:center;">
</td>
<td style="text-align:center;">
</td>
</tr>
<tr>
<td style="text-align:left;padding-left: 2em;" indentlevel="1">
2
</td>
<td style="text-align:center;">
7 (23%)
</td>
<td style="text-align:center;">
10 (36%)
</td>
<td style="text-align:center;">
</td>
<td style="text-align:center;">
</td>
</tr>
<tr>
<td style="text-align:left;">
mCi.net
</td>
<td style="text-align:center;">
6.42±0.59
</td>
<td style="text-align:center;">
6.27±0.71
</td>
<td style="text-align:center;">
0.771
</td>
<td style="text-align:center;">
0.4
</td>
</tr>
<tr>
<td style="text-align:left;">
mk6240.Tdiff
</td>
<td style="text-align:center;">
5±11
</td>
<td style="text-align:center;">
6±9
</td>
<td style="text-align:center;">
-0.117
</td>
<td style="text-align:center;">
\>0.9
</td>
</tr>
<tr>
<td style="text-align:left;">
EpiTrack.Score.age.corrected
</td>
<td style="text-align:center;">
37±3
</td>
<td style="text-align:center;">
35±5
</td>
<td style="text-align:center;">
1.77
</td>
<td style="text-align:center;">
0.083
</td>
</tr>
<tr>
<td style="text-align:left;">
Epi.Acc.D.per
</td>
<td style="text-align:center;">
68±20
</td>
<td style="text-align:center;">
48±22
</td>
<td style="text-align:center;">
3.51
</td>
<td style="text-align:center;">
\<0.001
</td>
</tr>
<tr>
<td style="text-align:left;">
task.semantic
</td>
<td style="text-align:center;">
0.83±0.10
</td>
<td style="text-align:center;">
0.84±0.09
</td>
<td style="text-align:center;">
-0.233
</td>
<td style="text-align:center;">
0.8
</td>
</tr>
<tr>
<td style="text-align:left;">
hip.ipsi
</td>
<td style="text-align:center;">
3,962±296
</td>
<td style="text-align:center;">
3,684±503
</td>
<td style="text-align:center;">
2.54
</td>
<td style="text-align:center;">
0.015
</td>
</tr>
<tr>
<td style="text-align:left;">
hip.cntr
</td>
<td style="text-align:center;">
4,106±394
</td>
<td style="text-align:center;">
3,847±490
</td>
<td style="text-align:center;">
2.21
</td>
<td style="text-align:center;">
0.032
</td>
</tr>
</tbody>
<tfoot>
<tr>
<td style="padding: 0; " colspan="100%">
<sup>1</sup> n (%); Mean±SD
</td>
</tr>
<tr>
<td style="padding: 0; " colspan="100%">
<sup>2</sup> Fisher’s exact test; Welch Two Sample t-test; Pearson’s
Chi-squared test
</td>
</tr>
</tfoot>
</table>

## 18F-MK6240 SUVR

<table style="NAborder-bottom: 0; color: black; " class="table">
<thead>
<tr>
<th style="text-align:left;">
18F-MK6240 SUVR
</th>
<th style="text-align:center;">
Healthy <br>N = 30
</th>
<th style="text-align:center;">
Patient <br>N = 28
</th>
<th style="text-align:center;">
Statistic
</th>
<th style="text-align:center;">
p-value
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
mk6240.mean
</td>
<td style="text-align:center;">
1.06±0.11
</td>
<td style="text-align:center;">
1.14±0.13
</td>
<td style="text-align:center;">
-2.47
</td>
<td style="text-align:center;">
0.017
</td>
</tr>
<tr>
<td style="text-align:left;">
mk6240.mean.ipsi
</td>
<td style="text-align:center;">
1.08±0.11
</td>
<td style="text-align:center;">
1.15±0.13
</td>
<td style="text-align:center;">
-2.26
</td>
<td style="text-align:center;">
0.028
</td>
</tr>
<tr>
<td style="text-align:left;">
mk6240.mean.contra
</td>
<td style="text-align:center;">
1.05±0.11
</td>
<td style="text-align:center;">
1.14±0.14
</td>
<td style="text-align:center;">
-2.55
</td>
<td style="text-align:center;">
0.014
</td>
</tr>
<tr>
<td style="text-align:left;">
mk6240.sig
</td>
<td style="text-align:center;">
1.17±0.12
</td>
<td style="text-align:center;">
1.35±0.15
</td>
<td style="text-align:center;">
-5.04
</td>
<td style="text-align:center;">
\<0.001
</td>
</tr>
<tr>
<td style="text-align:left;">
mk6240.sig.ipsi
</td>
<td style="text-align:center;">
1.21±0.11
</td>
<td style="text-align:center;">
1.39±0.16
</td>
<td style="text-align:center;">
-5.01
</td>
<td style="text-align:center;">
\<0.001
</td>
</tr>
<tr>
<td style="text-align:left;">
mk6240.sig.contra
</td>
<td style="text-align:center;">
1.14±0.12
</td>
<td style="text-align:center;">
1.31±0.17
</td>
<td style="text-align:center;">
-4.50
</td>
<td style="text-align:center;">
\<0.001
</td>
</tr>
</tbody>
<tfoot>
<tr>
<td style="padding: 0; " colspan="100%">
<sup>1</sup> Mean±SD
</td>
</tr>
<tr>
<td style="padding: 0; " colspan="100%">
<sup>2</sup> Welch Two Sample t-test
</td>
</tr>
</tfoot>
</table>
<table style="NAborder-bottom: 0; color: black; " class="table">
<thead>
<tr>
<th style="text-align:left;">
Patients
</th>
<th style="text-align:center;">
Patient <br>N = 28
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
duration
</td>
<td style="text-align:center;">
12.6±9.8
</td>
</tr>
<tr>
<td style="text-align:left;">
lateralization
</td>
<td style="text-align:center;">
</td>
</tr>
<tr>
<td style="text-align:left;padding-left: 2em;" indentlevel="1">
BL
</td>
<td style="text-align:center;">
1 (3.6%)
</td>
</tr>
<tr>
<td style="text-align:left;padding-left: 2em;" indentlevel="1">
L
</td>
<td style="text-align:center;">
12 (43%)
</td>
</tr>
<tr>
<td style="text-align:left;padding-left: 2em;" indentlevel="1">
R
</td>
<td style="text-align:center;">
13 (46%)
</td>
</tr>
<tr>
<td style="text-align:left;padding-left: 2em;" indentlevel="1">
unclear
</td>
<td style="text-align:center;">
2 (7.1%)
</td>
</tr>
<tr>
<td style="text-align:left;">
origin
</td>
<td style="text-align:center;">
</td>
</tr>
<tr>
<td style="text-align:left;padding-left: 2em;" indentlevel="1">
mTLE
</td>
<td style="text-align:center;">
16 (57%)
</td>
</tr>
<tr>
<td style="text-align:left;padding-left: 2em;" indentlevel="1">
TLE
</td>
<td style="text-align:center;">
12 (43%)
</td>
</tr>
</tbody>
<tfoot>
<tr>
<td style="padding: 0; " colspan="100%">
<sup>1</sup> Mean±SD; n (%)
</td>
</tr>
</tfoot>
</table>

## Figure1.B \| Group Differences in MK-6240 Uptake.

Violin plots display the mean SUVR values for significant regions in
each hemisphere by group, with significant differences assessed using a
two-tailed Wilcoxon rank-sum test.

![](Increased_in-vivo_tau_in_TLE_files/figure-gfm/figure1b.ipsi-1.png)<!-- -->

## Figure.3A \| Tau MK-6240 SUVR and clinical relationships

Scatter plot display of the relationship of mean MK-6240 SUVR with
behavioral and clinical measures. Duration and age are measured in
*years*, while all the behavioral measurements where z-scores based in
the control group.

![](Increased_in-vivo_tau_in_TLE_files/figure-gfm/figure3a-1.png)<!-- -->

## Figure.3B \| Tau MK-6240 SUVR and behavioural relationships

# Correlation with behavior 1000 x 945 px

![](Increased_in-vivo_tau_in_TLE_files/figure-gfm/figure3b-1.png)<!-- -->

## Supplementary Figure.4 \| Mean longitudinal *18F-MK6240*

Differential trajectories of $[F^18]MK-6240$ uptake by group between the
PET scan session 1 and 2 in the significant areas. The lines correspond
to the longitudinal subjects.

![](Increased_in-vivo_tau_in_TLE_files/figure-gfm/Supfigure4-1.png)<!-- -->

    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: mk6240.sig ~ mk6240.Tdiff.yrs * group + (1 | participant_id)
    ##    Data: mk.df
    ## 
    ## REML criterion at convergence: -67
    ## 
    ## Scaled residuals: 
    ##      Min       1Q   Median       3Q      Max 
    ## -1.35004 -0.40287  0.02544  0.50538  1.59876 
    ## 
    ## Random effects:
    ##  Groups         Name        Variance Std.Dev.
    ##  participant_id (Intercept) 0.013056 0.11426 
    ##  Residual                   0.004466 0.06683 
    ## Number of obs: 58, groups:  participant_id, 39
    ## 
    ## Fixed effects:
    ##                                Estimate Std. Error        df t value Pr(>|t|)
    ## (Intercept)                    1.353664   0.030303 41.836335  44.670  < 2e-16
    ## mk6240.Tdiff.yrs               0.007754   0.020111 20.844248   0.386    0.704
    ## groupHealthy                  -0.194792   0.042135 41.146436  -4.623 3.71e-05
    ## mk6240.Tdiff.yrs:groupHealthy -0.016574   0.025674 21.249543  -0.646    0.525
    ##                                  
    ## (Intercept)                   ***
    ## mk6240.Tdiff.yrs                 
    ## groupHealthy                  ***
    ## mk6240.Tdiff.yrs:groupHealthy    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##             (Intr) mk6240.T. grpHlt
    ## mk6240.Tdf. -0.246                 
    ## groupHelthy -0.719  0.177          
    ## mk6240.T.:H  0.193 -0.783    -0.223

    ##  group   mk6240.Tdiff.yrs.trend     SE   df lower.CL upper.CL
    ##  Patient                0.00775 0.0203 19.9  -0.0347   0.0502
    ##  Healthy               -0.00882 0.0162 20.9  -0.0425   0.0249
    ## 
    ## Degrees-of-freedom method: kenward-roger 
    ## Confidence level used: 0.95

No significant interaction (`mk6240.Tdiff.yrs:groupHealthy`, p = 0.525)
was found, suggesting that the rate of change in `mk6240.sig` over time
is not significantly different between the Healthy and Patient groups.

- **Healthy Group**: The estimated slope is `-0.0088` per year (95% CI:
  -0.0425 to 0.0249). This suggests a slight decrease in `mk6240.sig`
  over time, but the change is small and not statistically significant
  (p = 0.704).

- **Patient Group**: The estimated slope is `0.0078` per year (95% CI:
  -0.0347 to 0.0502), indicating a slight increase in `mk6240.sig` over
  time, though this change is also not statistically significant.

There is no strong evidence that `mk6240.sig` changes over time in
either the Healthy or Patient groups, nor that the rate of change
differs significantly between them. However, the significant main effect
of group (p \< 0.001) suggests that `mk6240.sig` is, on average, lower
in the Healthy group compared to Patients.
