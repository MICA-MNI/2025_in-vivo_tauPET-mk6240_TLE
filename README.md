# In-vivo evidence for increased tau deposition in temporal lobe epilepsy  

[![GitHub stars](https://img.shields.io/github/stars/MICA-MNI/2025_in-vivo_tauPET-mk6240_TLE.svg?style=flat&label=%E2%AD%90%EF%B8%8F%20stars&color=brightgreen)](https://github.com/MICA-MNI/2025_in-vivo_tauPET-mk6240_TLE/stargazers)
[![License: MIT](https://img.shields.io/github/license/MICA-MNI/2025_in-vivo_tauPET-mk6240_TLE)](https://github.com/MICA-MNI/2025_in-vivo_tauPET-mk6240_TLE/blob/main/LICENSE)

**Cite:** 
> *[Raúl R. Cruces](mailto:raul.rodriguezcruces@mcgill.ca), Jack Lam, Thaera Arafat, Jessica Royer, Judy Chen, Ella Sahlas, Arielle Dascal, Daniel Mendelson, Raluca Pana, Robert Hopewell, Chris Hung-Hsin Hsiao, Gassan Massarweh, Jean-Paul Soucy, Pedro Rosa-Neto, Marie-Christine Guiot, Sylvia Villeneuve, Lorenzo Caciagli, Matthias J. Koepp, Andrea Bernasconi, Neda Ladbon-Bernasconi, [Boris C. Bernhardt](mailto:boris.bernhardt@mcgill.ca)*. (2026). In-vivo evidence for increased tau deposition in temporal lobe epilepsy ... 

**DOI:** [TBD]().  

**Preprint:** avaliable at [bioRxiv](https://doi.org/10.1101/2025.05.14.654095 ).   

**Data repository:** avaliable at [OSF](https://osf.io/ct3gw).  

**Keywords:** Temporal lobe epilepsy | 18F-MK-6240 | tauopathy | neuroimaging | neuroinflammation
 
**Short title:** *In-vivo findings of increased tau uptake in TLE*


## Overview  
This repository contains the code and data processing pipelines to replicate the findings of our study investigating in-vivo tau aggregation in **drug-resistant temporal lobe epilepsy (TLE)** using **18F-MK6240 PET imaging**. The study explores:  

- Group differences in tau deposition between TLE patients and healthy controls.  
- Relationships between tau uptake and structural/functional connectivity.  
- Associations between tau accumulation, disease duration, and cognitive function.  

## Dataset  
### Participants  
- **28 TLE patients** (13 with longitudinal follow-up)  
- **28 healthy controls** (7 with longitudinal follow-up)  

### Imaging Data Acquisition  
- **MRI**: 3T Siemens Magnetom Prisma-Fit scanner  
- **PET**: Siemens high-resolution research tomograph (HRRT)  

## Processing Steps  
1. **Preprocessing**: Registration of PET to MRI, partial volume correction, and SUVR computation (normalized to cerebellar grey matter).  
2. **Statistical Analysis**: Linear mixed-effects models to assess group differences in tau deposition, controlling for age and sex.  
3. **Connectivity Analysis**: Examining functional and structural connectivity patterns in relation to tau uptake.  
4. **Cognitive and Clinical Correlations**: Investigating relationships between tau deposition, disease duration, and cognitive performance.  

 ## Repository content
| Directories   | Description                |
|---------------|----------------------------|
| [`./scripts`](https://github.com/MICA-MNI/in-vivo_tauPET-mk6240_TLE/tree/master/scripts)     | Notebooks for processing, and perform the analysis to reproduce the findings.   |
| [`OSF ct3gw`](https://osf.io/ct3gw/wiki/home/) | This repository hosts all data used in the project.  |

## Abstract

Temporal lobe epilepsy (TLE), the most common pharmaco-resistant epilepsy in adults, has been linked to structural brain changes extending beyond the mesiotemporal areas. While not traditionally viewed as a neurodegenerative disorder, recent ex-vivo studies have shown elevated levels of misfolded tau protein in TLE. 

This study investigated tau deposition in TLE patients using the in-vivo PET tracer [18F]MK-6240. We studied 28 TLE patients and 28 healthy controls to assess tau uptake and its relationship with brain connectivity, clinical variables, and cognitive function. 

Compared to controls, TLE patients exhibited markedly increased [18F]MK-6240 uptake in bilateral superior and medial temporal regions and the parietal cortex, with tau accumulation following regional functional and structural connectivity and cognitive impairment. 

These findings suggest that tau accumulation contributes to cognitive decline observed in TLE, supporting a potential role of tau in epilepsy-related neurodegeneration.

