# In-vivo evidence for increased tau deposition in temporal lobe epilepsy  

[![GitHub stars](https://img.shields.io/github/stars/MICA-MNI/2025_in-vivo_tauPET-mk6240_TLE.svg?style=flat&label=%E2%AD%90%EF%B8%8F%20stars&color=brightgreen)](https://github.com/MICA-MNI/2025_in-vivo_tauPET-mk6240_TLE/stargazers)
[![License: MIT](https://img.shields.io/github/license/MICA-MNI/2025_in-vivo_tauPET-mk6240_TLE)](https://github.com/MICA-MNI/2025_in-vivo_tauPET-mk6240_TLE/blob/main/LICENSE)

**Cite:** 
> *[Raúl R. Cruces](mailto:raul.rodriguezcruces@mcgill.ca), Jack Lam, Thaera Arafat, Jessica Royer, Judy Chen, Ella Sahlas, Raluca Pana, Robert Hopewell, Hung-Hsin Hsiao, Gassan Massarweh, Jean-Paul Soucy, Sylvia Villeneuve, Lorenzo Caciagli, Matthias Koepp, Andrea Bernasconi, Neda Ladbon-Bernasconi, [Boris C. Bernhardt](mailto:boris.bernhardt@mcgill.ca)*. (2025). In-vivo evidence for increased tau deposition in temporal lobe epilepsy ... 

**DOI:** [TBD]().  

**Preprint:** avaliable at [TBD]().   

**Data repository:** avaliable at [OSF](https://osf.io/ct3gw/wiki/home/).  

**Keywords:** Temporal lobe epilepsy | 18F-MK-6240 | tauopathy | neuroimaging | neuroinflammation
 
**Short title:** *In-vivo findings of increased tau uptake in TLE*


## Overview  
This repository contains the code and data processing pipelines to replicate the findings of our study investigating in-vivo tau aggregation in **drug-resistant temporal lobe epilepsy (TLE)** using **18F-MK6240 PET imaging**. The study explores:  

- Group differences in tau deposition between TLE patients and healthy controls.  
- Relationships between tau uptake and structural/functional connectivity.  
- Associations between tau accumulation, disease duration, and cognitive function.  

## Dataset  
### Participants  
- **18 TLE patients** (7 with longitudinal follow-up)  
- **20 healthy controls** (10 with longitudinal follow-up)  

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
| [`OSF ct3gw`](https://osf.io/ct3gw/wiki/home/) | This repository contains all the hosted data.  |

## Abstract

**Background and Objectives.** Temporal lobe epilepsy (TLE), the most common drug-resistant epilepsy in adults, is associated with structural alterations beyond the primary mesiotemporal substrate. Although TLE is traditionally not considered a neurodegenerative disorder, emerging evidence from ex-vivo specimens have shown elevated levels of misfolded tau protein, a hallmark of neurodegeneration. In this study, we assessed the in-vivo deposits of tau-aggregates in TLE using 18F-MK6240, a recently validated in-vivo positron emission tomography (PET) tracer.
 
**Methods.** We included 18 drug-resistant TLE patients, and 20 age-and sex-matched healthy controls (7 and 10 with longitudinal follow up). PET data were registered to the T1-weighted 3T MRI, where they were partial-volume corrected, normalized for cerebellar grey matter uptake to obtain a standardized uptake value ratio, and mapped to cortical surfaces. We compared MK6240 uptake between TLE patients and healthy controls using linear mixed effects models, controlling for age and sex, and multiple comparisons. We examined the correlations between the group difference in MK6240 and functional and structural connectivity information, to study network effects on tau uptake. Finally, we evaluated relationships between the tau uptake and clinical as well as cognitive measures.
 
**Results.** Compared to controls, patients presented with increased MK6240 uptake in bilateral superior and medial temporal as well as parietal regions. No significant longitudinal changes were observed within the one-year evaluation period. The spatial pattern of MK6240 uptake increases was related to local functional and structural connectome architecture, suggesting an effect of inter-regional connectivity. While age had no significant effect, disease duration was positively related to overall MK6240 uptake in patients. Behavioral measures of executive function and episodic memory demonstrated a negative relationship with MK6240 uptake. In contrast, measures of semantic memory showed no significant relationship.
 
**Discussion.** Our findings show elevated phosphorylated tau measured with MK6240 uptake in TLE patients in a bilateral temporo-posterior distribution, which correlates to disease duration, local connectivity patterns, and cognitive dysfunction.  While our findings suggest that tau accumulation plays a role in TLE progression, confirmation in larger samples with longer follow up is warranted to verify the hypothesized role of tau as a modulator of neurodegenerative cascades in drug-resistant epilepsy.

