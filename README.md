# Multi-Input Multi Output 3D CNN for Dementia Severity Assessment with Incomplete Multimodal Data
This is the repository for the paper Multi Input - Multi Output 3D CNN for Dementia Severity Assessment with Incomplete Multimodal Data

## Abstract

Alzheimer's Disease is the most common cause of dementia, whose progression spans in different stages, from very mild cognitive impairment to mild and severe conditions. In clinical trials, Magnetic Resonance Imaging (MRI) and Positron Emission Tomography (PET) are mostly used for the early diagnosis of neurodegenerative disorders since they provide volumetric and metabolic function information of the brain, respectively. In recent years, Deep Learning (DL) has been employed in medical imaging with promising results. Moreover, the use of the deep neural networks, especially Convolutional Neural Networks (CNNs), has also enabled the development of DL-based solutions in domains characterized by the need of leveraging information coming from multiple data sources, raising the Multimodal Deep Learning (MDL). In this paper, we conduct a systematic analysis of MDL approaches for dementia severity assessment exploiting MRI and PET scans. We propose a Multi Input - Multi Output 3D CNN whose training iterations change according to the characteristic of the input as it is able to handle incomplete acquisitions, in which one image modality is missed. Experiments performed on OASIS-3 dataset show the satisfactory results of the implemented network, which outperforms approaches exploiting both single image modality and different MDL fusion techniques. 

![Alt text](https://github.com/priamus-lab/Multi-Input---Multi-Output-3D-CNN-for-Dementia-Severity-Assessment-with-Incomplete-Multimodal-Data/blob/main/img/graphical.png)


## Dataset

In this work, we consider the OASIS-3 dataset[1], consisting in a compilation of MRI, PET imaging, and clinical data for 1098 patients. The study collects 605 cognitively normal patients and 493 individuals at various stages of cognitive decline. All the MRI, PET, and clinical data acquisition sessions report information about when they are acquired expressed as the number of days since the subject's entry date into the study. The status of dementia was assessed using the Clinical Dementia Rating Scale (CDR).

[1] P. J. LaMontagne, T. L. Benzinger, J. C. Morris, S. Keefe, R. Hornbeck, C. Xiong, E. Grant, J. Hassenstab, K. Moulder, A. G. Vlassenko, et al., Oasis-3: longitudinal neuroimaging, clinical, and cognitive dataset for normal aging and alzheimer disease, MedRxiv (2019).

## Authors
* **Michela Gravina, Angel García-Pedrero, Consuelo Gonzalo-Martín, Carlo Sansone, and Paolo Soda**
