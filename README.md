<div align="center">

# Healthy-Reference Microbial Network Modeling Framework

### A normative modeling framework for characterizing ecological deviations in microbiome networks

<br>

**Developed by MiniLab**

South China University of Technology

<br>

![Microbiome](https://img.shields.io/badge/Field-Microbiome%20Ecology-blue)
![Network](https://img.shields.io/badge/Method-Normative%20Network%20Modeling-green)
![Analysis](https://img.shields.io/badge/Workflow-Reproducible-orange)

</div>


---

# Overview

Microbial communities are highly organized ecological systems composed of interacting microorganisms rather than independent collections of individual taxa. Although conventional microbiome studies have successfully identified disease-associated microbial signatures, these approaches mainly focus on abundance changes and may overlook alterations in microbial interactions, community organization, and ecological stability.

The **Healthy-Reference Microbial Network Modeling Framework** was developed to characterize microbiome alterations from an ecosystem-level perspective. Instead of defining disease-associated changes solely through microbial abundance differences, this framework establishes a microbial ecological network derived from healthy populations and evaluates how microbial communities deviate from this reference organization under different biological conditions.

The framework follows the concept of **normative modeling**, where the ecological structure observed in healthy populations is considered a reference state. Disease-associated microbiomes are subsequently compared against this reference network to quantify deviations in microbial connectivity, module organization, and ecological relationships.

By integrating microbial network construction, community structure analysis, preservation evaluation, and phenotype association, this framework provides a systematic approach for studying microbiome ecosystem alterations across complex diseases.


---

# Scientific Concept

The central idea of this framework is that disease-associated microbiome alterations may not simply represent the appearance or disappearance of individual microbial members, but rather reflect changes in the organization and stability of the entire microbial ecosystem.

The analytical strategy can be summarized as:

```
Healthy microbial population

          ↓

Construction of healthy-reference ecological network

          ↓

Identification of microbial communities and network organization

          ↓

Projection of disease-associated microbiomes

          ↓

Quantification of ecological deviations

          ↓

Association with biological phenotypes
```

Rather than asking:

> Which microbial taxa are different between healthy and disease states?

the framework focuses on:

> How does the structure and organization of the microbial ecosystem deviate from a healthy reference state?


This perspective allows microbiome alterations to be interpreted from an ecological systems perspective and provides a quantitative approach for investigating microbial community instability, resilience, and disease-associated reorganization.


---

# Framework Description


## Healthy-reference microbial network construction

The first component of the framework is the establishment of a microbial ecological reference network based on healthy individuals.

Microbial abundance profiles from healthy populations are processed and transformed into a suitable representation for network inference. Microbial associations are subsequently estimated to construct an ecological interaction network that represents the organization of a relatively stable microbial community.

This reference network captures the relationships among microbial members, including microbial connectivity patterns, community structure, and ecological modules.

The generated reference network serves as the baseline against which disease-associated microbiome alterations can be evaluated.


---

## Microbial community and module identification

Microbial ecosystems are not randomly organized. Instead, groups of microorganisms with similar ecological relationships often form interconnected communities or modules.

This framework identifies microbial modules within the healthy-reference network and characterizes their structural properties.

Module-based analysis enables investigation of whether disease-associated alterations affect:

- the entire microbial ecosystem;
- specific ecological communities;
- internal module connectivity;
- interactions between different microbial communities.

This approach provides a higher-level interpretation of microbiome alterations beyond individual microbial biomarkers.


---

## Network preservation and ecological deviation analysis

After constructing the healthy-reference network, microbiome profiles from different biological states can be projected onto the reference ecological structure.

The framework evaluates the degree to which microbial relationships observed in healthy populations are preserved or disrupted.

Network deviation analysis includes evaluation of:

- microbial connection preservation;
- loss of ecological relationships;
- changes in module connectivity;
- alterations in network organization.

These measurements provide quantitative indicators of how microbial ecosystems depart from a healthy reference state.


---

## Integration with biological and clinical phenotypes

Microbial ecological deviations can be further integrated with biological information, including clinical phenotypes, disease characteristics, and multi-omics measurements.

The framework supports downstream analyses investigating whether specific network alterations are associated with:

- disease severity;
- clinical heterogeneity;
- biological subtypes;
- potential mechanistic processes.

Such integration provides opportunities for connecting microbial ecosystem changes with host phenotypes.


---

# Computational Workflow

The overall workflow consists of several major analytical steps:

```
Microbial abundance profiles

            ↓

Data preprocessing and quality control

            ↓

Healthy-reference network construction

            ↓

Network topology characterization

            ↓

Community/module detection

            ↓

Disease-state projection

            ↓

Network preservation evaluation

            ↓

Ecological deviation quantification

            ↓

Phenotype association analysis
```


Each step is implemented using reproducible computational scripts and can be adapted according to different microbiome datasets and biological questions.


---

# Repository Organization

```
Healthy-Reference-Microbial-Network-Modeling-Framework/

│
├── README.md
│
├── scripts/
│   ├── preprocessing/
│   ├── network_construction/
│   ├── module_identification/
│   ├── preservation_analysis/
│   ├── statistical_analysis/
│   └── visualization/
│
├── data/
│   └── example_data/
│
├── results/
│   └── output_results/
│
└── environment/
    └── software_information/
```


---

# Input Data

The framework requires microbial abundance profiles and corresponding sample metadata.

The microbial abundance matrix should contain microbial features as rows and samples as columns. Depending on the research design, microbial features can represent different taxonomic levels, including species, genera, or other microbial units.

Example:

| Microbial feature | Sample 1 | Sample 2 | Sample 3 |
|---|---|---|---|
| Species_A | abundance | abundance | abundance |
| Species_B | abundance | abundance | abundance |


Sample metadata can include:

- experimental groups;
- demographic information;
- clinical characteristics;
- quantitative phenotypes;
- additional biological measurements.


---

# Output

The framework generates multiple levels of network-based measurements, including:

- healthy-reference microbial networks;
- microbial interaction structures;
- network topology parameters;
- microbial community assignments;
- preservation metrics;
- ecological deviation scores;
- phenotype-associated network features.

These outputs provide a quantitative description of how microbial ecosystems differ from a healthy reference organization.


---

# Applications

Although initially developed for microbiome-based disease research, the framework can be applied broadly to different biological contexts, including:

- psychiatric disorders;
- brain–gut interaction studies;
- metabolic diseases;
- inflammatory disorders;
- longitudinal microbiome studies;
- multi-omics ecological modeling.


---

# Software Requirements

The framework is implemented using R and Python environments.

Recommended software:

- R ≥ 4.0
- Python ≥ 3.8


Major computational packages include:

**R packages**

- igraph
- FlashWeave
- WGCNA
- tidyverse
- vegan
- ggplot2


**Python packages**

- numpy
- pandas
- scipy
- networkx


---

# About MiniLab

MiniLab focuses on developing computational approaches for understanding complex biological systems through microbiome ecology, network science, and artificial intelligence.

The research interests of MiniLab include:

- microbial ecosystem modeling;
- microbiome–host interactions;
- brain–gut axis research;
- disease heterogeneity;
- multi-omics integration;
- computational methods for biomedical discovery.


MiniLab aims to develop reproducible analytical frameworks that bridge computational modeling and biological interpretation, providing new perspectives for understanding complex diseases.


---

# Reproducibility

All scripts are organized to facilitate transparent and reproducible microbiome network analysis.

Researchers can adapt this framework by modifying:

- reference population selection;
- microbial feature definitions;
- network inference parameters;
- statistical evaluation strategies.

The framework is designed to support applications across different cohorts and biological systems.


---

# Citation

If you use this framework in your research, please cite:

**Healthy-Reference Microbial Network Modeling Framework for Characterizing Ecological Deviations in Microbiome Ecosystems**


---

# Contact

**MiniLab**

South China University of Technology
