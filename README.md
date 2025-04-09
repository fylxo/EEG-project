# 🧠 Neural Patterns in Motor Execution and Imagery

**Author**: Flavio Caroli  
**Date**: April 9, 2025  
**University Project**: EEG Motor Imagery Analysis  
**Dataset**: [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)

---

## 🧩 Project Overview

This project investigates the **neural similarity between real and imagined hand movements**, using EEG data. The analysis combines **three complementary approaches**:

1. **Functional Connectivity** (PLV, iPLV, wPLI)  
2. **Classification Performance** (CSP + LDA)  
3. **Neural Dynamics** via Detrended Fluctuation Analysis (DFA)

The goal is to evaluate the *functional equivalence hypothesis*, which proposes that motor execution and imagery involve overlapping neural circuits.

---

## 🧪 Methods Summary

- **Preprocessing**: Bandpass filtering (6–30 Hz), epoch extraction, optional ICA  
- **Connectivity Analysis**: Phase-based metrics computed for motor and control electrode pairs  
- **Classification**: Common Spatial Patterns + Linear Discriminant Analysis to distinguish between conditions  
- **DFA**: Used to analyze scale-free dynamics and criticality of EEG signals in different frequency bands  

---

## 📁 Repository Structure
├── experiment1.ipynb # Analysis: Connectivity, Classification, ERD/ERS
├── experiment2.ipynb # DFA analysis & comparisons across conditions
├── my_functions.py # Core EEG preprocessing & connectivity functions
├── my_functions2.py # DFA and envelope-based signal analysis
├── Report-Flavio_Caroli.pdf # Full write-up of the project and results
└── README.md # This file



---

## 📊 Key Findings

- **Connectivity**: Real and Imagined movements both show increased PLV/iPLV/wPLI vs. Rest; minimal difference between Real and Imagined.  
- **Classification**: Real vs. Rest classification = 68.5% accuracy; Real vs. Imagined = 57.8% (barely above chance).  
- **DFA**: Real and Imagined movements show *nearly identical* scaling exponents, consistent with similar underlying neural dynamics.  

These findings **strongly support the functional equivalence hypothesis** and have implications for Brain-Computer Interfaces (BCIs) and neurorehabilitation.

---

📌 Notes
Analysis was run on EEG data from ~30 subjects (subset of 109 available)

ICA was applied optionally; most results were based on a streamlined pipeline with bandpass filtering only

The classification task was intentionally challenging to assess neural similarity, not just accuracy




DISCLAIMER:
This project was created as part of a university course for personal learning.
It is not peer-reviewed research, and the results should not be interpreted as definitive scientific conclusions.
It was fun—please don't take it seriously. 😄
