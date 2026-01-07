# Nemo — AI-Based Epilepsy Detection System

## Overview
Nemo is an AI-driven medical system designed to **classify whether a patient exhibits epileptic activity** based on neurological signal data. The project focuses on applying machine-learning techniques to neural data while accounting for **real-world medical, computational, and power-consumption constraints**, with the long-term goal of enabling **low-power, wearable, or edge medical devices**.

---

## Problem Statement
Epilepsy detection from neural signals requires models that are not only accurate, but also **reliable, efficient, and suitable for continuous monitoring**. Traditional machine-learning approaches often prioritize performance without considering deployment constraints such as energy consumption, inference cost, and hardware limitations. Nemo addresses this gap by exploring AI approaches that balance **detection accuracy with efficiency and feasibility**.

---

## Project Goals
- Detect epilepsy-related patterns from neural signal data  
- Evaluate machine-learning models for medical reliability and stability  
- Explore **spiking neural networks (SNNs)** for low-power inference  
- Analyze **power-consumption trade-offs** for potential wearable or edge deployment  
- Align AI design decisions with real-world medical and hardware constraints  

---

## Technical Approach

### Machine Learning & Model Evaluation
The project involves the development and evaluation of machine-learning models aimed at classifying epileptic activity from neurological signals. Models were analyzed based on:
- Classification accuracy and reliability  
- Inference behavior and stability  
- Limitations and failure cases in medical contexts  

### Neuromorphic Computing
To address energy-efficiency requirements, the project explored **neuromorphic AI approaches** using **spiking neural networks (SNNs)**. The **Rockpool framework** was integrated and evaluated to experiment with spiking-based architectures and to understand how they differ from traditional neural networks in terms of:
- Inference behavior  
- Computational efficiency  
- Suitability for low-power and embedded environments  

### Power Consumption Considerations
Power consumption was treated as a **core design constraint** rather than an afterthought. Model architectures and inference patterns were analyzed to understand how they impact:
- Energy efficiency  
- Continuous operation feasibility  
- Deployment on battery-powered or wearable medical devices  

Design decisions balanced detection performance with computational and energy costs.

---

## Data Handling
The project involved working with neurological signal data, including:
- Data preprocessing and structuring  
- Preparing signals for model evaluation  
- Ensuring compatibility with both traditional ML models and spiking neural networks  

---

## Collaboration & Engineering Focus
Development was carried out in collaboration with team members, ensuring that AI solutions aligned with:
- Medical applicability  
- Data limitations  
- Hardware and deployment assumptions  

The project emphasized **practical engineering trade-offs** rather than purely theoretical optimization.

---

## Key Contributions
- AI-based epilepsy detection using neural signal data  
- Development and evaluation of ML and SNN models  
- Integration and experimentation with the **Rockpool neuromorphic framework**  
- Analysis of model accuracy, behavior, and limitations  
- Evaluation of **power-consumption and performance trade-offs**  
- Alignment of AI design with medical and deployment constraints  

---

## Application Context
Nemo is designed as a foundation for **future low-power medical AI systems**, particularly those intended for:
- Wearable devices  
- Edge-based medical monitoring  
- Continuous neurological signal analysis  

---

## Disclaimer
This project focuses on technical feasibility and AI system design. It is not a standalone medical diagnostic product and is intended for research and engineering exploration within a controlled development environment.
