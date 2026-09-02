# Temperature-Control-Agent
A tiny ml project that used KAGGLE dataset to analyze key temperature control points and use them to provide coefficients for linear model to use as prediction model in  a ESP32.

# Project Overview & Description

The **[Temperature-Control-Agent](https://github.com/Dipunmohanty/Temperature-Control-Agent)** is an end-to-end TinyML and IoT project designed to perform temperature analysis, model training, and edge inference. 

The project leverages historical temperature control data from Kaggle to train a lightweight linear prediction model. The extracted model coefficients are then embedded onto an **ESP32 microcontroller** (written in C/C++), enabling real-time, low-latency temperature prediction and control directly on resource-constrained edge hardware without relying on cloud computation.

---

## 🛠️ Component Breakdown & Functionality

### 1. Data Engineering & Analysis (`/Kaggle_data` & Jupyter Notebooks)
* **Kaggle Dataset Integration**: Houses raw and preprocessed historical temperature readings used to identify critical thermal control points.
* **Feature Selection & Analysis**: Evaluates environmental parameters and control variables to determine key correlation factors for thermal management.

---

### 2. Model Training & Coefficient Extraction (`/Model` & `/RESULTS`)
* **TinyML Training Pipeline (`Python` / `Jupyter Notebook`)**: Builds and validates lightweight linear regression models targeting low computational complexity.
* **Coefficient Export**: Generates compact mathematical weights and bias values suitable for execution on microcontrollers with limited memory.
* **Validation & Performance Metrics (`/RESULTS` & `/RESULTS.zip`)**: Stores output logs, evaluation metrics, and comparative error analysis to ensure model reliability prior to edge deployment.

---

### 3. Edge Execution & Microcontroller Firmware (C / ESP32)
* **ESP32 Firmware (`C`)**: Implements the linear prediction algorithm directly on the ESP32 board using exported mathematical coefficients.
* **Real-Time Edge Inference**: Computes temperature trends and control signals locally with minimal power consumption and real-time responsiveness.

---

### 4. Documentation & Research (`/Paper`, `/Paper_2`, `/Paper_6`)
* **Academic & Technical Documentation**: Contains structural research papers, system diagrams, and experiment logs documenting the methodology, hardware setup, and model evaluation results.
