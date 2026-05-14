# 🛡️ Graph Intelligence System for Cyber Fraud Detection and Network Forensics

An advanced **Graph Neural Network (GNN)-powered cyber fraud detection system** designed to detect fraudulent financial transactions using **heterogeneous graph intelligence, network forensics, and relational learning**.

This project leverages **GraphSAGE-based Graph Neural Networks (GNNs)** to model complex relationships between **accounts, devices, IP addresses, locations, and merchants**, enabling high-performance fraud detection beyond traditional machine learning approaches.

---

## 🚀 Project Overview

Traditional fraud detection systems often treat transactions independently, making it difficult to identify hidden relationships between entities involved in fraudulent activities.

This project addresses that limitation using **graph intelligence**, where financial ecosystems are represented as interconnected networks to uncover suspicious patterns and relational anomalies.

The system performs:

- **Fraud Detection using Graph Neural Networks (GraphSAGE)**
- **Heterogeneous Graph Construction**
- **Transaction Intelligence & Network Forensics**
- **Behavioral Feature Engineering**
- **Fraud Risk Analysis Dashboard**
- **Comparative Benchmarking with Traditional ML Models**

---

## 🧠 Problem Statement

Conventional fraud detection approaches struggle to capture:

- Hidden relationships between accounts
- Device sharing patterns
- Suspicious IP associations
- Abnormal location-based behaviors
- Complex fraud rings and coordinated attacks

To solve this, we model financial transactions as a **heterogeneous graph network**, enabling the system to learn relational dependencies and detect fraud with significantly higher accuracy.

---

## 🏗️ System Architecture

### Heterogeneous Graph Components

The financial ecosystem is represented as:

### **Nodes**
- 👤 Accounts
- 💻 Devices
- 🌐 IP Addresses
- 📍 Locations
- 🏪 Merchants

### **Edges**
- Account → Account (Transaction)
- Account → Device
- Account → IP Address
- Account → Location
- Account → Merchant

This graph-based representation allows the model to detect **suspicious behavioral connections** that traditional tabular models fail to capture.

---

## ⚙️ Technologies Used

### Programming & Frameworks
- Python
- PyTorch
- PyTorch Geometric (PyG)
- Jupyter Notebook

### Data Science & ML
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

### Graph Analytics
- GraphSAGE
- NetworkX
- Heterogeneous Graph Modeling

### Visualization & Dashboard
- Streamlit
- Interactive Fraud Dashboard

---

## 📂 Project Structure

```bash
FINAL YEAR PROJECT/
│── 01_EDA_GNN CyberFraud Detection.ipynb
│── 02_Graph_Construction.ipynb
│── 03_GNN_Data_Preparation.ipynb
│── 04_GNN_Model.ipynb
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── Financial Transactions Dataset for Fraud Detection.csv
│   ├── fraud_graph.pt
│   └── transaction_graph.pkl
│
├── model/
│   └── fraud_gnn_model.pt
│
└── reports/
```

---

## 🔬 Workflow Pipeline

### 1️⃣ Exploratory Data Analysis (EDA)
- Fraud distribution analysis
- Transaction trend analysis
- Feature correlation analysis
- Data cleaning & preprocessing

### 2️⃣ Graph Construction
Built a **heterogeneous transaction graph** using:

- Accounts
- Devices
- IPs
- Locations
- Merchants

to represent hidden transactional relationships.

### 3️⃣ Graph Data Preparation
Performed:

- Node encoding
- Feature normalization
- Edge indexing
- Graph tensor conversion

using **PyTorch Geometric (HeteroData)**.

### 4️⃣ Graph Neural Network Training
Implemented a **GraphSAGE-based GNN model** for:

- Edge-level fraud classification
- Relational fraud detection
- Network anomaly learning

---

## 📊 Model Performance

### 🔥 GraphSAGE Fraud Detection Results

| Metric | Score |
|--------|-------|
| Accuracy | **99.65%** |
| Precision | **83.44%** |
| Recall | **100%** |
| F1-Score | **90.97%** |
| ROC-AUC | **0.9996** |
| PR-AUC | **0.9785** |

### Confusion Matrix

| | Predicted Legit | Predicted Fraud |
|---|---:|---:|
| Actual Legit | 14,680 | 53 |
| Actual Fraud | 0 | 267 |

✅ **Zero False Negatives** achieved.

This ensures **all fraudulent transactions are detected**, making the system highly reliable for financial fraud prevention.

---

## 📈 Benchmark Comparison

Traditional machine learning models struggled due to the relational complexity of fraud networks.

### Baseline Models Tested

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

These models achieved approximately:

**ROC-AUC ≈ 0.49–0.50**

(Graph-aware learning dramatically outperformed traditional approaches.)

---

## 🧪 Feature Engineering

Several behavioral fraud indicators were engineered:

### Velocity Score
Measures abnormal transaction frequency.

### Geo-Anomaly Score
Detects unusual geographic activity patterns.

### Spending Deviation Score
Identifies spending behavior deviations from historical patterns.

### Temporal Features
- 24-hour transaction count
- Transaction amount aggregation
- Device/IP reuse analysis

---

## 📊 Dashboard Features

The system includes an interactive fraud analytics dashboard with:

### Transaction Intelligence
Monitor suspicious transactions and fraud probability.

### Model Benchmark
Compare GraphSAGE against traditional ML models.

### Graph Structure Analysis
Visualize interconnected fraud networks.

### GNN Deep Dive
Understand node relationships and prediction reasoning.

### Fraud Forensics
Perform network investigation and anomaly tracing.

### Data Explorer
Analyze transaction behavior interactively.

---

## 📦 Dataset

Dataset contains **100,000+ financial transaction records** with highly imbalanced fraud distribution.

### Graph Statistics

| Entity Type | Count |
|-------------|------:|
| Accounts | 179,590 |
| Devices | 99,473 |
| IP Addresses | 99,999 |
| Locations | 8 |
| Merchants | 8 |
| Transaction Edges | 100,000 |

---

## ▶️ Installation & Setup

### Clone Repository

```bash
git clone https://github.com/yourusername/graph-intelligence-fraud-detection.git
cd graph-intelligence-fraud-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Dashboard

```bash
cd dashboard
streamlit run app.py
```

---

## 💡 Key Contributions

✔ Built a **heterogeneous fraud graph network**

✔ Implemented **GraphSAGE-based GNN fraud detection**

✔ Achieved **99.65% fraud classification accuracy**

✔ Developed an **interactive fraud intelligence dashboard**

✔ Performed **network forensics for cyber fraud analysis**

✔ Outperformed traditional ML models by leveraging **graph relationships**

---

## 🎯 Future Improvements

- Real-time fraud detection pipeline
- Explainable AI (XAI) for fraud reasoning
- Dynamic graph learning
- Temporal Graph Neural Networks (TGNN)
- Large-scale deployment using distributed systems

---

## 👨‍💻 Authors

**Princy Hareshbhai Patel** 

**Vivek Nair** 

**Meet Kathiriya** 


**GitHub:** https://github.com/princyhareshbhai1530

---

## 📜 License

This project is developed for **academic and research purposes**.
