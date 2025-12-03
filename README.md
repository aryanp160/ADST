# ADST 2.0: Secure Federated Learning 

**ADST (Authenticated Distributed Secure Training)** is a privacy-preserving Federated Learning system simulation. It demonstrates how multiple workers can collaboratively train a machine learning model without sharing their raw data or their individual gradient updates with the central server.

## 🌟 Key Features

*   **Secure Aggregation**: Uses a pairwise masking protocol (similar to Google's Secure Aggregation) to ensure the coordinator can only decrypt the *sum* of the gradients, not individual ones.
*   **Differential Privacy (DP)**: Implements **Gradient Clipping** and **Gaussian Noise** injection at the worker level to provide rigorous privacy guarantees.
*   **Privacy-Preserving**:
    *   **Data Privacy**: Raw training data never leaves the worker's local machine.
    *   **Model Privacy**: Individual model updates are masked using cryptographic blinding factors.
*   **Hybrid Networking**:
    *   **TCP**: Used for reliable control messages, key exchange, and peer discovery.
    *   **UDP**: Used for high-throughput transmission of encrypted gradient chunks.
*   **Real-Time Dashboard**: A professional-grade Streamlit dashboard ("Mission Control") to visualize training progress, gradient norms, privacy status, and validation accuracy.
*   **Cryptography**: Built with standard primitives (`cryptography` library):
    *   **Ed25519**: For digital signatures and identity.
    *   **X25519**: For Diffie-Hellman key exchange.
    *   **AES-GCM**: For authenticated encryption of transport and gradients.

## 📂 Project Structure

*   `coordinator.py`: The central server. It manages epochs, handles worker registration, and aggregates the masked gradients.
*   `worker.py`: The client. It trains a local CNN on its private data, communicates with peers to generate masks, and sends encrypted updates.
*   `demo.py`: An orchestrator script that automatically launches one coordinator, multiple workers, and the dashboard for a complete local demo.
*   `dashboard.py`: A web interface to monitor the training process.
*   `data/`: Directory containing local datasets for each worker (`site1`, `site2`, etc.) and a validation set (`val`).

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed.

### Installation

Install the required dependencies:

```bash
pip install torch torchvision cryptography streamlit pandas matplotlib numpy
```

### Running the Demo

The easiest way to see the system in action is to run the `demo.py` script. This will start the coordinator, 4 workers, and the dashboard automatically.

1.  **Start the demo**:
    ```bash
    python demo.py
    ```

2.  **View the Dashboard**:
    Open your browser and navigate to:
    [http://localhost:8501](http://localhost:8501)

3.  **Watch it Train**:
    *   The system will run for a fixed number of epochs (default: 3).
    *   You will see workers connecting, performing handshakes, and sending gradients.
    *   The dashboard will update with the current epoch, active workers, and training metrics.

4.  **Stopping**:
    The demo will automatically exit after the training completes. You can also press `Ctrl+C` in the terminal to stop it early.

## 🛠️ Configuration

*   **Number of Workers**: You can modify `NUM_WORKERS` in `demo.py`.
*   **Epochs**: Adjust `MAX_EPOCHS` in `coordinator.py` and `worker.py`.
*   **Privacy Budget**: Tune `DP_NOISE_SCALE` and `DP_CLIP_NORM` in `worker.py` to balance privacy and accuracy.
*   **Network Ports**: Default ports are 9000 (TCP) and 9001 (UDP). These can be changed in the arguments.

## ⚠️ Note

This is a **demonstration** project intended for educational purposes. While it uses real cryptography, it is a simulation running on `localhost` and is not intended for production deployment without further hardening.
