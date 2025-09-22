# Race Strategy Optimization with Reinforcement Learning

This project focuses on **optimizing race strategies** (pit stops, tire choices, fuel management, etc.) using **Reinforcement Learning (RL)**.  
To train RL agents effectively, the system first generates **synthetic race simulations** based on telemetry data collected from the **Le Mans Ultimate** racing simulator, allowing for controlled and repeatable experiments.

## 📋 Project Overview
Modern motorsport strategy relies heavily on data-driven decision making.  
The goal of this project is to:
1. **Process telemetry data from the Le Mans Ultimate simulator** (lap times, tire wear, fuel consumption, weather, collision impacts, etc.).
2. **Train neural network models** to reproduce realistic race dynamics and generate synthetic race sequences.
3. Use these simulations as an **environment for RL algorithms**, enabling agents to learn optimal race strategies without the cost of real-world or in-game testing.

## ⚙️ Current Features
- Data ingestion from Le Mans Ultimate telemetry logs.
- Preprocessing and scaling of continuous race data.
- **Neural network model (MLP)** for predicting next race states and generating synthetic race runs.

## 🚧 Next Steps
- Extend the simulator with probabilistic events (e.g., car damage from collisions).
- Integrate a **Reinforcement Learning environment** to allow agents to optimize strategies (pit stops, tire/fuel management).
- Experiment with more advanced architectures (e.g., **LSTM/RNN** for sequential dependencies).

## 🛠️ Technologies
- **Python 3**, **PyTorch**, **Scikit-learn** – neural network training and inference  
- **NumPy, SQlite** – data handling and preprocessing  
- **Matplotlib** – visualization 


## 💡 Why This Project?
This simulator provides a **safe, data-driven playground** for experimenting with race strategy optimization.  
By combining **machine learning** with **reinforcement learning**, it creates an adaptable environment for testing strategies that would be costly or time-consuming to explore even in a racing simulator.
