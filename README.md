# Race Strategy Optimization with Reinforcement Learning

This project focuses on **optimizing race strategies** (pit stops, tire choices, fuel management, etc.) using **Reinforcement Learning (RL)**.  
To train RL agents effectively, the system first generates **synthetic race simulations** based on telemetry data collected from the **Le Mans Ultimate** racing simulator, allowing for controlled and repeatable experiments.

## 📋 Project Overview
Modern motorsport strategy relies heavily on data-driven decision making.  
The goal of this project is to:
1. **Process telemetry data from the Le Mans Ultimate simulator** (lap times, tire wear, fuel consumption, weather, collision impacts, etc.).
2. **Train recurrent neural network model (LSTM)** to reproduce realistic race dynamics and generate synthetic race sequences.
3. Use these simulations as an **environment for RL algorithms**, enabling agents to learn optimal race strategies without the cost of real-world or in-game testing.

<img width="5752" height="1928" alt="image" src="https://github.com/user-attachments/assets/c1684989-c6ab-47b2-b730-7c10d07b0c25" />


## ⚙️ Current Features
- Data ingestion from Le Mans Ultimate telemetry logs.
- Preprocessing and scaling of continuous race data.
- **Long short-term memory(LSTM) model** for predicting next race states and generating synthetic race runs.
- Extend the simulator with probabilistic events (e.g., car damage from collisions).
- Integrated a **Reinforcement Learning env** to allow agent to optimize strategies (pit stops, tire/fuel management).

## 🚧 Next Steps
- Tune hyperparameters and learn RL agent 
- Test/fine-tune RL model directly with LMU

## 🛠️ Technologies
- **Python**, **PyTorch**, **Scikit-learn** – neural network training and inference  
- **NumPy, SQlite** – data handling and preprocessing  
- **Matplotlib** – visualization 


## 💡 Why This Project?
This simulator provides a **safe, data-driven playground** for experimenting with race strategy optimization.  
By combining **machine learning** with **reinforcement learning**, it creates an adaptable environment for testing strategies that would be costly or time-consuming to explore even in a racing simulator.
