import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from homework4 import Hw5Env, bezier, CNP

def collect_demonstrations(num_trajectories=120):
    """Generates the dataset using the physics environment."""
    env = Hw5Env(render_mode="offscreen")
    
    all_X = []
    all_Y = []
    
    print(f"Generating {num_trajectories} demonstrations...")
    for i in tqdm(range(num_trajectories)):
        env.reset()
        
        # Generate random Bezier curve path
        p_1 = np.array([0.5, 0.3, 1.04])
        p_2 = np.array([0.5, 0.15, np.random.uniform(1.04, 1.4)])
        p_3 = np.array([0.5, -0.15, np.random.uniform(1.04, 1.4)])
        p_4 = np.array([0.5, -0.3, 1.04])
        points = np.stack([p_1, p_2, p_3, p_4], axis=0)
        curve = bezier(points, steps=100) # 100 timesteps

        env._set_ee_in_cartesian(curve[0], rotation=[-90, 0, 180], n_splits=100, max_iters=100, threshold=0.05)
        
        states = []
        for p in curve:
            env._set_ee_pose(p, rotation=[-90, 0, 180], max_iters=10)
            states.append(env.high_level_state())
            
        states = np.stack(states) # Shape: (100, 5) -> [e_y, e_z, o_y, o_z, h]
        
        # Create Time dimension (t) normalized from 0 to 1
        t = np.linspace(0, 1, 100).reshape(-1, 1)
        
        # Extract height (h), which is constant for the whole trajectory
        h = states[:, 4:5]
        
        # X = [t, h]
        X = np.concatenate([t, h], axis=1)
        
        # Y = [e_y, e_z, o_y, o_z]
        Y = states[:, 0:4]
        
        all_X.append(X)
        all_Y.append(Y)

    return torch.FloatTensor(np.array(all_X)), torch.FloatTensor(np.array(all_Y))

def load_data(data_path = "cnmp_dataset.pt"):
    if os.path.exists(data_path):
        print(f"Loading dataset from {data_path}...")
        dataset = torch.load(data_path)
        X_data, Y_data = dataset['X'], dataset['Y']
    else:
        X_data, Y_data = collect_demonstrations(300)
        torch.save({'X': X_data, 'Y': Y_data}, data_path)

    return X_data, Y_data

def train(model, optimizer, X_train, Y_train, epochs, batch_size, max_observations_per_trajectory=10):
    train_losses = []

    print("\nTraining CNMP...")
    for epoch in tqdm(range(epochs)):
        # Sample a random batch of trajectories
        batch_idx = np.random.choice(len(X_train), batch_size, replace=False)
        batch_X = X_train[batch_idx]
        batch_Y = Y_train[batch_idx]
        
        # Sample random number of context points
        n_context = np.random.randint(1, max_observations_per_trajectory + 1)
        context_idx = np.random.choice(100, n_context, replace=False)
        
        # Observation = [X_context, Y_context] concatenated
        obs_X = batch_X[:, context_idx, :]
        obs_Y = batch_Y[:, context_idx, :]
        observation = torch.cat([obs_X, obs_Y], dim=-1)
        
        # Targets are all points in the trajectory (to learn the whole curve)
        target = batch_X
        target_truth = batch_Y
        
        # Calculate NLL Loss
        optimizer.zero_grad()
        loss = model.nll_loss(observation, target, target_truth)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())

        if (epoch + 1) % 200 == 0:
            tqdm.write(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "assets/cnmp_model.pth")
    print("Saved trained model to assets/cnmp_model.pth")

    # Plot and Save Training Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, color='blue', alpha=0.6)
    plt.title("CNMP Training Loss (Negative Log-Likelihood)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("assets/cnmp_training_loss.png")
    print("Saved training loss curve to assets/cnmp_training_loss.png")

def test(model, X_test, Y_test, max_observations_per_trajectory=10):
    print("\nRunning 100 Random Evaluation Tests...")
    mse_ee_list = []
    mse_obj_list = []

    model.eval()
    for _ in range(100):
        # Pick 1 random trajectory from the test set
        test_idx = np.random.randint(0, len(X_test))
        t_X = X_test[test_idx:test_idx+1]
        t_Y = Y_test[test_idx:test_idx+1]
        
        # Random context and query counts
        n_context = np.random.randint(1, max_observations_per_trajectory + 1)
        n_target = np.random.randint(1, 100)
        
        context_idx = np.random.choice(100, n_context, replace=False)
        target_idx = np.random.choice(100, n_target, replace=False)
        
        # Build observation tensor
        obs_X = t_X[:, context_idx, :]
        obs_Y = t_Y[:, context_idx, :]
        observation = torch.cat([obs_X, obs_Y], dim=-1)
        
        # Build target tensors
        target = t_X[:, target_idx, :]
        truth = t_Y[:, target_idx, :]
        
        # Predict
        with torch.no_grad():
            mean, _ = model(observation, target)
        
        # Compute MSE for End-Effector (indices 0, 1) and Object (indices 2, 3)
        mse_ee = torch.nn.functional.mse_loss(mean[:, :, 0:2], truth[:, :, 0:2]).item()
        mse_obj = torch.nn.functional.mse_loss(mean[:, :, 2:4], truth[:, :, 2:4]).item()
        
        mse_ee_list.append(mse_ee)
        mse_obj_list.append(mse_obj)

    # Plot Bar Plot with Mean and Std
    ee_mean, ee_std = np.mean(mse_ee_list), np.std(mse_ee_list)
    obj_mean, obj_std = np.mean(mse_obj_list), np.std(mse_obj_list)

    labels = ['End-Effector Position', 'Object Position']
    means = [ee_mean, obj_mean]
    stds = [ee_std, obj_std]

    print(f"\nEnd-Effector MSE: Mean={ee_mean:.4f}, Std={ee_std:.4f}")
    print(f"Object MSE: Mean={obj_mean:.4f}, Std={obj_std:.4f}")

    plt.figure(figsize=(7, 6))
    bars = plt.bar(labels, means, yerr=stds, capsize=10, color=['#4C72B0', '#DD8452'], alpha=0.8)
    plt.title("Mean Squared Error of CNMP Predictions (100 Tests)")
    plt.ylabel("MSE")
    
    # Add exact values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig("assets/cnmp_mse_barplot.png")
    print("Saved MSE bar plot to assets/cnmp_mse_barplot.png")

if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get Data
    X_data, Y_data = load_data() # Shapes: (num_trajectories, 100, 2), (num_trajectories, 100, 4)

    # Split: 80% Train, 20% Test
    train_ratio = 0.8
    train_size = int(train_ratio * len(X_data))
    X_train, Y_train = X_data[:train_size].to(device), Y_data[:train_size].to(device)
    X_test, Y_test = X_data[train_size:].to(device), Y_data[train_size:].to(device)

    # Initialize CNP
    # in_shape = (d_x, d_y) -> (2, 4)
    model = CNP(in_shape=(2, 4), hidden_size=128, num_hidden_layers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    epochs = 3000
    batch_size = 64

    max_observations_per_trajectory = 10
    
    # Train the model
    train(model, optimizer, X_train, Y_train, epochs, batch_size, max_observations_per_trajectory)

    # Evaluation (100 Tests)
    test(model, X_test, Y_test, max_observations_per_trajectory)
