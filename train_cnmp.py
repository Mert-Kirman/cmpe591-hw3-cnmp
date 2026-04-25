import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from homework4 import Hw5Env, bezier, CNMP

def collect_demonstrations(num_trajectories=360):
    """Generates the dataset using the physics environment."""
    env = Hw5Env(render_mode="offscreen")
    
    all_X = []
    all_C = [] # Condition (Height)
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
            
        states = np.stack(states) 
        
        # Time dimension (t)
        t = np.linspace(0, 1, 100).reshape(-1, 1)
        
        # Condition (h)
        h = states[:, 4:5]
        
        # Targets (e_y, e_z, o_y, o_z)
        Y = states[:, 0:4]
        
        all_X.append(t)
        all_C.append(h)
        all_Y.append(Y)

    return torch.FloatTensor(np.array(all_X)), torch.FloatTensor(np.array(all_C)), torch.FloatTensor(np.array(all_Y))

def load_data(data_path="cnmp_dataset.pt"):
    if os.path.exists(data_path):
        print(f"Loading dataset from {data_path}...")
        dataset = torch.load(data_path)
        X_data, C_data, Y_data = dataset['X'], dataset['C'], dataset['Y']
    else:
        X_data, C_data, Y_data = collect_demonstrations(360)
        torch.save({'X': X_data, 'C': C_data, 'Y': Y_data}, data_path)

    return X_data, C_data, Y_data

def train(model, optimizer, scheduler, X_train, C_train, Y_train, X_val, C_val, Y_val, epochs, batch_size, max_obs_per_traj=10):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print("\nTraining CNMP...")
    for epoch in tqdm(range(epochs)):
        model.train()
        
        # Sample a random batch of training trajectories
        batch_idx = np.random.choice(len(X_train), batch_size, replace=False)
        batch_X = X_train[batch_idx]
        batch_C = C_train[batch_idx]
        batch_Y = Y_train[batch_idx]
        
        n_context = np.random.randint(1, max_obs_per_traj + 1)
        context_idx = np.random.choice(100, n_context, replace=False)
        
        # Observation = [X_context, Y_context] (No Condition in encoder)
        obs_X = batch_X[:, context_idx, :]
        obs_Y = batch_Y[:, context_idx, :]
        observation = torch.cat([obs_X, obs_Y], dim=-1)
        
        # Calculate Training Loss
        optimizer.zero_grad()
        loss = model.nll_loss(observation, target=batch_X, condition=batch_C, target_truth=batch_Y)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())

        # --- Validation Phase ---
        model.eval()
        with torch.no_grad():
            val_batch_idx = np.random.choice(len(X_val), min(batch_size, len(X_val)), replace=False)
            v_batch_X = X_val[val_batch_idx]
            v_batch_C = C_val[val_batch_idx]
            v_batch_Y = Y_val[val_batch_idx]
            
            v_n_context = np.random.randint(1, max_obs_per_traj + 1)
            v_context_idx = np.random.choice(100, v_n_context, replace=False)
            
            v_obs_X = v_batch_X[:, v_context_idx, :]
            v_obs_Y = v_batch_Y[:, v_context_idx, :]
            v_observation = torch.cat([v_obs_X, v_obs_Y], dim=-1)
            
            val_loss = model.nll_loss(v_observation, target=v_batch_X, condition=v_batch_C, target_truth=v_batch_Y)
            val_losses.append(val_loss.item())

            # Save best model
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                torch.save(model.state_dict(), "assets/best_cnmp_model.pth")

        if (epoch + 1) % 200 == 0:
            tqdm.write(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

        scheduler.step()

    print(f"Training Complete. Best Validation Loss: {best_val_loss:.4f} saved to assets/best_cnmp_model.pth")

    # Plot Training vs Validation Loss
    plt.figure(figsize=(10, 6))
    # Add smoothing to visualize trends better
    smooth_train = np.convolve(train_losses, np.ones(50)/50, mode='valid')
    smooth_val = np.convolve(val_losses, np.ones(50)/50, mode='valid')
    
    plt.plot(smooth_train, color='blue', alpha=0.8, label='Train Loss (50-ep MA)')
    plt.plot(smooth_val, color='orange', alpha=0.8, label='Validation Loss (50-ep MA)')
    plt.title("CNMP Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log-Likelihood")
    plt.legend()
    plt.grid(True)
    plt.savefig("assets/cnmp_training_loss.png")
    print("Saved training/val loss curve to assets/cnmp_training_loss.png")

def test(model, X_test, C_test, Y_test, max_obs_per_traj=10):
    print("\nRunning 100 Random Evaluation Tests on Best Model...")
    # Load the best weights
    model.load_state_dict(torch.load("assets/best_cnmp_model.pth"))
    model.eval()
    
    mse_ee_list = []
    mse_obj_list = []

    for _ in range(100):
        test_idx = np.random.randint(0, len(X_test))
        t_X = X_test[test_idx:test_idx+1]
        t_C = C_test[test_idx:test_idx+1]
        t_Y = Y_test[test_idx:test_idx+1]
        
        n_context = np.random.randint(1, max_obs_per_traj + 1)
        n_target = np.random.randint(1, 100)
        
        context_idx = np.random.choice(100, n_context, replace=False)
        target_idx = np.random.choice(100, n_target, replace=False)
        
        obs_X = t_X[:, context_idx, :]
        obs_Y = t_Y[:, context_idx, :]
        observation = torch.cat([obs_X, obs_Y], dim=-1)
        
        target = t_X[:, target_idx, :]
        condition = t_C[:, target_idx, :]
        truth = t_Y[:, target_idx, :]
        
        with torch.no_grad():
            mean, _ = model(observation, target, condition)
        
        mse_ee = torch.nn.functional.mse_loss(mean[:, :, 0:2], truth[:, :, 0:2]).item()
        mse_obj = torch.nn.functional.mse_loss(mean[:, :, 2:4], truth[:, :, 2:4]).item()
        
        mse_ee_list.append(mse_ee)
        mse_obj_list.append(mse_obj)

    ee_mean, ee_std = np.mean(mse_ee_list), np.std(mse_ee_list)
    obj_mean, obj_std = np.mean(mse_obj_list), np.std(mse_obj_list)

    print(f"\nEnd-Effector MSE: Mean={ee_mean:.4f}, Std={ee_std:.4f}")
    print(f"Object MSE: Mean={obj_mean:.4f}, Std={obj_std:.4f}")

    labels = ['End-Effector Position', 'Object Position']
    means = [ee_mean, obj_mean]
    stds = [ee_std, obj_std]

    plt.figure(figsize=(7, 6))
    bars = plt.bar(labels, means, yerr=stds, capsize=10, color=['#4C72B0', '#DD8452'], alpha=0.8)
    plt.title("Mean Squared Error of CNMP Predictions (100 Tests)")
    plt.ylabel("MSE")
    
    # Add exact values on top of bars, factoring in the standard deviation (std)
    for bar, std in zip(bars, stds):
        yval = bar.get_height()
        # Place text above the error bar + a tiny margin
        plt.text(bar.get_x() + bar.get_width()/2, yval + std + 0.0001, 
                 f'{yval:.5f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig("assets/cnmp_mse_barplot.png")
    print("Saved MSE bar plot to assets/cnmp_mse_barplot.png")

if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    X_data, C_data, Y_data = load_data() 

    # 70-15-15 Train/Val/Test Split
    total_len = len(X_data)
    train_size = int(0.70 * total_len)
    val_size = int(0.15 * total_len)
    
    # Train
    X_train = X_data[:train_size].to(device)
    C_train = C_data[:train_size].to(device)
    Y_train = Y_data[:train_size].to(device)
    
    # Validation
    X_val = X_data[train_size:train_size+val_size].to(device)
    C_val = C_data[train_size:train_size+val_size].to(device)
    Y_val = Y_data[train_size:train_size+val_size].to(device)
    
    # Test
    X_test = X_data[train_size+val_size:].to(device)
    C_test = C_data[train_size+val_size:].to(device)
    Y_test = Y_data[train_size+val_size:].to(device)

    # Initialize CNMP
    model = CNMP(in_shape=(1, 4), condition_dim=1, hidden_size=128, num_hidden_layers=4).to(device)
    
    epochs = 3000
    batch_size = 64
    max_observations_per_trajectory = 10

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # Train the model (pass the scheduler, val sets)
    train(model, optimizer, scheduler, X_train, C_train, Y_train, X_val, C_val, Y_val, epochs, batch_size, max_observations_per_trajectory)

    # Evaluate the model (only on the unseen Test set)
    test(model, X_test, C_test, Y_test, max_observations_per_trajectory)
