import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from homework4 import Hw5Env, bezier, CNMP

def collect_demonstrations(num_trajectories=500):
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
        X_data, C_data, Y_data = collect_demonstrations()
        torch.save({'X': X_data, 'C': C_data, 'Y': Y_data}, data_path)

    return X_data, C_data, Y_data

def train(model, optimizer, scheduler, X_train, C_train, Y_train, X_val, C_val, Y_val, epochs, batch_size, max_obs_per_traj=10, max_target_per_traj=10):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print("\nTraining CNMP...")
    for epoch in tqdm(range(epochs)):
        model.train()
        
        # --- Training Phase ---
        # Sample a random batch of training trajectories
        batch_idx = np.random.choice(len(X_train), batch_size, replace=False)
        
        n_context = np.random.randint(1, max_obs_per_traj + 1)
        n_target = np.random.randint(1, max_target_per_traj + 1)

        # Initialize empty tensors to hold the batch
        obs_X = torch.empty((batch_size, n_context, X_train.shape[-1]), device=device)
        obs_Y = torch.empty((batch_size, n_context, Y_train.shape[-1]), device=device)
        target = torch.empty((batch_size, n_target, X_train.shape[-1]), device=device)
        batch_C = torch.empty((batch_size, n_target, C_train.shape[-1]), device=device)
        target_truth = torch.empty((batch_size, n_target, Y_train.shape[-1]), device=device)

        # Generate UNIQUE time indices for each trajectory in the batch
        for i, idx in enumerate(batch_idx):
            c_idx = np.random.choice(100, n_context, replace=False) # context indices
            t_idx = np.random.choice(100, n_target, replace=False) # target indices
            
            obs_X[i] = X_train[idx, c_idx, :]
            obs_Y[i] = Y_train[idx, c_idx, :]
            target[i] = X_train[idx, t_idx, :]
            batch_C[i] = C_train[idx, t_idx, :]
            target_truth[i] = Y_train[idx, t_idx, :]
            
        observation = torch.cat([obs_X, obs_Y], dim=-1)
        
        optimizer.zero_grad()
        loss = model.nll_loss(observation, target=target, condition=batch_C, target_truth=target_truth)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())

        # --- Validation Phase ---
        model.eval()
        with torch.no_grad():
            val_batch_idx = np.random.choice(len(X_val), min(batch_size, len(X_val)), replace=False)
            
            v_n_context = np.random.randint(1, max_obs_per_traj + 1)
            v_n_target = np.random.randint(1, max_target_per_traj + 1)
            
            v_obs_X = torch.empty((len(val_batch_idx), v_n_context, X_val.shape[-1]), device=device)
            v_obs_Y = torch.empty((len(val_batch_idx), v_n_context, Y_val.shape[-1]), device=device)
            v_target = torch.empty((len(val_batch_idx), v_n_target, X_val.shape[-1]), device=device)
            v_batch_C = torch.empty((len(val_batch_idx), v_n_target, C_val.shape[-1]), device=device)
            v_target_truth = torch.empty((len(val_batch_idx), v_n_target, Y_val.shape[-1]), device=device)

            # Generate UNIQUE time indices for each trajectory in the batch
            for i, idx in enumerate(val_batch_idx):
                v_c_idx = np.random.choice(100, v_n_context, replace=False)
                v_t_idx = np.random.choice(100, v_n_target, replace=False)

                v_obs_X[i] = X_val[idx, v_c_idx, :]
                v_obs_Y[i] = Y_val[idx, v_c_idx, :]
                v_target[i] = X_val[idx, v_t_idx, :]
                v_batch_C[i] = C_val[idx, v_t_idx, :]
                v_target_truth[i] = Y_val[idx, v_t_idx, :]

            v_observation = torch.cat([v_obs_X, v_obs_Y], dim=-1)
            
            val_loss = model.nll_loss(v_observation, target=v_target, condition=v_batch_C, target_truth=v_target_truth)
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

def test(model, X_test, C_test, Y_test, max_obs_per_traj=10, max_target_per_traj=10):
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
        n_target = np.random.randint(1, max_target_per_traj + 1)
        
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

    plt.tight_layout()
    plt.savefig("assets/cnmp_mse_barplot.png")
    print("Saved MSE bar plot to assets/cnmp_mse_barplot.png")

if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    device = torch.device("cpu")
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
    
    epochs = 10000
    batch_size = 64
    max_observations_per_trajectory = 10
    max_target_per_trajectory = 10

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # Train the model (pass the scheduler, val sets)
    train(model, optimizer, scheduler, X_train, C_train, Y_train, X_val, C_val, Y_val, epochs, batch_size, max_observations_per_trajectory, max_target_per_trajectory)

    # Evaluate the model (only on the unseen Test set)
    test(model, X_test, C_test, Y_test, max_observations_per_trajectory, max_target_per_trajectory)
