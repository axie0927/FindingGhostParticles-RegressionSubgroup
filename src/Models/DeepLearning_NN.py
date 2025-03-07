import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.lines as mlines
import torch.nn.functional as F

# Select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==========================================
# Data Loading and Preprocessing
# ==========================================
def load_and_preprocess_data():
    # create varaibles that holds a dataframe
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, "Data/")
    data_files = [f for f in os.listdir(str(data_dir)) if f.endswith('csv')]    
    data_train_name = [f for f in data_files if 'TRAIN' in f]
    data_test_name = [f for f in data_files if 'TEST' in f]
    data_NPML_name = [f for f in data_files if 'NPML' in f]
    data_NPML_cut_name = [f for f in data_files if 'npml_cut' in f] 
    train_df = pd.read_csv(os.path.join(data_dir,data_train_name[0]))
    test_df = pd.read_csv(os.path.join(data_dir,data_test_name[0]))
    NPML_df = pd.read_csv(os.path.join(data_dir,data_NPML_name[0])) 
    NPML_cut_id = np.array(pd.read_csv(os.path.join(data_dir,data_NPML_cut_name[0]))['id'])
    NPML_cut_df = NPML_df[NPML_df['id'].isin(NPML_cut_id)]  
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    NPML_df = NPML_df.dropna()
    NPML_cut_df = NPML_cut_df.dropna()  
    def manipulate_cols(df):
        boolean_col = ['highavse','lowavse','truedcr','lq']
        useless_col = ['tdrift50','tdrift10']
        new_df = df.drop(columns=boolean_col+['id']+useless_col)
        new_df.columns = [col.strip().replace(' ','_') for col in new_df.columns]
        return new_df   
    train_df = manipulate_cols(train_df)
    test_df = manipulate_cols(test_df)
    NPML_df = NPML_df.drop(columns=['tdrift50','tdrift10','id'])
    NPML_cut_df = NPML_cut_df.drop(columns=['tdrift50','tdrift10','id'])    
    return train_df,test_df, NPML_df, NPML_cut_df

def Splitting_and_Standardization(train_df,test_df,NPML_df,NPML_cut_df):
    X_train = train_df.drop(columns=['energylabel']).values
    X_test = test_df.drop(columns=['energylabel']).values
    y_train = train_df['energylabel'].values.reshape(-1,1)
    y_test = test_df['energylabel'].values.reshape(-1,1)
    X_NPML = NPML_df.values
    X_NPML_cut = NPML_cut_df.values


    # Standardization

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    X_NPML = scaler_X.transform(X_NPML)
    X_NPML_cut = scaler_X.transform(X_NPML_cut)

    scaler_y = StandardScaler() # VERY IMPORTANT! We also need to transform it back to original after prediction!
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    return X_train, X_test, X_NPML,X_NPML_cut, y_train, y_test, scaler_X, scaler_y


# ==========================================
#  PyTorch Dataset & DataLoader
# ==========================================
def convert_to_Pytorch(X_train, X_test, y_train, y_test, X_NPML, X_NPML_cut):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32) 
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)   
    X_NPML_tensor = torch.tensor(X_NPML, dtype=torch.float32)
    X_NPML_cut_tensor = torch.tensor(X_NPML_cut, dtype=torch.float32)   
    return X_train_tensor, y_train_tensor,X_test_tensor, y_test_tensor, X_NPML_tensor, X_NPML_cut_tensor

# ==========================================
#  Neural Network Model
# ==========================================
class NPDL(Dataset): # Neutrino Physics Deep Learning
    def __init__(self,X,y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]

class SuperPredictor(nn.Module):
    def __init__(self, input_size):
        super(SuperPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)  
        self.out = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)  
        self.dropout2 = nn.Dropout(0.4)  


    def forward(self, x):
        x = self.relu(self.fc1(x))
        #x = self.dropout1(x)  # Apply dropout after activation
        x = self.relu(self.fc2(x))
        #x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.out(x)
        return x



# ==========================================
#  Model Evaluation and application
# ==========================================
def evaluate_on_testset(model, X_test_tensor, y_test_tensor, scaler_y, criterion):
    model.eval()
    with torch.no_grad():
        X_test_device = X_test_tensor.to(device)
        y_test_device = y_test_tensor.to(device)
        predictions = model(X_test_device)  
        final_test_loss = criterion(predictions, y_test_device).item()  
    predictions_np = predictions.cpu().numpy()  
    y_test_np      = y_test_device.cpu().numpy()    
    predictions_original = scaler_y.inverse_transform(predictions_np)
    y_test_original      = scaler_y.inverse_transform(y_test_np)    
    MSE = mean_squared_error(y_test_original, predictions_original)
    MAE = mean_absolute_error(y_test_original, predictions_original)
    r2  = r2_score(y_test_original, predictions_original)   
    print('------------------------------------------------------------------------------------------')
    print(f"Final Results on Test Data:")
    print(f"MSE: {MSE:.4f}")
    print(f"MAE: {MAE:.4f}")
    print(f"R^2: {r2:.4f}")
    print('------------------------------------------------------------------------------------------')
    return predictions_original,  y_test_original


def apply_on_NPML(X_NPML_tensor, X_NPML_cut_tensor, model, scaler_y):
    model.eval()
    with torch.no_grad():
        X_NPML_device = X_NPML_tensor.to(device)
        X_NPML_cut_device = X_NPML_cut_tensor.to(device)
        predictions_NPML = model(X_NPML_device)
        predictions_NPML_cut = model(X_NPML_cut_device)
    predictions_NPML_np = predictions_NPML.cpu().numpy()  
    predictions_NPML_cut_np = predictions_NPML_cut.cpu().numpy() 
    predictions_NPML_original = scaler_y.inverse_transform(predictions_NPML_np)
    predictions_NPML_cut_original = scaler_y.inverse_transform(predictions_NPML_cut_np)
    np.save("predictions_NPML_original.npy", predictions_NPML_original)
    np.save("predictions_NPML_cut_original.npy", predictions_NPML_cut_original)

    return predictions_NPML_original, predictions_NPML_cut_original

# ==========================================
#  Visualization 
# ==========================================

def actual_vs_predicted(y_test_original, predictions_original, residuals):
    sns.set_style("whitegrid")   
    sns.set_context("talk")      
    plt.figure(figsize=(8, 8))

    scatter = plt.scatter(
        x=y_test_original.flatten(),
        y=predictions_original.flatten(),
        c=residuals,              
        cmap="coolwarm",          
        alpha=0.7,                
        edgecolors="black",
        s=80                      
    )

    cbar = plt.colorbar(scatter, pad=0.01)
    cbar.set_label("Residual (Predicted - Actual)")

    # 45-degree reference line for perfect predictions
    min_val = min(y_test_original.min(), predictions_original.min())
    max_val = max(y_test_original.max(), predictions_original.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs. Actual")

    plt.tight_layout()
    plt.savefig("NN_Predicted_vs_Actual.png", dpi=300, bbox_inches="tight")
    plt.show()



def residual_distribution_plot(predictions_original, y_test_original, residuals):
    sns.set(style="whitegrid", context="talk")
    residuals = predictions_original.flatten() - y_test_original.flatten()
    # mean and std of residuals
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)

    sns.set(style="whitegrid", context="talk")

    plt.figure(figsize=(8, 6))

    sns.histplot(residuals, kde=True, label="Residual Distr.")

    dashed_line = mlines.Line2D(
        [], [], 
        color="#4C72B0",  # match Seaborn's default blue
        linestyle="-",
        linewidth=2,
        label="KDE Dashed"
    )

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(dashed_line)
    labels.append("KDE")

    plt.legend(handles, labels, loc="upper right")

    plt.axvline(0, color='red', linestyle='--', linewidth=2)
    plt.xlim(-100, 100)
    plt.xlabel("Residuals")
    plt.title("Distribution of Residuals")
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    textstr = (f"Mean Residual: {residual_mean:.2f}\n"
               f"Std Dev: {residual_std:.2f}")
    plt.text(0.98, 0.75, textstr, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', horizontalalignment='right',
             bbox=props)

    plt.tight_layout()
    plt.savefig("NN_Distribution_Residual.png", dpi=300, bbox_inches="tight")
    plt.show()

def visualize_energy_spectrum(predictions_NPML_original, predictions_NPML_cut_original):
    #Visualize final energy spectrum
    energy_full = predictions_NPML_original  # Full spectrum
    energy_cut = predictions_NPML_cut_original  # Cut spectrum
    bins = np.linspace(0, 4000, 300)

    plt.figure(figsize=(10, 4),dpi=300)
    plt.hist(energy_full, bins=bins, histtype="step", color="black", label="Full Spectrum")
    plt.hist(energy_cut, bins=bins, histtype="step", color="magenta", label="Survival Spectrum After Cut")

    plt.yscale("log")
    plt.xlim(0, 4000)
    # Labels and title
    plt.xlabel("Energy [keV]")
    plt.ylabel("Counts (log)")
    plt.title("Energy Spectrum(NPML dataset only)")
    plt.legend()

    plt.savefig("Energy_Spectrum_NPML_only.png", dpi=300, bbox_inches="tight")

# ==========================================
# Main Execution !!
# ==========================================
def main():
    data_dir = os.path.join(os.getcwd(), "Data/")
    
    print("Loading and preprocessing data...")
    train_df,test_df, NPML_df, NPML_cut_df = load_and_preprocess_data()

    print("Reshaping and standardizing data...")
    X_train, X_test, X_NPML, X_NPML_cut, y_train, y_test, scaler_X, scaler_y = (
        Splitting_and_Standardization(train_df, test_df, NPML_df, NPML_cut_df)
    )

    print("Initializing and training model...")

    X_train_tensor, y_train_tensor,X_test_tensor, y_test_tensor, X_NPML_tensor, X_NPML_cut_tensor = (
        convert_to_Pytorch(X_train, X_test, y_train, y_test, X_NPML, X_NPML_cut)
    )
    
    train_NPDL = NPDL(X_train_tensor,y_train_tensor)
    test_NPDL = NPDL(X_test_tensor, y_test_tensor)
    
    batch_size = 32
    train_loader = DataLoader(train_NPDL, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_NPDL, batch_size=batch_size, shuffle=False)

    input_size = X_train.shape[1]
    model = SuperPredictor(input_size).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)

    num_epochs = 20
    best_loss = float('inf')
    patience = 5  # Stop training if no improvement after 5 epochs
    counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
    
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        train_loss = running_loss / len(train_NPDL)

        model.eval()
        with torch.no_grad():
            ######################################################################################
            # Evaluate on test set, only for tuning things like learning rates and batchsizes.    #
            ######################################################################################
            test_predictions = model(X_test_tensor)
            test_loss_val = criterion(test_predictions, y_test_tensor).item()

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Test Loss(For tuning): {test_loss_val:.4f}")
        # Early stopping logic
        if test_loss_val < best_loss:
            best_loss = test_loss_val
            counter = 0  # Reset counter if loss improves
        else:
            counter += 1  # Increment if no improvement
            if counter >= patience:
                print("Early stopping triggered!")
                break  # Stop training if no improvement for 'patience' epochs

    predictions_original, y_test_original = evaluate_on_testset(model, X_test_tensor, y_test_tensor, scaler_y, criterion)
    residuals = predictions_original.flatten() - y_test_original.flatten()

    # Visualization
    print('------------------------------------------------------------------------------------------')
    print('A plot of Actual value vs Predicted Value is being generated... ')
    print('You can also find it in the same directory with Deepleaning_NN.py')
    print('Please close this image to continue.')
    actual_vs_predicted(y_test_original, predictions_original, residuals)
    print('------------------------------------------------------------------------------------------')
    print('A plot of Distribution of Residuals with KDE is being generated... ')
    residual_distribution_plot(predictions_original, y_test_original, residuals)
    print('You can also find it in the same directory with Deepleaning_NN.py')
    print('Please close this image to continue.')
    print('------------------------------------------------------------------------------------------')

    # Apply the model on the final NPML dataset
    print('------------------------------------------------------------------------------------------')
    print('Applying our NN on NPML dataset which is a real world data with unknow true value.')
    predictions_NPML_original, predictions_NPML_cut_original = apply_on_NPML(X_NPML_tensor, X_NPML_cut_tensor, model, scaler_y)
    print('Two .npy files which are our final predictions in the same order with original NPML dataset have been saved in the current directory 😋.')
    print('------------------------------------------------------------------------------------------')
    print('Now generating the final spectrum, hold on 🚀')
    visualize_energy_spectrum(predictions_NPML_original, predictions_NPML_cut_original)
    print('------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------')
    print('CONGRATULATIONS!')
    print('The final energy spectrum has been saved in the current directory! See you next time!')
    print("\U00002764")

if __name__ == "__main__":
    main()


