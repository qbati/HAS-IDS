import torch
import numpy as np
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import torch
import torch.nn as nn
import math
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score
import scipy.optimize as opt
import torch.distributions as dist
from sklearn.metrics import accuracy_score

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

class SplitData(BaseEstimator, TransformerMixin):
    def __init__(self, dataset):
        super(SplitData, self).__init__()
        self.dataset = dataset

    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, labels, one_hot_label=True):
        if self.dataset == 'nsl':
            # Preparing the labels
            y = X[labels]
            X_ = X.drop(['labels5', 'labels2'], axis=1)
            # abnormal data is labeled as 1, normal data 0
            y = (y != 'normal')
            y_ = np.asarray(y).astype('float32')

        elif self.dataset == 'unsw':
            # UNSW dataset processing
            y_ = X[labels]
            X_ = X.drop('label', axis=1)

        else:
            raise ValueError("Unsupported dataset type")

        # Normalization
        # This fits on the whole dataset (X_) being passed.
        # This is the paper's original methodology.
        normalize = MinMaxScaler().fit(X_)
        x_ = normalize.transform(X_)

        return x_, y_

def description(data):
    print("Number of samples(examples) ",data.shape[0]," Number of features",data.shape[1])
    print("Dimension of data set ",data.shape)

class AE(nn.Module):
    def __init__(self, input_dim):
        super(AE, self).__init__()

        # Find the nearest power of 2 to input_dim
        # Note: for input_dim=196, nearest_power_of_2 = 128
        # This is a small bug in their logic (2**round(log2(196)) = 128, not 256)
        # But we must follow it.
        # 128 // 2 = 64
        # 128 // 4 = 32
        # This gives [196, 64, 32] -> [32, 64, 196]
        # This *differs* from the paper text [196, 128, 64], but we follow the code.
        nearest_power_of_2 = 2 ** round(math.log2(input_dim))

        # Calculate the dimensions of the 2nd/4th layer and the 3rd layer.
        second_fourth_layer_size = nearest_power_of_2 // 2  # A half
        third_layer_size = nearest_power_of_2 // 4      # A quarter

        # Create encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, third_layer_size),
        )

        # Create decoder
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(third_layer_size, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, input_dim),
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

class CRCLoss(nn.Module):
    def __init__(self, device, temperature=0.1, scale_by_temperature=True):
        super(CRCLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):        
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float()
        # compute logits
        logits = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # Calculate the dot product similarity between pairwise samples
        # create mask 
        logits_mask = torch.ones_like(mask).to(self.device) - torch.eye(batch_size).to(self.device)  
        logits_without_ii = logits * logits_mask
        
        logits_normal = logits_without_ii[(labels == 0).squeeze()]
        # Handle edge case: no normal samples in batch
        if logits_normal.shape[0] == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        logits_normal_normal = logits_normal[:,(labels == 0).squeeze()]
        logits_normal_abnormal = logits_normal[:,(labels > 0).squeeze()]
        
        # Handle edge case: no abnormal samples in batch
        if logits_normal_abnormal.shape[1] == 0:
             return torch.tensor(0.0, device=self.device, requires_grad=True)
             
        # Handle edge case: only one normal sample in batch
        if logits_normal_normal.shape[1] == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        ## This is the denominator for our proposed CRC loss: TWO times of traversal
        sum_of_vium = torch.sum(torch.exp(logits_normal_abnormal))
        denominator = torch.exp(logits_normal_normal) + sum_of_vium
        
        # Add 1e-9 to avoid log(0)
        log_probs = logits_normal_normal - torch.log(denominator + 1e-9)
 
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        
        loss = loss.mean()
        # Handle potential NaNs
        if torch.isnan(loss):
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return loss
    
def score_detail(y_test,y_test_pred,if_print=False):
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    if if_print == True:
        print("Confusion matrix")
        print(cm)
        # Accuracy 
        print('Accuracy ',accuracy_score(y_test, y_test_pred))
        # Precision 
        print('Precision ',precision_score(y_test, y_test_pred, zero_division=0))
        # Recall
        print('Recall ',recall_score(y_test, y_test_pred, zero_division=0))
        # F1 score
        print('F1 score ',f1_score(y_test,y_test_pred, zero_division=0))

    return (
        accuracy_score(y_test, y_test_pred), 
        precision_score(y_test, y_test_pred, zero_division=0), 
        recall_score(y_test, y_test_pred, zero_division=0), 
        f1_score(y_test,y_test_pred, zero_division=0),
        cm
    )

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def gaussian_pdf(x, mu, sigma):
    sigma = max(sigma, 1e-9) # Avoid division by zero
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def log_likelihood(params, data):
    mu1, sigma1, mu2, sigma2 = params
    pdf1 = gaussian_pdf(data, mu1, sigma1)
    pdf2 = gaussian_pdf(data, mu2, sigma2)
    # Add 1e-9 to avoid log(0)
    return -np.sum(np.log(0.5 * pdf1 + 0.5 * pdf2 + 1e-9))

# *** MODIFIED EVALUATE FUNCTION ***
def evaluate(normal_temp, normal_recon_temp, x_train, y_train, x_test, y_test, model):
    num_of_layer = 0
    
    # Detach tensors from graph to save memory
    x_train = x_train.detach()
    y_train = y_train.detach()
    x_test = x_test.detach()

    x_train_normal = x_train[(y_train == 0).squeeze()]
    x_train_abnormal = x_train[(y_train == 1).squeeze()]
    
    with torch.no_grad():
        train_features = F.normalize(model(x_train)[num_of_layer], p=2, dim=1)
        train_features_normal = F.normalize(model(x_train_normal)[num_of_layer], p=2, dim=1)
        train_features_abnormal = F.normalize(model(x_train_abnormal)[num_of_layer], p=2, dim=1)
        test_features = F.normalize(model(x_test)[num_of_layer], p=2, dim=1)

        values_features_all = F.cosine_similarity(train_features, normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1)
        values_features_normal = F.cosine_similarity(train_features_normal, normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1)
        values_features_abnormal = F.cosine_similarity(train_features_abnormal, normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1)
        values_features_test = F.cosine_similarity(test_features, normal_temp.reshape([-1, normal_temp.shape[0]]))

        num_of_output = 1
        train_recon = F.normalize(model(x_train)[num_of_output], p=2, dim=1)
        train_recon_normal = F.normalize(model(x_train_normal)[num_of_output], p=2, dim=1)
        train_recon_abnormal = F.normalize(model(x_train_abnormal)[num_of_output], p=2, dim=1)
        test_recon = F.normalize(model(x_test)[num_of_output], p=2, dim=1)

        values_recon_all = F.cosine_similarity(train_recon, normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1)
        values_recon_normal = F.cosine_similarity(train_recon_normal, normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1)
        values_recon_abnormal = F.cosine_similarity(train_recon_abnormal, normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1)
        values_recon_test = F.cosine_similarity(test_recon, normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1)

    values_features_all_np = values_features_all.cpu().numpy()
    values_features_normal_np = values_features_normal.cpu().numpy()
    values_features_abnormal_np = values_features_abnormal.cpu().numpy()
    
    values_recon_all_np = values_recon_all.cpu().numpy()
    values_recon_normal_np = values_recon_normal.cpu().numpy()
    values_recon_abnormal_np = values_recon_abnormal.cpu().numpy()

    mu1_initial = np.mean(values_features_normal_np) if len(values_features_normal_np) > 0 else 0.9
    sigma1_initial = np.std(values_features_normal_np) if len(values_features_normal_np) > 0 else 0.1
    mu2_initial = np.mean(values_features_abnormal_np) if len(values_features_abnormal_np) > 0 else 0.1
    sigma2_initial = np.std(values_features_abnormal_np) if len(values_features_abnormal_np) > 0 else 0.1

    initial_params = np.array([mu1_initial, sigma1_initial, mu2_initial, sigma2_initial])
    result = opt.minimize(log_likelihood, initial_params, args=(values_features_all_np,), method='Nelder-Mead')
    mu1_fit, sigma1_fit, mu2_fit, sigma2_fit = result.x

    if mu1_fit > mu2_fit:
        gaussian1 = dist.Normal(mu1_fit, max(abs(sigma1_fit), 1e-9))
        gaussian2 = dist.Normal(mu2_fit, max(abs(sigma2_fit), 1e-9))
    else:
        gaussian2 = dist.Normal(mu1_fit, max(abs(sigma1_fit), 1e-9))
        gaussian1 = dist.Normal(mu2_fit, max(abs(sigma2_fit), 1e-9))

    pdf1 = gaussian1.log_prob(values_features_test).exp()
    pdf2 = gaussian2.log_prob(values_features_test).exp()
    
    y_test_pred_enc = (pdf2 > pdf1).cpu().numpy().astype("int32")
    y_test_score_enc = (pdf2 / (pdf1 + pdf2 + 1e-9)).cpu().numpy().astype("float32")

    mu3_initial = np.mean(values_recon_normal_np) if len(values_recon_normal_np) > 0 else 0.9
    sigma3_initial = np.std(values_recon_normal_np) if len(values_recon_normal_np) > 0 else 0.1
    mu4_initial = np.mean(values_recon_abnormal_np) if len(values_recon_abnormal_np) > 0 else 0.1
    sigma4_initial = np.std(values_recon_abnormal_np) if len(values_recon_abnormal_np) > 0 else 0.1

    initial_params = np.array([mu3_initial, sigma3_initial, mu4_initial, sigma4_initial])
    result = opt.minimize(log_likelihood, initial_params, args=(values_recon_all_np,), method='Nelder-Mead')
    mu3_fit, sigma3_fit, mu4_fit, sigma4_fit = result.x

    if mu3_fit > mu4_fit:
        gaussian3 = dist.Normal(mu3_fit, max(abs(sigma3_fit), 1e-9))
        gaussian4 = dist.Normal(mu4_fit, max(abs(sigma4_fit), 1e-9))
    else:
        gaussian4 = dist.Normal(mu3_fit, max(abs(sigma3_fit), 1e-9))
        gaussian3 = dist.Normal(mu4_fit, max(abs(sigma4_fit), 1e-9))

    pdf3 = gaussian3.log_prob(values_recon_test).exp()
    pdf4 = gaussian4.log_prob(values_recon_test).exp()
    y_test_pred_dec = (pdf4 > pdf3).cpu().numpy().astype("int32")
    y_test_score_dec = (pdf4 / (pdf3 + pdf4 + 1e-9)).cpu().numpy().astype("float32")

    # Original confidence calculation
    y_test_pro_en = np.abs(y_test_score_enc - 0.5) * 2
    y_test_pro_de = np.abs(y_test_score_dec - 0.5) * 2

    y_pred_final = np.where(y_test_pro_en > y_test_pro_de, y_test_pred_enc, y_test_pred_dec)
    y_score_final = np.where(y_test_pro_en > y_test_pro_de, y_test_score_enc, y_test_score_dec)
    
    if not isinstance(y_test, int):
        # This is the final evaluation
        y_test_np = y_test.cpu().numpy()
        result_final = score_detail(y_test_np, y_pred_final, if_print=True)
        # Return what's needed for the bake-off
        return y_pred_final, y_score_final, result_final
    else:
        # This is for pseudo-label generation
        return y_pred_final