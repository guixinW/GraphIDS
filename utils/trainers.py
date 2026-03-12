import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataloaders import SequentialDataset, collate_fn


class CLLoss(ABC):
    """Abstract class to define losses in the CL framework that use one
    positive pair and one negative pair"""

    @abstractmethod
    def loss(self, z1, z2_con_z1, z3, z1_rec, z2_con_z1_rec, z3_rec):
        pass

    def __call__(self, z1, z2_con_z1, z3, z1_rec, z2_con_z1_rec, z3_rec):
        return self.loss(z1, z2_con_z1, z3, z1_rec, z2_con_z1_rec, z3_rec)


class AnInfoNCELoss(CLLoss):
    def __init__(self, batch_size, lambda_train, lambda_activation, device='cuda'):
        self.lambda_train = lambda_train
        self.lambda_activation = lambda_activation
        self.mask = self.mask_correlated_samples(batch_size).to(device)
        self.normalize = True
        self.activation = torch.nn.functional.softplus

    def mask_correlated_samples(self, batch_size):
        N = batch_size
        mask = np.ones((N,N))
        mask -= np.diag(np.ones(N-1), 1)
        mask[-1][0] = 0
        mask = mask.astype('bool')
        return torch.Tensor(mask).bool()
    
    @property
    def effective_lambda(self):
        return self.lambda_activation(self.lambda_train)

    def loss(self, z1, z2_con_z1, z3, z1_rec, z2_con_z1_rec, z3_rec):
        del z1, z2_con_z1, z3

        batch_size = z1_rec.size(0)
        N = 2 * batch_size

        if self.normalize:
            z1_rec = z1_rec / torch.norm(z1_rec, p=2, dim=-1, keepdim=True)
            z2_con_z1_rec = z2_con_z1_rec / torch.norm(
                z2_con_z1_rec, p=2, dim=-1, keepdim=True
            )
            if z3_rec is not None:
                z3_rec = z3_rec / torch.norm(z3_rec, p=2, dim=-1, keepdim=True)
        
        def get_neg_term(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
            # Random permutation avoids sequential bias in burst attacks
            perm = torch.randperm(z_b.size(0), device=z_b.device)
            z_b = z_b[perm]
            partial_a = torch.einsum("ij,ij -> i", z_a,  self.effective_lambda * z_a)
            partial_b = torch.einsum("ij,ij -> i", z_b,  self.effective_lambda * z_b)
            neg = - partial_a.unsqueeze(1) / 2 \
              - partial_b.unsqueeze(0) / 2 \
              + torch.einsum("ij,kj -> ik", z_a,  self.effective_lambda * z_b)

            # Remove pairs of identical samples
            neg = neg[self.mask].reshape(len(z_a), -1)

            return neg
        
        neg_cross = get_neg_term(z1_rec, z2_con_z1_rec)

        neg = torch.cat((
            torch.cat((get_neg_term(z1_rec, z1_rec), neg_cross), 1), 
            torch.cat((get_neg_term(z2_con_z1_rec, z2_con_z1_rec), neg_cross), 1)
        ), dim=0)
        
        diff_z1_rec_z2_con_z1_rec = z1_rec - z2_con_z1_rec
        pos = - torch.einsum("ij,ij -> i", diff_z1_rec_z2_con_z1_rec,  self.effective_lambda * diff_z1_rec_z2_con_z1_rec) / 2 
        
        pos = torch.cat((pos, pos), dim=0).reshape(N, 1)        
        neg_and_pos = torch.cat((neg, pos), dim=1)
        loss_neg = torch.logsumexp(neg_and_pos , dim=1)
   
        loss_pos = -pos
    
        loss = loss_pos + loss_neg
            
        loss_mean = torch.mean(loss)
        
        loss_pos_mean, loss_neg_mean = torch.mean(loss_pos), torch.mean(loss_neg)
    
        return loss_mean, loss, [loss_pos_mean, loss_neg_mean]


def train(
    model,
    window_size,
    step_percent,
    ae_batch_size,
    train_loader,
    val_loader,
    test_loader,
    start_epoch,
    num_epochs,
    optimizer,
    run,
    patience,
    checkpoint,
    device="cuda",
    alpha=0.1,
    temperature=0.1,
):
    best_pr_auc = 0.0
    cnt_wait = 0
    criterion = nn.MSELoss(reduction="none")
    total_train_loss = 0
    total_train_loss_mse = 0
    total_train_loss_cl = 0
    for epoch in (pbar := tqdm(range(start_epoch + 1, num_epochs + 1), desc="Epochs")):
        total_train_loss = 0
        total_train_loss_mse = 0
        total_train_loss_cl = 0
        model.train()
        for batch in train_loader:
            batch.batch_edge_couples = batch.edge_label_index.t()
            batch = batch.to(device)

            train_emb = model.encoder(
                batch.edge_index,
                batch.edge_attr,
                batch.batch_edge_couples,
                batch.num_nodes,
            )
            ae_train_loader = DataLoader(
                SequentialDataset(
                    train_emb,
                    window=window_size,
                    step=int(window_size * step_percent),
                    device=device,
                ),
                batch_size=ae_batch_size,
                collate_fn=collate_fn,
            )
            accumulated_loss = torch.tensor(0.0, device=device)
            accumulated_loss_mse = torch.tensor(0.0, device=device)
            accumulated_loss_cl = torch.tensor(0.0, device=device)
            seq_count = 0
            for ae_batch, mask in ae_train_loader:
                outputs = model.transformer(ae_batch, mask)
                # NOTE ON IMPLEMENTATION:
                # We purposefully do not detach the target embedding here.
                # Empirically, we observed that allowing gradients to flow through
                # the target improves convergence speed and representation quality
                # compared to a standard stop-gradient approach, likely by
                # enforcing tighter coupling between the encoder and transformer
                # during training.
                loss_mse = criterion(outputs, ae_batch)
                loss_mse = torch.sum(loss_mse * mask) / torch.sum(mask)

                # Cross-View Contrastive Learning (InfoNCE)
                valid_mask = mask.sum(dim=-1) > 0 # Find valid tokens
                # Only use a subset to avoid OOM for similarity matrix
                valid_indices = torch.nonzero(valid_mask, as_tuple=True)
                num_valid = len(valid_indices[0])
                if num_valid > 0:
                    max_samples = 2048
                    if num_valid > max_samples:
                        perm = torch.randperm(num_valid, device=device)[:max_samples]
                        idx_0 = valid_indices[0][perm]
                        idx_1 = valid_indices[1][perm]
                    else:
                        idx_0 = valid_indices[0]
                        idx_1 = valid_indices[1]
                    
                    z_graph = model.projector(ae_batch[idx_0, idx_1])
                    z_trans = model.projector(outputs[idx_0, idx_1])
                    
                    z_graph = nn.functional.normalize(z_graph, dim=1)
                    z_trans = nn.functional.normalize(z_trans, dim=1)
                    
                    # AnInfoNCE Loss calculation
                    # Ensure effective_lambda parameter matches dimension (scalar or vector)
                    # For simplicity, using a uniform lambda initially, or you can parameritize it
                    lambda_train = torch.tensor(1.0, device=device)
                    # Softplus activation for lambda
                    lambda_activation = torch.nn.functional.softplus

                    # Initialize AnInfoNCELoss passing current effective batch size
                    an_info_nce = AnInfoNCELoss(batch_size=len(z_graph), 
                                                lambda_train=lambda_train, 
                                                lambda_activation=lambda_activation, 
                                                device=device)
                    
                    # z1_rec = z_graph, z2_con_z1_rec = z_trans, others are not used (del)
                    loss_cl, _, _ = an_info_nce(z1=None, z2_con_z1=None, z3=None, 
                                                z1_rec=z_graph, z2_con_z1_rec=z_trans, z3_rec=None)
                else:
                    loss_cl = torch.tensor(0.0, device=device)
                
                loss = loss_mse + alpha * loss_cl

                accumulated_loss += loss
                accumulated_loss_mse += loss_mse
                accumulated_loss_cl += loss_cl
                seq_count += 1

            # Calculate the mean loss for the batch and backpropagate through both components
            if seq_count > 0:
                loss = accumulated_loss / seq_count
                total_train_loss += loss.item()
                total_train_loss_mse += (accumulated_loss_mse / seq_count).item()
                total_train_loss_cl += (accumulated_loss_cl / seq_count).item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        total_train_loss /= len(train_loader)
        total_train_loss_mse /= len(train_loader)
        total_train_loss_cl /= len(train_loader)
        val_loss, val_errors, val_labels = validate(
            model, val_loader, ae_batch_size, window_size, device
        )
        val_pr_auc = average_precision_score(val_labels.cpu(), val_errors.cpu())
        # Find the best threshold based on the validation set
        threshold = find_threshold(val_errors, val_labels, method="supervised")
        # For debugging purposes
        test_f1, test_pr_auc, _, _, _ = test(
            model, test_loader, ae_batch_size, window_size, device, threshold
        )

        # Keep saving the model if it produces the same or better validation PR-AUC
        if val_pr_auc >= best_pr_auc:
            model.save_checkpoint(
                checkpoint,
                optimizer=optimizer,
                epoch=epoch,
                threshold=threshold,
            )

        # Stop training if the validation PR-AUC does not improve for a number of epochs
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            cnt_wait = 0
        else:
            cnt_wait += 1
            if cnt_wait >= patience:
                print("Early stopping!")
                break
        pbar.set_postfix(
            {
                "tot_loss": f"{total_train_loss:.4f}",
                "mse": f"{total_train_loss_mse:.4f}",
                "cl": f"{total_train_loss_cl:.4f}",
                "val_pr_auc": f"{val_pr_auc:.4f}",
            }
        )
        run.log(
            {
                "train_loss": total_train_loss,
                "train_loss_mse": total_train_loss_mse,
                "train_loss_cl": total_train_loss_cl,
                "val_loss": val_loss,
                "val_pr_auc": val_pr_auc,
                "test_f1": test_f1,
                "test_pr_auc": test_pr_auc,
            }
        )
    chk = torch.load(checkpoint, weights_only=True)
    model.load_state_dict(chk["model_state_dict"])
    return model, chk["threshold"]


def find_threshold(errors, labels=None, method="unsupervised", multiplier=10.0):
    if method == "unsupervised":
        median = errors.median()
        mad = (
            errors - median
        ).abs().median() * 1.4826  # Factor for normal distribution
        best_threshold = median + multiplier * mad
    elif method == "supervised" and labels is not None:
        best_f1 = 0.0
        best_threshold = errors.mean()
        for threshold in torch.linspace(errors.min(), errors.max(), steps=500):
            val_pred = (errors > threshold).int()
            f1 = f1_score(
                labels.cpu(), val_pred.cpu(), average="macro", zero_division=0
            )
            if f1 > best_f1:
                best_threshold = threshold.item()
                best_f1 = f1
    else:
        raise ValueError(
            "Invalid method for threshold finding. Use 'unsupervised' or 'supervised' with labels."
        )
    return best_threshold


def calculate_errors(outputs, batch, mask):
    squared_errors = ((outputs - batch) ** 2) * mask
    valid_mask = mask.sum(dim=-1) > 0
    valid_counts = torch.sum(mask, dim=-1)
    mean_errors = torch.zeros_like(valid_counts, dtype=torch.float32)
    if valid_mask.any():
        mean_errors[valid_mask] = torch.sum(squared_errors, dim=-1)[
            valid_mask
        ] / torch.clamp(valid_counts[valid_mask], min=1)
    mean_errors = torch.nan_to_num(mean_errors, nan=0.0, posinf=1e6, neginf=-1e6)
    return mean_errors[valid_mask]


def validate(model, val_loader, ae_batch_size, window_size, device):
    criterion = nn.MSELoss(reduction="none")
    model.eval()
    errors = []
    labels = []
    total_val_loss = 0
    with torch.inference_mode():
        for batch in val_loader:
            batch.batch_edge_couples = batch.edge_label_index.t()
            batch = batch.to(device)

            val_emb = model.encoder(
                batch.edge_index,
                batch.edge_attr,
                batch.batch_edge_couples,
                batch.num_nodes,
            )
            labels.append(batch.edge_label.cpu())
            ae_val_loader = DataLoader(
                SequentialDataset(
                    val_emb, window=window_size, step=window_size, device=device
                ),
                batch_size=ae_batch_size,
                collate_fn=collate_fn,
            )
            accumulated_loss = torch.tensor(0.0, device=device)
            seq_count = 0
            for ae_batch, mask in ae_val_loader:
                outputs = model.transformer(ae_batch, mask)
                loss = criterion(outputs, ae_batch)
                loss = torch.sum(loss * mask) / torch.sum(mask)
                accumulated_loss += loss
                seq_count += 1
                batch_errors = calculate_errors(outputs, ae_batch, mask)
                errors.append(batch_errors.cpu())
            if seq_count > 0:
                total_val_loss += (accumulated_loss / seq_count).item()
    total_val_loss /= len(val_loader)
    labels = torch.cat(labels)
    errors = torch.cat(errors)
    return total_val_loss, errors, labels


def test(model, test_loader, ae_batch_size, window_size, device, threshold):
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.perf_counter()
    model.eval()
    errors = []
    labels = []
    with torch.inference_mode():
        for batch in test_loader:
            batch.batch_edge_couples = batch.edge_label_index.t()
            batch = batch.to(device)

            test_emb = model.encoder(
                batch.edge_index,
                batch.edge_attr,
                batch.batch_edge_couples,
                batch.num_nodes,
            )
            labels.append(batch.edge_label.cpu())
            ae_test_loader = DataLoader(
                SequentialDataset(
                    test_emb, window=window_size, step=window_size, device=device
                ),
                batch_size=ae_batch_size,
                collate_fn=collate_fn,
            )
            for ae_batch, mask in ae_test_loader:
                outputs = model.transformer(ae_batch, mask)
                batch_errors = calculate_errors(outputs, ae_batch, mask)
                errors.append(batch_errors.cpu())
    labels = torch.cat(labels)
    errors = torch.cat(errors)
    if threshold is not None:
        test_pred = (errors > threshold).int()
    else:
        print("No threshold provided, using mean of errors for prediction.")
        test_pred = (errors > errors.mean()).int()
    torch.cuda.synchronize() if device == "cuda" else None
    prediction_time = time.perf_counter() - start_time
    f1 = f1_score(labels, test_pred, average="macro", zero_division=0)
    pr_auc = average_precision_score(labels, errors)
    return f1, pr_auc, errors, labels, prediction_time
