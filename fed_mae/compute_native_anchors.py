import sys
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add the directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_peft_finetune_FedAvg import get_args
import models_vit
from util.data_utils import DatasetFLFinetune
from util.misc import safe_torch_load

def compute_native_anchors():
    # Construct minimal arguments needed for the dataset and model
    class Args:
        data_set = 'Retina'
        data_path = '/content/retina_local/Retina'
        nb_classes = 2
        model = 'vit_base_patch16'
        drop_path = 0.1
        global_pool = True
        finetune = '/content/fedmamba_salt/data/ckpts/mae_vit_base.pth'
        # FL specific args needed by DatasetFLFinetune
        split_type = 'central'
        n_clients = 5
        num_local_clients = -1
        seed = 0
        device = 'cuda'

    args = Args()

    device = torch.device(args.device)

    # 1. Build Model
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    # Load frozen MAE checkpoint
    checkpoint = safe_torch_load(args.finetune, map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            del checkpoint_model[k]
    
    from util.pos_embed import interpolate_pos_embed
    interpolate_pos_embed(model, checkpoint_model)
    model.load_state_dict(checkpoint_model, strict=False)
    
    model.to(device)
    model.eval()

    # 2. Build Dataset
    # We use the training set to compute the native centroids.
    # We use the 'central' split to ensure we have access to all data.
    dataset_train = DatasetFLFinetune(args=args, phase='train')
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=128, num_workers=4, pin_memory=True, drop_last=False
    )

    # 3. Compute Centroids
    class_sums = {i: torch.zeros(768, device=device) for i in range(args.nb_classes)}
    class_counts = {i: 0 for i in range(args.nb_classes)}

    print("Computing Native MAE Centroids...")
    with torch.no_grad():
        for samples, targets in tqdm(data_loader):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Extract pristine MAE features
            with torch.cuda.amp.autocast():
                features = model.forward_features(samples)
            
            features = F.normalize(features, dim=-1)
            
            for c in range(args.nb_classes):
                mask = targets == c
                if mask.sum() > 0:
                    class_sums[c] += features[mask].sum(dim=0)
                    class_counts[c] += mask.sum().item()

    # 4. Finalize
    anchors = torch.zeros(args.nb_classes, 768)
    for c in range(args.nb_classes):
        centroid = class_sums[c] / max(class_counts[c], 1)
        anchors[c] = F.normalize(centroid, dim=-1).cpu()
        print(f"Class {c} Centroid computed from {class_counts[c]} samples.")

    # 5. Check orthogonality/similarity
    sim = F.cosine_similarity(anchors[0].unsqueeze(0), anchors[1].unsqueeze(0)).item()
    print(f"Cosine Similarity between Native Anchors: {sim:.4f}")

    # 6. Save
    save_path = '/content/fedmamba_salt/data/ckpts/native_anchors.pt'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(anchors, save_path)
    print(f"Native Anchors saved to {save_path}")

if __name__ == '__main__':
    compute_native_anchors()
