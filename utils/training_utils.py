import os
import torch
from utils.file_utlis import get_last_file_with_extension

def save_checkpoint(model, optimizer, scheduler, epoch, global_step, checkpoint_dir, checkpoint_total_limit):
    """
    Save model checkpoint
    """
    os.path.exists(checkpoint_dir) or os.makedirs(checkpoint_dir)
    
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, os.path.join(checkpoint_dir + f'/epoch_{epoch+1}.pth'))

    # Remove old checkpoints if the total number of checkpoints exceeds the limit
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if len(checkpoint_files) > checkpoint_total_limit:
        checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        os.remove(os.path.join(checkpoint_dir, checkpoint_files[0]))
    print(f'Checkpoint saved at {checkpoint_dir + f"/epoch_{epoch+1}.pth"}')
    
def load_checkpoint(model, optimizer, scheduler, checkpoint_dir, gpu_id):
    """
    Loads the latest checkpoint if available and updates the model, optimizer, and scheduler.

    Args:
        model: The model to load state dict into.
        optimizer: The optimizer to load state dict into.
        scheduler: The scheduler to load state dict into.
        proj_exp_dir: Directory to search for the checkpoint file.
        gpu_id: The GPU ID, used for printing messages if gpu_id == 0.

    Returns:
        continue_ep: The next epoch to continue training from.
        global_step: The global step loaded from the checkpoint.
    """
    checkpoint_path = get_last_file_with_extension(checkpoint_dir, '.pth')
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{gpu_id}')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        continue_ep = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        if gpu_id == 0:
            print(f'Continue_training...ep:{continue_ep + 1}')
    else:
        continue_ep = 0
        global_step = 0

    return continue_ep, global_step

