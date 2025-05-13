# for checking
# How Can You Check If Adversarial Attacks Are Affecting Your Model?
# You can conduct an adversarial attack on your model to check if it’s vulnerable to small perturbations. 
# The simplest adversarial attack is FGSM (Fast Gradient Sign Method), which adds a small noise to the 
# input in the direction of the gradient of the loss with respect to the input.

# Here’s a small example of how you can apply FGSM:

import torch
import torch.nn.functional as F

def fgsm_attack(model, data, target, epsilon=0.1):
    data.requires_grad = True
    output = model(data)
    loss = F.mse_loss(output, target)  # Use the appropriate loss function
    model.zero_grad()
    loss.backward()
    
    # Collect the data gradient
    data_grad = data.grad.data
    
    # Create the adversarial example
    perturbed_data = data + epsilon * data_grad.sign()
    return perturbed_data

# Example usage:
adv_seismic = fgsm_attack(model, seismic_input, velocity_gt)
output = model(adv_seismic)

# How to Mitigate This and Improve Your Model’s Robustness:
# Adversarial Training: Train the model on both clean and adversarial examples. 
# This helps the model become more robust to such attacks.

# Example (simple adversarial training loop):

for seismic, velocity in tqdm(dataloader, desc="Adversarial Training"):
    seismic = seismic.to(device)
    velocity = velocity.to(device)

    # Adversarial example
    adv_seismic = fgsm_attack(model, seismic, velocity)
    
    # Train on both original and adversarial examples
    optimizer.zero_grad()
    output = model(seismic)
    loss = criterion(output, velocity)
    loss.backward()
    optimizer.step()

    # Repeat for adversarial example
    optimizer.zero_grad()
    output_adv = model(adv_seismic)
    loss_adv = criterion(output_adv, velocity)
    loss_adv.backward()
    optimizer.step()
