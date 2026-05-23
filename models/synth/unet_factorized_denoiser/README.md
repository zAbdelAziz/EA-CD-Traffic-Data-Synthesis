# Factorized U-Net denoiser

This is the main structured denoiser used by the repository.

## Input and output

Input:

```text
x_t: [B, T, 20]
t:   [B]
y:   [B] or None
```

Output:

```text
eps_ego:   [B, T, 2]
eps_slots: [B, T, 6, 2]
p_logits:  [B, T, 6]
```

## Internal stages

1. Build conditioning vector from time and label embeddings.
2. Split `x_t` into ego and six neighbor slots.
3. Project ego and neighbor states into token space.
4. Add learned slot embeddings.
5. Apply entity-attention blocks at each time step.
6. Flatten entity tokens into temporal channels.
7. Apply temporal U-Net denoising.
8. Add token residual delta and project to factorized heads.

## Why factorized heads matter

Continuous motion/geometry and binary occupancy have different semantics. The model therefore predicts continuous channels in noise space while supervising occupancy in clean-mask space. The diffusion kernel later converts the occupancy probability to an equivalent epsilon value for reverse sampling.

This is the code-level counterpart of thesis Sections 3.3.1-3.3.3.
