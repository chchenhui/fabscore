# Runtime Stability Controller

A runtime stability and recovery layer for neural network training that supervises optimizer updates using external measurement signals, without modifying the optimizer.

This library introduces a control-theoretic supervisory layer for training deep neural networks. It monitors optimizer-proposed updates at runtime using secondary measurement signals (e.g. validation probes) and enables automatic detection and recovery from destabilizing updates via rollback to a previously accepted safe state.

The controller is optimizer-agnostic, framework-compatible, and designed to integrate into existing training pipelines with minimal overhead.

---

## Motivation

Training modern neural networks is increasingly fragile. Rare but severe destabilizing updates — caused by outlier minibatches, numerical instabilities, or transient distribution shifts — can irreversibly corrupt the training trajectory.

Existing approaches (adaptive optimizers, gradient clipping, trust regions) are primarily preventive. Once a catastrophic update has occurred, standard training pipelines offer no principled mechanism for detection, rollback, or recovery.

This project addresses that gap by introducing a runtime supervisory controller that treats optimization as a controlled stochastic process.

---

## Key Properties

- Optimizer-agnostic: works with SGD, Adam, AdamW, etc.
- External supervision: does not modify the optimizer update rule
- Runtime detection of destabilizing updates
- Exact recovery via rollback to a safe snapshot
- Minimal overhead using small validation probes
- Compatible with PyTorch training loops (initial reference implementation)

---

## High-Level Concept

At each training step:

1. The optimizer proposes a parameter update.
2. A secondary measurement signal evaluates the proposed update.
3. An innovation signal measures deviation from expected stable behavior.
4. The controller decides to:
   - accept the update, or
   - reject and rollback to the last safe state.

This decouples optimization from stability enforcement.

---
## Minimal Usage Sketch (PyTorch)

```python
controller = StabilityController(
    probe=ValidationProbe(model, val_loader),
    threshold=epsilon
)

for batch in train_loader:
    loss = compute_loss(model, batch)
    loss.backward()

    controller.step(
        model=model,
        optimizer=optimizer,
        loss=loss
    )
```
## Related Paper

This repository provides a reference implementation for the paper: 

Automatic Stability and Recovery for Neural Network Training
Barak Or
Preprint, 2026

The paper formalizes the method, provides theoretical runtime safety guarantees, and demonstrates empirical recovery behavior on convolutional and Transformer-based architectures.

The full paper PDF is available here:  
[`paper/Automatic_Stability_and_Recovery_for_Neural_Network_Training.pdf`](paper/Automatic_Stability_and_Recovery_for_Neural_Network_Training.pdf)

You can easily run this on Google Colab:
```python
!git clone https://github.com/BarakOr1/runtime-stability-controller.git && \
pip install -q torch && \
cd runtime-stability-controller && \
export PYTHONPATH=$PWD && \
python -c "import runtime_stability_controller; print('import ok')" && \
python examples/minimal_pytorch.py
```

