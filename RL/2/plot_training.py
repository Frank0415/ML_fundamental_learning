"""Plot DPO training progress from logged metrics."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

# Data parsed from training output
# Each entry: step, loss, entropy, logits_c, logits_r, acc, r_c, r_r, r_acc, margin, logps_c, logps_r
# Missing steps interpolated with None

raw_data = {
    5:  {'loss': 0.2098,   'entropy': 2.869, 'logits_c': -1.197, 'logits_r': -1.265, 'acc': 0.4821, 'r_c': 2.829,  'r_r': -1.871, 'r_acc': 0.9, 'margin': 4.7,    'logps_c': -151.3, 'logps_r': -109.5},
    10: {'loss': 0.0792,   'entropy': 2.608, 'logits_c': -1.142, 'logits_r': -1.292, 'acc': 0.5317, 'r_c': 4.653,  'r_r': -3.443, 'r_acc': 1.0, 'margin': 8.095,  'logps_c': -123.8, 'logps_r': -124.9},
    15: {'loss': 0.02698,  'entropy': 2.263, 'logits_c': -1.279, 'logits_r': -1.185, 'acc': 0.5435, 'r_c': 5.394,  'r_r': -4.185, 'r_acc': 1.0, 'margin': 9.58,   'logps_c': -114.4, 'logps_r': -135},
    20: {'loss': 0.001193, 'entropy': 2.157, 'logits_c': -1.234, 'logits_r': -1.193, 'acc': 0.5476, 'r_c': 5.638,  'r_r': -4.993, 'r_acc': 1.0, 'margin': 10.63,  'logps_c': -106.4, 'logps_r': -140.3},
    30: {'loss': 1.055e-05,'entropy': 2.121, 'logits_c': -1.233, 'logits_r': -1.403, 'acc': 0.6024, 'r_c': 6.979,  'r_r': -5.879, 'r_acc': 1.0, 'margin': 12.86,  'logps_c': -96.87, 'logps_r': -149.2},
    35: {'loss': 8.946e-06,'entropy': 2.07,  'logits_c': -1.216, 'logits_r': -1.377, 'acc': 0.6054, 'r_c': 6.869,  'r_r': -6.245, 'r_acc': 1.0, 'margin': 13.11,  'logps_c': -97.31, 'logps_r': -152.9},
    40: {'loss': 4.345e-06,'entropy': 1.946, 'logits_c': -1.401, 'logits_r': -1.187, 'acc': 0.583,  'r_c': 6.574,  'r_r': -6.589, 'r_acc': 1.0, 'margin': 13.16,  'logps_c': -99.66, 'logps_r': -157.1},
    50: {'loss': 9.356e-06,'entropy': 2.047, 'logits_c': -1.188, 'logits_r': -1.298, 'acc': 0.5795, 'r_c': 6.242,  'r_r': -5.89,  'r_acc': 1.0, 'margin': 12.13,  'logps_c': -96.29, 'logps_r': -149},
    55: {'loss': 4.802e-06,'entropy': 2.063, 'logits_c': -1.255, 'logits_r': -1.327, 'acc': 0.5999, 'r_c': 7.441,  'r_r': -6.688, 'r_acc': 1.0, 'margin': 14.13,  'logps_c': -98.33, 'logps_r': -156.3},
    60: {'loss': 4.37e-06, 'entropy': 1.955, 'logits_c': -1.392, 'logits_r': -1.226, 'acc': 0.5786, 'r_c': 6.899,  'r_r': -6.71,  'r_acc': 1.0, 'margin': 13.61,  'logps_c': -100,   'logps_r': -158.4},
    70: {'loss': 3.525e-06,'entropy': 2.096, 'logits_c': -1.274, 'logits_r': -1.348, 'acc': 0.6132, 'r_c': 7.643,  'r_r': -6.44,  'r_acc': 1.0, 'margin': 14.08,  'logps_c': -96.94, 'logps_r': -153.2},
    75: {'loss': 6.707e-06,'entropy': 2.11,  'logits_c': -1.174, 'logits_r': -1.379, 'acc': 0.6064, 'r_c': 7.079,  'r_r': -6.019, 'r_acc': 1.0, 'margin': 13.1,   'logps_c': -94.48, 'logps_r': -149.3},
    80: {'loss': 7.314e-06,'entropy': 2.045, 'logits_c': -1.272, 'logits_r': -1.413, 'acc': 0.6047, 'r_c': 6.27,   'r_r': -5.842, 'r_acc': 1.0, 'margin': 12.11,  'logps_c': -95.34, 'logps_r': -150},
    90: {'loss': 1.309e-06,'entropy': 1.978, 'logits_c': -1.473, 'logits_r': -1.222, 'acc': 0.6053, 'r_c': 7.496,  'r_r': -7.017, 'r_acc': 1.0, 'margin': 14.51,  'logps_c': -100.9, 'logps_r': -161.4},
    95: {'loss': 7.568e-06,'entropy': 2.047, 'logits_c': -1.261, 'logits_r': -1.377, 'acc': 0.5953, 'r_c': 6.19,   'r_r': -5.96,  'r_acc': 1.0, 'margin': 12.15,  'logps_c': -96.3,  'logps_r': -151.1},
    100:{'loss': 5.276e-06,'entropy': 1.93,  'logits_c': -1.39,  'logits_r': -1.211, 'acc': 0.5747, 'r_c': 6.394,  'r_r': -6.657, 'r_acc': 1.0, 'margin': 13.05,  'logps_c': -100.1, 'logps_r': -158.4},
    110:{'loss': 3.196e-06,'entropy': 1.976, 'logits_c': -1.44,  'logits_r': -1.335, 'acc': 0.6127, 'r_c': 6.883,  'r_r': -6.126, 'r_acc': 1.0, 'margin': 13.01,  'logps_c': -94.98, 'logps_r': -152.4},
    115:{'loss': 7.358e-06,'entropy': 2.092, 'logits_c': -1.149, 'logits_r': -1.392, 'acc': 0.5927, 'r_c': 6.787,  'r_r': -6.211, 'r_acc': 1.0, 'margin': 13,     'logps_c': -96.62, 'logps_r': -151.5},
    120:{'loss': 4.866e-06,'entropy': 2.011, 'logits_c': -1.281, 'logits_r': -1.246, 'acc': 0.5867, 'r_c': 7.107,  'r_r': -6.744, 'r_acc': 1.0, 'margin': 13.85,  'logps_c': -99.39, 'logps_r': -157},
    130:{'loss': 8.73e-06, 'entropy': 2.068, 'logits_c': -1.274, 'logits_r': -1.393, 'acc': 0.6064, 'r_c': 6.352,  'r_r': -6.185, 'r_acc': 1.0, 'margin': 12.54,  'logps_c': -98.8,  'logps_r': -155},
    135:{'loss': 4.73e-06, 'entropy': 2.026, 'logits_c': -1.373, 'logits_r': -1.36,  'acc': 0.6142, 'r_c': 6.862,  'r_r': -6.284, 'r_acc': 1.0, 'margin': 13.15,  'logps_c': -97.11, 'logps_r': -154.3},
    140:{'loss': 1.436e-06,'entropy': 1.932, 'logits_c': -1.505, 'logits_r': -1.171, 'acc': 0.5976, 'r_c': 7.226,  'r_r': -6.926, 'r_acc': 1.0, 'margin': 14.15,  'logps_c': -100.4, 'logps_r': -160.5},
    150:{'loss': 4.302e-06,'entropy': 2.004, 'logits_c': -1.356, 'logits_r': -1.278, 'acc': 0.6035, 'r_c': 7.242,  'r_r': -6.657, 'r_acc': 1.0, 'margin': 13.9,   'logps_c': -98.8,  'logps_r': -157.5},
}

steps = sorted(raw_data.keys())

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('DPO Training Progress - Qwen2.5-0.5B-Instruct\n(β=0.1, lr=1e-5, batch=2, 100 samples, 3 epochs)',
             fontsize=14, fontweight='bold', y=1.02)

colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#607D8B']
epoch_boundaries = [50, 100]
epoch_labels = [f'Epoch {i+1}' for i in range(3)]

# 1) Loss (log scale)
ax = axes[0, 0]
losses = [raw_data[s]['loss'] for s in steps]
ax.plot(steps, losses, 'o-', color=colors[0], linewidth=1.8, markersize=4)
ax.set_yscale('log')
ax.set_xlabel('Training Step', fontsize=11)
ax.set_ylabel('DPO Loss (log scale)', fontsize=11)
ax.set_title('Loss', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
for eb in epoch_boundaries:
    ax.axvline(eb, color='gray', linestyle='--', alpha=0.5)
for i, eb in enumerate(epoch_boundaries):
    ax.text(eb-25, ax.get_ylim()[1]*0.9, epoch_labels[i], ha='center', fontsize=9, color='gray')

# Annotate key values
ax.annotate(f'{losses[0]:.4f}', (steps[0], losses[0]), textcoords="offset points",
            xytext=(0, 12), ha='center', fontsize=8, color=colors[0])

# 2) Rewards: Chosen vs Rejected
ax = axes[0, 1]
r_chosen = [raw_data[s]['r_c'] for s in steps]
r_rejected = [raw_data[s]['r_r'] for s in steps]
ax.plot(steps, r_chosen, 'o-', color=colors[1], linewidth=1.8, markersize=4, label='chosen (y_w)')
ax.plot(steps, r_rejected, 's-', color=colors[2], linewidth=1.8, markersize=4, label='rejected (y_l)')
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Training Step', fontsize=11)
ax.set_ylabel('Implicit Reward (β × logratio)', fontsize=11)
ax.set_title('Implicit Rewards', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
for eb in epoch_boundaries:
    ax.axvline(eb, color='gray', linestyle='--', alpha=0.5)
ax.annotate(f'+{r_chosen[0]:.2f}', (steps[0], r_chosen[0]), textcoords="offset points",
            xytext=(0, 10), ha='center', fontsize=8, color=colors[1])
ax.annotate(f'{r_rejected[0]:.2f}', (steps[0], r_rejected[0]), textcoords="offset points",
            xytext=(0, -14), ha='center', fontsize=8, color=colors[2])

# 3) Reward Margin
ax = axes[0, 2]
margins = [raw_data[s]['margin'] for s in steps]
ax.plot(steps, margins, 'o-', color=colors[3], linewidth=2, markersize=4)
ax.fill_between(steps, margins, alpha=0.15, color=colors[3])
ax.set_xlabel('Training Step', fontsize=11)
ax.set_ylabel('Reward Margin', fontsize=11)
ax.set_title('Reward Margin (chosen - rejected)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
for eb in epoch_boundaries:
    ax.axvline(eb, color='gray', linestyle='--', alpha=0.5)
ax.annotate(f'{margins[0]:.1f}', (steps[0], margins[0]), textcoords="offset points",
            xytext=(0, 10), ha='center', fontsize=8, color=colors[3])
ax.annotate(f'{margins[-1]:.1f}', (steps[-1], margins[-1]), textcoords="offset points",
            xytext=(0, -14), ha='right', fontsize=8, color=colors[3])

# 4) Log-Probs Chosen vs Rejected
ax = axes[1, 0]
logps_c = [raw_data[s]['logps_c'] for s in steps]
logps_r = [raw_data[s]['logps_r'] for s in steps]
ax.plot(steps, logps_c, 'o-', color=colors[1], linewidth=1.8, markersize=4, label='chosen (y_w)')
ax.plot(steps, logps_r, 's-', color=colors[2], linewidth=1.8, markersize=4, label='rejected (y_l)')
ax.set_xlabel('Training Step', fontsize=11)
ax.set_ylabel('Log-Probs (sum over tokens)', fontsize=11)
ax.set_title('log π_θ(y|x)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
for eb in epoch_boundaries:
    ax.axvline(eb, color='gray', linestyle='--', alpha=0.5)
# Add arrow annotation showing divergence
mid = len(steps)//2
ax.annotate('', xy=(steps[-1], logps_r[-1]), xytext=(steps[-1], logps_c[-1]),
            arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
ax.text(steps[-1]+3, (logps_c[-1]+logps_r[-1])/2, f'd={abs(logps_c[-1]-logps_r[-1]):.0f}',
        fontsize=9, color='purple', fontweight='bold')

# 5) Entropy
ax = axes[1, 1]
entropies = [raw_data[s]['entropy'] for s in steps]
ax.plot(steps, entropies, 'o-', color=colors[4], linewidth=1.8, markersize=4)
ax.set_xlabel('Training Step', fontsize=11)
ax.set_ylabel('Entropy', fontsize=11)
ax.set_title('Policy Entropy', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
for eb in epoch_boundaries:
    ax.axvline(eb, color='gray', linestyle='--', alpha=0.5)
ax.annotate(f'{entropies[0]:.2f}', (steps[0], entropies[0]), textcoords="offset points",
            xytext=(0, 10), ha='center', fontsize=8, color=colors[4])

# 6) Token Accuracy & Logits
ax = axes[1, 2]
accs = [raw_data[s]['acc'] for s in steps]
ax.plot(steps, accs, 'o-', color=colors[5], linewidth=1.8, markersize=4, label='token accuracy')
logits_c = [raw_data[s]['logits_c'] for s in steps]
ax2 = ax.twinx()
ax2.plot(steps, logits_c, '^--', color=colors[1], linewidth=1.2, markersize=3, alpha=0.6, label='avg logit (chosen)')
ax.set_xlabel('Training Step', fontsize=11)
ax.set_ylabel('Token Accuracy', fontsize=11, color=colors[5])
ax2.set_ylabel('Avg Logit (chosen)', fontsize=11, color=colors[1])
ax.set_title('Token Accuracy & Avg Logits', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
for eb in epoch_boundaries:
    ax.axvline(eb, color='gray', linestyle='--', alpha=0.5)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig('/home/xc/Documents/learning_ML/RL/2/training_progress.png', dpi=150, bbox_inches='tight')
print('Saved to training_progress.png')
