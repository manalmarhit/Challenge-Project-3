import numpy as np
import matplotlib.pyplot as plt

# Read names
with open('names.txt', 'r') as f:
    names = f.read().splitlines()

# All characters
chars = ['.'] + list('abcdefghijklmnopqrstuvwxyz')
n = len(chars)

char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# Build transition matrix
matrix = np.zeros((n, n), dtype=int)

for name in names:
    word = '.' + name + '.'
    for ch1, ch2 in zip(word, word[1:]):
        i = char_to_idx[ch1]
        j = char_to_idx[ch2]
        matrix[i][j] += 1

# Convert counts to probabilities (each row sums to 1)
prob_matrix = matrix / matrix.sum(axis=1, keepdims=True)

# Plot heatmap
fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(prob_matrix, cmap='Blues')

# Label axes
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(chars)
ax.set_yticklabels(chars)

ax.set_xlabel("Second Letter")
ax.set_ylabel("First Letter")
ax.set_title("Bigram Transition Probabilities (Real Names)")

plt.colorbar(im)
plt.tight_layout()
plt.savefig('heatmap_real.png', dpi=150)
print("Saved heatmap_real.png")

# written questions

# Starting letters: look at row 0 (the '.' row)
start_probs = prob_matrix[0]
sorted_start = sorted(range(n), key=lambda i: start_probs[i], reverse=True)
print("\nTop 3 starting letters:")
for i in sorted_start[1:4]:  # skip index 0 which is '.' itself
    print(f"  {chars[i]}: {start_probs[i]:.4f}")

print("Bottom 3 starting letters:")
# filter out '.' and zeros
non_zero_start = [i for i in sorted_start if i != 0 and start_probs[i] > 0]
for i in non_zero_start[-3:]:
    print(f"  {chars[i]}: {start_probs[i]:.4f}")

# Ending letters: look at column 0 (the '.' column)
end_probs = prob_matrix[:, 0]
sorted_end = sorted(range(n), key=lambda i: end_probs[i], reverse=True)
print("\nTop 3 ending letters:")
for i in sorted_end[1:4]:
    print(f"  {chars[i]}: {end_probs[i]:.4f}")

print("Bottom 3 ending letters:")
non_zero_end = [i for i in sorted_end if i != 0 and end_probs[i] > 0]
for i in non_zero_end[-3:]:
    print(f"  {chars[i]}: {end_probs[i]:.4f}")

# Letters following 'q'
q_idx = char_to_idx['q']
print("\nLetters following 'q':")
for i in range(n):
    if matrix[q_idx][i] > 0:
        print(f"  '{chars[i]}': {matrix[q_idx][i]} times")

# Most likely second letter for names starting with 'x'
x_idx = char_to_idx['x']
x_row = prob_matrix[x_idx]
most_likely = np.argmax(x_row)
print(f"\nMost likely second letter after 'x': '{chars[most_likely]}' ({x_row[most_likely]:.4f})")
# Fix: most likely second letter after 'x' (excluding end token)
x_row_no_end = x_row.copy()
x_row_no_end[0] = 0  # zero out the '.' column
most_likely_x = np.argmax(x_row_no_end)
print(f"Most likely letter after 'x' (excluding end): '{chars[most_likely_x]}' ({x_row_no_end[most_likely_x]:.4f})")