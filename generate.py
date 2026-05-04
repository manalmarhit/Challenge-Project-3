import numpy as np

# Read names
with open('names.txt', 'r') as f:
    names = f.read().splitlines()

# Setup
chars = ['.'] + list('abcdefghijklmnopqrstuvwxyz')
n = len(chars)
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# Build transition matrix
matrix = np.zeros((n, n), dtype=int)
for name in names:
    word = '.' + name + '.'
    for ch1, ch2 in zip(word, word[1:]):
        matrix[char_to_idx[ch1]][char_to_idx[ch2]] += 1

# Convert to probabilities
prob_matrix = matrix / matrix.sum(axis=1, keepdims=True)

# Generate names
def generate_name():
    name = ''
    letter = '.'
    while True:
        row = prob_matrix[char_to_idx[letter]]
        letter = np.random.choice(chars, p=row)
        if letter == '.':
            break
        name += letter
    return name

# Generate 25 names, reject if fewer than 3 letters
generated = []
while len(generated) < 25:
    name = generate_name()
    if len(name) >= 3:
        generated.append(name)

print("Generated names:")
for name in generated:
    print(f"  {name}")