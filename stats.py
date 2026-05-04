with open('names.txt', 'r') as f:
    names = f.read().splitlines()

print(f"Total names: {len(names)}")
print("First 10:", names[:10])
