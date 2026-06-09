import nbformat

nb_path = '/Users/prakhar/Coding/stock-predictor/notebooks/05_MLP.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

code = """# Convert pandas dataframes to PyTorch tensors
X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

print(f"X_train_t shape: {X_train_t.shape}")
print(f"X_test_t shape: {X_test_t.shape}")
print(f"y_train_t shape: {y_train_t.shape}")
print(f"y_test_t shape: {y_test_t.shape}")
"""

new_cell = nbformat.v4.new_code_cell(source=code)
nb.cells.append(new_cell)

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print("Successfully added the tensor conversion cell.")
