import nbformat

nb_path = '/Users/prakhar/Coding/stock-predictor/notebooks/05_MLP.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

code = """# Recreate Train/Test Split from 03_models
features = pd.read_csv(os.path.join(data_dir, 'features.csv'), index_col=0, parse_dates=True)
target = pd.read_csv(os.path.join(data_dir, 'target.csv'), index_col=0, parse_dates=True).squeeze()

feature_names = features.columns.tolist()
X = features.replace([np.inf, -np.inf], np.nan)
y = target

data = X.join(y.rename('target')).dropna()
X = data[feature_names]
y = data['target']

n = int(len(data) * 0.8)
X_train, X_test = X.iloc[:n], X.iloc[n:]
y_train, y_test = y.iloc[:n], y.iloc[n:]

print(f"Train: {X_train.index.min()} → {X_train.index.max()}")
print(f"Test:  {X_test.index.min()} → {X_test.index.max()}")
"""

new_cell = nbformat.v4.new_code_cell(source=code)
nb.cells.append(new_cell)

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print("Successfully added the cell.")
