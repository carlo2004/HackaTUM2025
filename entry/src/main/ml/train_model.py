import pandas as pd
import m2cgen as m2c
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, GridSearchCV

# ==========================================
# 1. PREPARE DATA (Same as before)
# ==========================================

# Features: [std_dev_z, min_val, rescuer_hr]
# Class 0: GOOD, 1: LEANING, 2: TIRED
data = [
    # --- CLASS 0: GOOD FORM ---
    [8.5, 0.4, 90,  0], [8.2, 0.3, 95,  0], [8.8, 0.5, 100, 0], [9.0, 0.2, 85,  0],
    [8.4, 0.4, 92,  0], [8.3, 0.3, 94,  0], [8.9, 0.5, 99,  0], [9.1, 0.2, 88,  0],

    # --- CLASS 1: LEANING ---
    [8.5, 4.5, 95,  1], [8.2, 5.1, 92,  1], [8.0, 3.8, 98,  1], [8.4, 4.2, 96,  1],
    [8.1, 4.9, 91,  1], [7.9, 3.9, 97,  1],

    # --- CLASS 2: TIRED / WEAK ---
    [7.5, 0.5, 155, 2], [6.0, 0.4, 160, 2], [2.5, 0.4, 110, 2], [1.8, 0.3, 115, 2],
    [7.2, 0.6, 152, 2], [6.3, 0.4, 158, 2], [2.8, 0.5, 112, 2], [1.9, 0.3, 118, 2],
]

df = pd.DataFrame(data, columns=['std_dev_z', 'min_val', 'rescuer_hr', 'label'])
X = df[['std_dev_z', 'min_val', 'rescuer_hr']]
y = df['label']

# Split Hold-out Test Set (Strictly for final evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. GENERATE ALPHAS (The "Path")
# ==========================================

# Grow the full tree T0
clf = DecisionTreeClassifier(random_state=42)

# AUTOMATICALLY find the sequence of alphas where the tree changes
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Remove the last alpha (which prunes the tree to a single root node)
ccp_alphas = ccp_alphas[:-1]

print(f"Found {len(ccp_alphas)} potential alphas for pruning.")

# ==========================================
# 3. K-FOLD CROSS VALIDATION
# ==========================================

# We use GridSearchCV to test every single Alpha with 5-Fold CV
# This replaces the need to write your own loop.
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid={'ccp_alpha': ccp_alphas},
    cv=5,           # 5-Fold Cross Validation
    scoring='accuracy'
)

print("Running 5-Fold Cross Validation on all alphas...")
grid_search.fit(X_train, y_train)

# ==========================================
# 4. GET THE WINNER
# ==========================================

best_clf = grid_search.best_estimator_
best_alpha = grid_search.best_params_['ccp_alpha']

print(f"\nWINNER FOUND:")
print(f"Best Alpha: {best_alpha:.5f}")
print(f"CV Score:   {grid_search.best_score_:.2f}")
print(f"Test Score: {best_clf.score(X_test, y_test):.2f}")
print(f"Leaves:     {best_clf.get_n_leaves()}")

# ==========================================
# 5. EXPORT LOGIC
# ==========================================

print("\n--- RULES ---")
print(export_text(best_clf, feature_names=['std_dev_z', 'min_val', 'rescuer_hr']))

print("\n--- ARKTS CODE ---")
js_code = m2c.export_to_javascript(best_clf)

print("private static runDecisionTree(input: number[]): number {")
print(js_code)
print("}")