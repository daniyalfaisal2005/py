#Tic Tac Toe
import math

def check_terminal(board):
    win_paths = [
        [board[0][0], board[0][1], board[0][2]], [board[1][0], board[1][1], board[1][2]], [board[2][0], board[2][1], board[2][2]], # Rows
        [board[0][0], board[1][0], board[2][0]], [board[0][1], board[1][1], board[2][1]], [board[0][2], board[1][2], board[2][2]], # Cols
        [board[0][0], board[1][1], board[2][2]], [board[0][2], board[1][1], board[2][0]]  # Diagonals
    ]
    if ['X', 'X', 'X'] in win_paths: return 1
    if ['O', 'O', 'O'] in win_paths: return -1
    if all(cell != ' ' for row in board for cell in row): return 0
    return None

def minimax(board, is_maximizing):
    score = check_terminal(board)
    if score is not None: return score

    if is_maximizing:
        best_score = -math.inf
        for r in range(3):
            for c in range(3):
                if board[r][c] == ' ':
                    board[r][c] = 'X'
                    best_score = max(best_score, minimax(board, False))
                    board[r][c] = ' '
        return best_score
    else:
        best_score = math.inf
        for r in range(3):
            for c in range(3):
                if board[r][c] == ' ':
                    board[r][c] = 'O'
                    best_score = min(best_score, minimax(board, True))
                    board[r][c] = ' '
        return best_score

def get_best_ai_move(board):
    best_score = -math.inf
    best_move = None
    for r in range(3):
        for c in range(3):
            if board[r][c] == ' ':
                board[r][c] = 'X'
                score = minimax(board, False)
                board[r][c] = ' '
                if score > best_score:
                    best_score = score
                    best_move = (r, c)
    print(f"AI chose {best_move} with guaranteed utility of: {best_score}")
    return best_move
def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 9)

def play_game():
    board = [[' ' for _ in range(3)] for _ in range(3)]
    
    print("Welcome to Tic-Tac-Toe! You are 'O' and the AI is 'X'.")
    print_board(board)
    
    while True:
        try:
            print("\nYour turn!")
            row = int(input("Enter row (0, 1, or 2): "))
            col = int(input("Enter column (0, 1, or 2): "))
            
            if board[row][col] != ' ':
                print("That spot is taken! Try again.")
                continue
            board[row][col] = 'O'
            
        except (ValueError, IndexError):
            print("Invalid input. Please enter numbers 0, 1, or 2.")
            continue
            
        print_board(board)
        if check_terminal(board) is not None:
            break
            

        print("\nAI is thinking...")
        best_move = get_best_ai_move(board)
        
        if best_move:
            board[best_move[0]][best_move[1]] = 'X'
            
        print_board(board)
        if check_terminal(board) is not None:
            break

    print("\n--- Game Over ---")
    result = check_terminal(board)
    if result == 1:
        print("The AI Wins! (As expected of Minimax)")
    elif result == -1:
        print("You Win! (Wait, this should be impossible!)")
    else:
        print("It's a Draw! Good game.")

if __name__ == "__main__":
    play_game()


#beysian network 

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# --- 1.1 Define Network Structure (DAG) ---
# Edges: Intelligence→Grade, StudyHours→Grade, Difficulty→Grade, Grade→Pass
bn_model = DiscreteBayesianNetwork([
    ('Intelligence', 'Grade'),
    ('StudyHours',   'Grade'),
    ('Difficulty',   'Grade'),
    ('Grade',        'Pass')
])

# --- 1.2 Prior CPTs for root nodes ---
# Intelligence: 0=High, 1=Low
cpd_intelligence = TabularCPD(
    variable='Intelligence', variable_card=2,
    values=[[0.7], [0.3]]
)

# StudyHours: 0=Sufficient, 1=Insufficient
cpd_studyhours = TabularCPD(
    variable='StudyHours', variable_card=2,
    values=[[0.6], [0.4]]
)

# Difficulty: 0=Easy, 1=Hard
cpd_difficulty = TabularCPD(
    variable='Difficulty', variable_card=2,
    values=[[0.6], [0.4]]
)

# --- 1.3 Conditional CPT for Grade (3 states: 0=A, 1=B, 2=C) ---
# Parents: Intelligence(2) × StudyHours(2) × Difficulty(2) = 8 combinations
# Columns ordered: (I=H,S=Suf,D=Easy),(I=H,S=Suf,D=Hard),(I=H,S=Ins,D=Easy),
#                  (I=H,S=Ins,D=Hard),(I=L,S=Suf,D=Easy),(I=L,S=Suf,D=Hard),
#                  (I=L,S=Ins,D=Easy),(I=L,S=Ins,D=Hard)
cpd_grade = TabularCPD(
    variable='Grade', variable_card=3,
    values=[
        # P(A)
        [0.80, 0.60, 0.60, 0.40, 0.50, 0.30, 0.20, 0.10],
        # P(B)
        [0.15, 0.30, 0.30, 0.40, 0.35, 0.45, 0.45, 0.40],
        # P(C)
        [0.05, 0.10, 0.10, 0.20, 0.15, 0.25, 0.35, 0.50],
    ],
    evidence=['Intelligence', 'StudyHours', 'Difficulty'],
    evidence_card=[2, 2, 2]
)

# --- 1.4 CPT for Pass (0=Yes, 1=No) given Grade ---
cpd_pass = TabularCPD(
    variable='Pass', variable_card=2,
    values=[
        [0.95, 0.80, 0.50],   # P(Pass=Yes | Grade=A/B/C)
        [0.05, 0.20, 0.50],   # P(Pass=No  | Grade=A/B/C)
    ],
    evidence=['Grade'], evidence_card=[3]
)

# --- 1.5 Add CPDs and validate ---
bn_model.add_cpds(cpd_intelligence, cpd_studyhours, cpd_difficulty,
                  cpd_grade, cpd_pass)
assert bn_model.check_model(), "BN model is invalid!"

# --- 1.6 Inference: Variable Elimination ---
inference = VariableElimination(bn_model)

print("=" * 60)
print("BAYESIAN NETWORK — Student Performance")
print("=" * 60)

# Query 1: P(Pass | StudyHours=Sufficient, Difficulty=Hard)
# StudyHours: 0=Sufficient; Difficulty: 1=Hard
q1 = inference.query(variables=['Pass'],
                     evidence={'StudyHours': 0, 'Difficulty': 1})
print("\nQ1: P(Pass | StudyHours=Sufficient, Difficulty=Hard)")
print(q1)

# Query 2: P(Intelligence | Pass=Yes)
q2 = inference.query(variables=['Intelligence'], evidence={'Pass': 0})
print("\nQ2: P(Intelligence | Pass=Yes)")
print(q2)

#K-means clustering

#importing libraries
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Importing the dataset
df = pd.read_csv('Mall_Customers.csv')
print(df.head())

# Extracting Variables
x = df.iloc[:, [3, 4]].values

#finding optimal number of clusters using the elbow method
from sklearn.cluster import KMeans
wcss_list= [] #Initializing the list for the values of WCSS

#Using for loop for iterations from 1 to 10.
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_)
mtp.plot(range(1, 11), wcss_list)
mtp.title('The Elobw Method Graph')
mtp.xlabel('Number of clusters(k)')
mtp.ylabel('wcss_list')
mtp.show()

# Normalize features for better clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

#training the K-means model on a dataset
kmeans = KMeans(n_clusters=5, init='k-means++', random_state= 42)
y_predict= kmeans.fit_predict(X_scaled)

#visulaizing the clusters
mtp.scatter(x[y_predict == 0, 0], x[y_predict == 0, 1], s = 100, c ='blue',   label = 'Cluster 1') #for first cluster
mtp.scatter(x[y_predict == 1, 0], x[y_predict == 1, 1], s = 100, c ='green',  label = 'Cluster 2') #for second cluster
mtp.scatter(x[y_predict == 2, 0], x[y_predict == 2, 1], s = 100, c ='red',    label = 'Cluster 3') #for third cluster
mtp.scatter(x[y_predict == 3, 0], x[y_predict == 3, 1], s = 100, c ='black',  label = 'Cluster 4') #for fourth cluster
mtp.scatter(x[y_predict == 4, 0], x[y_predict == 4, 1], s = 100, c ='purple', label = 'Cluster 5') #for fifth cluster

mtp.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroid')

mtp.title('Clusters of customers')
mtp.xlabel('Annual Income (k$)')
mtp.ylabel('Spending Score (1-100)')
mtp.legend()
mtp.show()


#minimax algorithm 

import math

class Node:
    def __init__(self, value=None):
        self.value = value
        self.children = []

def minimax(node, depth, maximizing_player):
    # Base case: leaf node or depth limit
    if depth == 0 or not node.children:
        return node.value

    if maximizing_player:
        max_eval = -math.inf
        for child in node.children:
            eval = minimax(child, depth - 1, False)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = math.inf
        for child in node.children:
            eval = minimax(child, depth - 1, True)
            min_eval = min(min_eval, eval)
        return min_eval


# -------- Tree Construction --------
root = Node('A')
n1 = Node('B')
n2 = Node('C')

root.children = [n1, n2]

n3 = Node('D')
n4 = Node('E')
n5 = Node('F')
n6 = Node('G')

n1.children = [n3, n4]
n2.children = [n5, n6]

# Leaf nodes (numeric values)
n7 = Node(2)
n8 = Node(3)
n9 = Node(5)
n10 = Node(9)

n3.children = [n7, n8]
n4.children = [n9, n10]

n11 = Node(0)
n12 = Node(1)
n13 = Node(7)
n14 = Node(5)

n5.children = [n11, n12]
n6.children = [n13, n14]

# -------- Run Minimax --------
depth = 3
result = minimax(root, depth, True)

print("Minimax value for root:", result)


# Alpha beta pruning W/ agent and Environment

import math

class Node:
    def __init__(self, value=None):
        self.value = value
        self.children = []
        self.minmax_value = None

class MinimaxAgent:
    def __init__(self, depth):
        self.depth = depth

    def act(self, node, environment):
        value = environment.alpha_beta_search(node, self.depth, -math.inf, math.inf, True)
        node.minmax_value = value
        return f"Minimax value for root node: {value}"

class Environment:
    def __init__(self, tree):
        self.tree = tree
        self.computed_nodes = []

    def get_percept(self, node):
        return node

    def alpha_beta_search(self, node, depth, alpha, beta, maximizing_player):
        # Track visited nodes
        self.computed_nodes.append(node.value)

        # Base case (leaf node or depth limit)
        if depth == 0 or not node.children:
            return node.value

        if maximizing_player:
            value = -math.inf
            for child in node.children:
                value = max(value, self.alpha_beta_search(child, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)

                if beta <= alpha:
                    print("Pruned node:", child.value)
                    break

        else:
            value = math.inf
            for child in node.children:
                value = min(value, self.alpha_beta_search(child, depth - 1, alpha, beta, True))
                beta = min(beta, value)

                if beta <= alpha:
                    print("Pruned node:", child.value)
                    break

        node.minmax_value = value
        return value


def run_agent(agent, environment, start_node):
    percept = environment.get_percept(start_node)
    print(agent.act(percept, environment))


# Constructing the tree
root = Node('A')
n1 = Node('B')
n2 = Node('C')

root.children = [n1, n2]

n3 = Node('D')
n4 = Node('E')
n5 = Node('F')
n6 = Node('G')

n1.children = [n3, n4]
n2.children = [n5, n6]

# Leaf nodes (numeric values)
n7 = Node(2)
n8 = Node(3)
n9 = Node(5)
n10 = Node(9)

n3.children = [n7, n8]
n4.children = [n9, n10]

n11 = Node(0)
n12 = Node(1)
n13 = Node(7)
n14 = Node(5)

n5.children = [n11, n12]
n6.children = [n13, n14]

# Run
depth = 3
agent = MinimaxAgent(depth)
environment = Environment(root)

run_agent(agent, environment, root)

print("Computed Nodes:", environment.computed_nodes)

print("\nMinimax values:")
print(f"A: {root.minmax_value}")
print(f"B: {n1.minmax_value}")
print(f"C: {n2.minmax_value}")
print(f"D: {n3.minmax_value}")
print(f"E: {n4.minmax_value}")
print(f"F: {n5.minmax_value}")
print(f"G: {n6.minmax_value}")


#alpha beta pruning W/o agent and Environment

import math

class Node:
    def __init__(self, value=None):
        self.value = value
        self.children = []

def alpha_beta(node, depth, alpha, beta, maximizing_player):
    # Base case: leaf node or depth limit
    if depth == 0 or not node.children:
        return node.value

    if maximizing_player:
        value = -math.inf
        for child in node.children:
            value = max(value, alpha_beta(child, depth - 1, alpha, beta, False))
            alpha = max(alpha, value)

            if beta <= alpha:
                print("Pruned at node:", child.value)
                break
        return value

    else:
        value = math.inf
        for child in node.children:
            value = min(value, alpha_beta(child, depth - 1, alpha, beta, True))
            beta = min(beta, value)

            if beta <= alpha:
                print("Pruned at node:", child.value)
                break
        return value


# -------- Tree Construction --------
root = Node('A')
n1 = Node('B')
n2 = Node('C')

root.children = [n1, n2]

n3 = Node('D')
n4 = Node('E')
n5 = Node('F')
n6 = Node('G')

n1.children = [n3, n4]
n2.children = [n5, n6]

# Leaf nodes (numeric values)
n7 = Node(2)
n8 = Node(3)
n9 = Node(5)
n10 = Node(9)

n3.children = [n7, n8]
n4.children = [n9, n10]

n11 = Node(0)
n12 = Node(1)
n13 = Node(7)
n14 = Node(5)

n5.children = [n11, n12]
n6.children = [n13, n14]

# -------- Run Alpha-Beta --------
depth = 3
result = alpha_beta(root, depth, -math.inf, math.inf, True)

print("\nAlpha-Beta result (Minimax value):", result)

#EDA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Replace 'your_dataset.csv' with your actual filename
file_path = 'Iris.csv' 
df = pd.read_csv(file_path)

# 1. Records attributes data types
print("--- 1. Data Types ---")
print(df.info())

# 2. Missing inconsistent abnormal value
print("\n--- 2. Missing Values ---")
print(df.isnull().sum())
print("\n--- Summary Statistics (Check for Abnormalities) ---")
print(df.describe())

# 3. Relationship b/w attributes (Correlation Matrix)
print("\n--- 3. Correlation Matrix ---")
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Relationship b/w Attributes")
plt.show()

# 4. Pattern distribution relationship using plots
print("\n--- 4. Distributions and Pairplots ---")
df.hist(figsize=(10,8)) 
plt.suptitle("Feature Distributions")
plt.show()

# Automated pairplot using the last column as category if applicable
target_col = df.columns[-1]
sns.pairplot(df, hue=target_col) 
plt.show()

# 5. Handle missing or inconsistent data
# Filling numeric missing values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)
# Removing duplicate records
df.drop_duplicates(inplace=True)

# 6. Transform numerical to categorical & vice versa
# Example: Label Encoding the target if it is categorical
le = LabelEncoder()
if df[target_col].dtype == 'object':
    df[target_col + '_encoded'] = le.fit_transform(df[target_col])
    print(f"\n--- 6. Encoded {target_col} to {target_col}_encoded ---")

# 7. Normalization 
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)
print("\n--- 7. Normalized Data (First 5 rows) ---")
print(df_scaled.head())

# 8. Check dataset suffer from imbalance
print("\n--- 8. Class Balance ---")
print(df[target_col].value_counts())

# 9. Remove outliers (Using IQR on the first numeric column)
sample_col = numeric_cols[0] 
Q1 = df[sample_col].quantile(0.25)
Q3 = df[sample_col].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[sample_col] < (Q1 - 1.5 * IQR)) | (df[sample_col] > (Q3 + 1.5 * IQR)))]
print(f"\n--- 9. Outliers removed based on {sample_col} ---")

# 10. Generate two meaningful features
if len(numeric_cols) >= 2:
    # Feature 1: Ratio between two attributes
    df['feature_ratio'] = df[numeric_cols[0]] / (df[numeric_cols[1]] + 1e-5)
    # Feature 2: Sum of two attributes
    df['feature_combined'] = df[numeric_cols[0]] + df[numeric_cols[1]]
    print("\n--- 10. New Features Generated: 'feature_ratio', 'feature_combined' ---")

# 11. Compute and compare statistical analysis 
print("\n--- 11. Statistical Analysis (Mean values grouped by target) ---")
print(df.groupby(target_col).mean(numeric_only=True))

# SVM + Confusion matrix 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Load Dataset
df = pd.read_csv('Iris.csv')

# 2. Preprocessing: Drop unnecessary columns and encode target
if 'Id' in df.columns:
    df.drop(inplace=True, axis=1, columns='Id')

# Find categorical columns and apply Label Encoding
cat_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# 3. Split Dataset into Features (X) and Target (y)
y = df.pop("Species")
X = df

# 4. Split into Training and Testing sets[cite: 4]
# test_size=0.2 means 20% of data is used for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Initialize and Train SVM Model[cite: 4]
# Using RBF kernel with standard parameters[cite: 4]
svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train, y_train)

# 6. Make Predictions[cite: 4]
y_pred = svm_model.predict(X_test)

# 7. Evaluation[cite: 4]
print("SVM Testing Accuracy:", accuracy_score(y_test, y_pred))

# ================= CONFUSION MATRIX =================[cite: 4]
print('\n===================== Confusion Matrix =================')
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize Confusion Matrix using Seaborn[cite: 4]
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()

# ================= FULL REPORT =================[cite: 4]
print('\n===================== Classification Report =================')
print(classification_report(y_test, y_pred))

#svm + Confusion matrix gernalized

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Load your dataset
# Replace 'Dataset.csv' with your actual filename
df = pd.read_csv('Iris.csv') 

# 2. Preprocessing
# Identify categorical columns and transform them to numerical for the model
cat_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# 3. Define Features (X) and Target (y)
# Assuming the last column is the target variable
target_column = df.columns[-1]
y = df[target_column]
X = df.drop(target_column, axis=1)

# 4. Split into Training and Testing sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Initialize and Train SVM Model
# Using the Radial Basis Function (RBF) kernel as a standard default
svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train, y_train)

# 6. Make Predictions on the test set
y_pred = svm_model.predict(X_test)

# 7. Evaluate Performance
print(f"SVM Testing Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# ================= CONFUSION MATRIX =================
print('\n===================== Confusion Matrix =================')
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize the Confusion Matrix using a Heatmap for better clarity
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.title('SVM Classification: Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()

# ================= CLASSIFICATION REPORT =================
# Provides detailed metrics: Precision, Recall, and F1-Score
print('\n===================== Classification Report =================')
print(classification_report(y_test, y_pred))

#csp
# To run: pip install python-constraint
from constraint import Problem

def solve_csp():
    problem = Problem()

    # 1. IDENTIFY VARIABLES (The 7 regions of Australia)
    regions = ["WA", "NT", "Q", "NSW", "V", "SA", "T"]
    
    # 2. IDENTIFY DOMAINS (Available colors)
    colors = ["red", "green", "blue"]
    
    # Add variables and their domains to the problem
    problem.addVariables(regions, colors)

    # 3. IDENTIFY CONSTRAINTS (Adjacency list)
    # Binary constraints: neighboring regions cannot have the same color
    adjacencies = [
        ("WA", "NT"), ("WA", "SA"),
        ("SA", "NT"), ("SA", "Q"), ("SA", "NSW"), ("SA", "V"),
        ("NT", "Q"), ("Q", "NSW"), ("NSW", "V")
    ]

    for (region1, region2) in adjacencies:
        # Constraint: value of region1 != value of region2
        problem.addConstraint(lambda x, y: x != y, (region1, region2))

    # Solve and display the first valid result
    solutions = problem.getSolutions()
    print(f"Total valid colorings found: {len(solutions)}")
    print("Example Solution:", solutions[0])

solve_csp()

#CSP 2
# Simple CSP Solver using Backtracking

def is_consistent(variable, value, assignment, constraints):
    for neighbor in constraints[variable]:
        if neighbor in assignment and assignment[neighbor] == value:
            return False
    return True


def backtrack(assignment, variables, domains, constraints):
    # If all variables assigned → solution found
    if len(assignment) == len(variables):
        return assignment

    # Select unassigned variable
    unassigned = [v for v in variables if v not in assignment]
    var = unassigned[0]

    for value in domains[var]:
        if is_consistent(var, value, assignment, constraints):
            assignment[var] = value

            result = backtrack(assignment, variables, domains, constraints)
            if result:
                return result

            # Backtrack
            del assignment[var]

    return None


# -------- Example: Map Coloring --------

variables = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']

domains = {
    'WA': ['Red', 'Green', 'Blue'],
    'NT': ['Red', 'Green', 'Blue'],
    'SA': ['Red', 'Green', 'Blue'],
    'Q':  ['Red', 'Green', 'Blue'],
    'NSW':['Red', 'Green', 'Blue'],
    'V':  ['Red', 'Green', 'Blue'],
    'T':  ['Red', 'Green', 'Blue']
}

# Neighbor constraints (no same color)
constraints = {
    'WA': ['NT', 'SA'],
    'NT': ['WA', 'SA', 'Q'],
    'SA': ['WA', 'NT', 'Q', 'NSW', 'V'],
    'Q':  ['NT', 'SA', 'NSW'],
    'NSW':['Q', 'SA', 'V'],
    'V':  ['SA', 'NSW'],
    'T':  []
}

# Solve
solution = backtrack({}, variables, domains, constraints)

print("CSP Solution:")
print(solution)
