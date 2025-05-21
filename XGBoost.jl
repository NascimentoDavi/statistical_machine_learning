using Pkg
Pkg.add("XGBoost")

# Sample data (features and labels)
X = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0]  # 4 samples, 2 features each
y = [0, 0, 1, 1]                          # Labels for binary classification

# Create DMatrix from data
dtrain = DMatrix(X, label=y)

# Set parameters for classification
params = Dict("objective" => "binary:logistic", "eval_metric" => "logloss")

# Train XGBoost model
num_round = 10
bst = xgboost(dtrain, num_round, params)

# Predict
preds = predict(bst, dtrain)
println("Predictions: ", preds)