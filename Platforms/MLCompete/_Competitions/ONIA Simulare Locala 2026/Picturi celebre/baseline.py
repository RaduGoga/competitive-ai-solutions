import pandas as pd
import numpy as np

# ============================================================
# STARTER KIT / BASELINE SUBMISSION
# ------------------------------------------------------------
# This script generates a VALID submission file (dummy_output.csv)
# that respects the required format for all three subtasks.
#
# IMPORTANT:
# - This is NOT a competitive solution
# - It is meant as a sanity check / starter template
# - Participants are expected to improve all three tasks
# ============================================================


# =========================
# Load test data
# =========================
# The test set contains all features EXCEPT target_price
test = pd.read_csv("test.csv")

# Try loading train.csv in order to compute a reasonable
# constant baseline price for Task 3.
# If train.csv is missing, fallback to 0.
try:
    train = pd.read_csv("train.csv")
    default_price = int(train["target_price"].mean())
except:
    default_price = 0

n_samples = len(test)


# =========================
# Task 1 – Artistic Authenticity Score (Dummy)
# =========================
# Baseline strategy:
# - Label ALL paintings as "Incert"
# - This will NOT score well, but ensures correct formatting
task1_answers = np.array(["Incert"] * n_samples)


# =========================
# Task 2 – Painter Style Grouping (Dummy)
# =========================
# Baseline strategy:
# - Assign ALL paintings to the same cluster (ID = 0)
# - This yields a very low ARI score, but is valid
task2_answers = np.array([0] * n_samples)


# =========================
# Task 3 – Price Estimation (Dummy)
# =========================
# Baseline strategy:
# - Predict a constant price for all paintings
# - Uses the mean target_price from train.csv (if available)
task3_answers = np.array([default_price] * n_samples)


# =========================
# Build submission file
# =========================
# Required format:
# SampleID | subtaskID | Answer
#
# Each SampleID must appear exactly 3 times:
# - once for Task1
# - once for Task2
# - once for Task3
submission = pd.DataFrame({
    "SampleID": np.concatenate([
        test["SampleID"],
        test["SampleID"],
        test["SampleID"]
    ]),
    "subtaskID": (
        ["Task1"] * n_samples +
        ["Task2"] * n_samples +
        ["Task3"] * n_samples
    ),
    "Answer": np.concatenate([
        task1_answers,
        task2_answers,
        task3_answers
    ])
})


# =========================
# Save output file
# =========================
submission.to_csv("dummy_output.csv", index=False)

print("This is a starter baseline. Improve each task for better scores.")
