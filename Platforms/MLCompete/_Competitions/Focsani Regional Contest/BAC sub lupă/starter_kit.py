import pandas as pd
import numpy as np

np.random.seed(42)

# =====================
# Load data
# =====================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

submission = []

# ==========================
# CERINTA 1 – materie random
# ==========================
materii_posibile = train["materie"].unique()
materie_random = np.random.choice(materii_posibile)

submission.append({
    "id": 1,
    "subtaskID": 1,
    "answer": materie_random
})

# =====================
# CERINTA 2 – an random
# =====================
ani_posibili = train["an"].unique()
an_random = int(np.random.choice(ani_posibili))

submission.append({
    "id": 2,
    "subtaskID": 2,
    "answer": an_random
})

# =====================
# CERINTA 3 & 4 – complet random
# =====================
for _, row in test.iterrows():
    # medie random (nota de bac)
    medie_random = np.round(np.random.uniform(1, 10), 2)

    # anomalie random
    anomalie_random = np.random.choice([0, 1])

    submission.append({
        "id": row["id"],
        "subtaskID": 3,
        "answer": medie_random
    })

    submission.append({
        "id": row["id"],
        "subtaskID": 4,
        "answer": anomalie_random
    })

# =====================
# Save submission
# =====================
submission_df = pd.DataFrame(submission)
submission_df.to_csv("sample_output.csv", index=False)
