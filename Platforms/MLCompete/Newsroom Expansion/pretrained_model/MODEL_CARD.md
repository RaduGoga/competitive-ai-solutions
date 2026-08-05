# Model Phase 1 furnizat

Modelul este un clasificator liniar cu funcție *softmax*, aplicat peste o reprezentare textuală cu 2048 caracteristici obținute prin *feature hashing*.

A fost antrenat numai pe clasele:

- `0` — World;
- `1` — Sports.

Participanții nu primesc exemplele folosite în Phase 1. Primesc numai checkpoint-ul și statistici agregate necesare unor metode de stabilizare.

Fișierul `state.pkl` conține:

| Câmp | Semnificație |
| --- | --- |
| `W`, `b` | parametrii modelului la sfârșitul Phase 1 |
| `theta_W`, `theta_b` | copia parametrilor de referință Phase 1 |
| `fisher_W`, `fisher_b` | aproximația diagonală a informației Fisher |

Performanța *macro-F1* pe setul privat de validare Phase 1 este `0.842475`.

SHA-256 pentru `state.pkl`:

```text
c47eb8773dc790d0d034dd9b6879f8f241357afd27b1f2f68757ee15652c41f1
```

Reprezentarea textului este descrisă integral în `starter_submission/common.py` și în `meta.json`.
