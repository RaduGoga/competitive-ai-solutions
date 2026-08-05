# Sample submission

Arhiva trebuie să conțină `solution.py` direct în rădăcină.

Funcțiile obligatorii sunt:

```python
def adapt(phase2_train_csv, pretrained_model_dir, output_dir, seed):
    ...

def predict(test_csv, model_dir):
    ...
```

Implementarea furnizată este intenționat minimală:

- nu adaptează efectiv checkpoint-ul Phase 1;
- salvează un model dummy;
- prezice eticheta `0` pentru toate exemplele;
- conține TODO-uri pentru implementarea concurentului.

Pentru trimitere, arhivați conținutul directorului, astfel încât
`solution.py` să fie direct în rădăcina arhivei ZIP.
