from train import Trainer
import json

history = Trainer()
print(history)
with open('history.json','w') as f:
    f.write(history)