from train import Trainer
import json

history = Trainer()
with open('history.json','w') as f:
    f.write(json.dumps(history))