This folder contains the trained model and its corresponding class.

Remember to include the model class provided in [model.py](https://github.com/adityabhongade/LightSeg/blob/main/Model/model.py) into your code where you are loading the [trained model](https://github.com/adityabhongade/LightSeg/blob/main/Model/Fine_Tuned.pth).

**Model Details** :

| **Attribute** | **Value** |
|----------|----------|
| Parameters    | 5.8 M     |
| FLOPS    | 14.86 G    |
| Parameter Size    | 23 MB     |
| Input Shape      | (B, 3, 224, 224)  |
| Output Shape      | (B, 1, 224, 224)  |

**Performance** :

| **Metric** | **Value** |
|----------|----------|
| Accuracy    | 97.89     |
| Precision    | 96.75      |
| Recall    | 94.99     |
| IoU      | 92.05  |
| Dice     | 97.89   |
