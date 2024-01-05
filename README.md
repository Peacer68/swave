## SWave: Improving Vocoder Efficiency by Straightening the Waveform Generation Path

> Official Implementation of SWave. Submission to ICME 2024, under review.

| ![workflow.png](E:\研究生\研二\ICME\swave\misc\workflow.png) |
| :----------------------------------------------------------- |
| *Training Workflow of SWave. During rectification, we randomly sample some noises and speeches to construct the data pairs $(X_0,X_1)$, and then apply the operator $K$ times to straighten the generation path from noise to speech. During distillation, we utilize the data pairs constructed by the $F$-step VFE in the $K$-th operator to distill an $N$-step VFE. Finally, we fine-tune the $N$-step VFE with the ground truth and obtain $N$-step SWave. $F$ is generally set to 1,000, and $N\ll F$.* |

### Preparation

[LJSpeech Dataset](https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2)

Some pretrained checkpoint for test:

|      |      |
| ---- | ---- |
|      |      |
|      |      |
|      |      |



### Stage 1: Rectification

First get the 1st $F$-step VFE:

```bash
bash runs/reflow.sh
```

Then construct new data pairs with the trained model:

```bash
bash runs/re-pair.sh
```

Next retrain the 1st $F$-step VFE:

```bash
bash runs/reflow-k.sh # k>1
```

Repeat the operation several times to straighten the waveform generation path.

### Stage 2: Distillation

Construct new data pairs with the last $F$-step VFE:

```bash
bash runs/re-pair.sh
```

Distill:

```bash
bash runs/distill.sh
```

### Stage 3: Fine-tuning

There's no need to construct new data pairs. Just fine-tune directly:

```bash
bash runs/finetune.sh
```

### Inference:

You can modify the `configs/inference.json` to set the number of generation steps and the generative model. To generate all the speech samples in test set, run:

```bash
bash runs/swave_inference.sh
```





