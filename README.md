## SWave: Improving Vocoder Efficiency by Straightening the Waveform Generation Path

> Official Implementation of *[SWave: Improving Vocoder Efficiency by Straightening the Waveform Generation Path](https://link.springer.com/chapter/10.1007/978-3-031-78172-8_26)* (ICPR'24)

| ![workflow.png](misc/workflow.png)                           |
| :----------------------------------------------------------- |
| *Training Workflow of SWave. During rectification, we randomly sample some noises and speeches to construct the data pairs $(X_0,X_1)$, and then apply the operator $K$ times to straighten the generation path from noise to speech. During distillation, we utilize the data pairs constructed by the $F$-step VFE in the $K$-th operator to distill an $N$-step VFE. Finally, we fine-tune the $N$-step VFE with the ground truth and obtain $N$-step SWave. $F$ is generally set to 1,000, and $N\ll F$.* |

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

### Links

[LJSpeech Dataset](https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2)


Some pretrained checkpoints for test: [2-step SWave](https://drive.google.com/file/d/1yj8xSKtrIS3cfvNhnC7Y3f7vYIFNAxYQ/view?usp=drive_link), [10-step SWave](https://drive.google.com/file/d/13X4TyO5VY3Gu6pC93DOctI5YJNaza_Pj/view?usp=sharing).

### Citation
Please add the citation if our paper or code helps you.
```tex
@inproceedings{liu2025swave,
  title={SWave: Improving Vocoder Efficiency by Straightening the Waveform Generation Path},
  author={Liu, Pan and Zhou, Jianping and Tian, Xiaohua and Lin, Zhouhan},
  booktitle={International Conference on Pattern Recognition},
  pages={397--408},
  year={2025},
  organization={Springer}
}
```



