# Large synthetic curriculum: 200,000 scenes

The generator `scripts/build_large_scene_curriculum.py` streams **200,000**
validated bilingual scene records to disk without retaining them in RAM.

```bash
python scripts/build_large_scene_curriculum.py \
  --count 200000 --output datasets/python_scene_large
```

This generates train/validation/test JSONL files and a manifest. Topic families,
not individual rows, determine the split. **No 200k data files were committed to
GitHub**, and the generator has not been run as part of this commit.

## Training: conservative baseline first

```bash
python -m voice2reel.train_python_scene \
  --train datasets/python_scene_large/train.jsonl \
  --validation datasets/python_scene_large/validation.jsonl \
  --output outputs/qwen_python_scene_large \
  --epochs 3 --patience 2
```

An upper-bound experiment can set `--epochs 50`, but **50 epochs is not the
recommended default**. Use validation early stopping, monitor overfitting,
token budgets, training throughput, and GPU cost. No GPU training has been run
and no Qwen adapter is supplied.

## Important quality caveat

The generator produces structurally valid examples with variable gestures,
framing, durations, shot counts, topics and Arabic/English narration. It is a
**synthetic bootstrap dataset, not 200,000 unique artist-authored, cinematic
scenes**. Much of its prose is repetitive and its gestures are randomly assigned
rather than semantically choreographed. Increasing the count does not
automatically improve model quality and can hurt generalization.

Before committing to full training, add:
- Artist-authored demonstrations with meaningful scene choreography.
- Audio-duration-aware scene planning and explicit caption safe-zone tests.
- Distinct narrative structures, code visualizations and transition behavior.
- Model-generated hard examples, correction traces, and human preference ratings.
- Executed render tests across all 12 locked characters and A/V synchronization.
- Evaluation on **unseen topics and prompts**, including a base-model baseline.

Training data, checkpoints, and rendered videos should live in object storage
or artifact storage, not in Git history. Dataset size in GB is not a quality
metric.
