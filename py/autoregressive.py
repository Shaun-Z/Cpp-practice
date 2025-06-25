# %%
import torch
import torch.nn as nn

from labml import experiment
from labml.configs import option
from labml_nn.transformers.basic.autoregressive_experiment import Configs

# %%
experiment.create(name="transformer", writers={'screen'})

# %%
conf = Configs()

# %%
experiment.configs(conf, {
    # Use character level tokenizer
    'tokenizer': 'character',
    # Prompt separator is blank
    'prompt_separator': '',
    # Starting prompt for sampling
    'prompt': 'It is ',
    # Use Tiny Shakespeare dataset
    'text': 'tiny_shakespeare',

    # Use a context size of $256$
    'seq_len': 512,
    # Train for 32 epochs
    'epochs': 32,
    # Batch size $32$
    'batch_size': 16,
    # Switch between training and validation for $10$ times
    # per epoch
    'inner_iterations': 10,

    # Model size
    'd_model': 256,
    'transformer.n_heads': 16,
    'transformer.ffn.d_ff': 1024,

    # Use [Noam optimizer](../../optimizers/noam.html)
    'optimizer.optimizer': 'Noam',
    'optimizer.learning_rate': 1.,
})

# %%
experiment.add_pytorch_models({'model': conf.model})

# %%
with experiment.start():
    conf.run()


