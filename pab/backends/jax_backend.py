# jax_backend.py - Actual content from coding session
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
from loguru import logger
from tqdm import tqdm

class JaxModel(nn.Module):
    """A simple JAX model for demonstration."""
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x

def create_train_state(model, learning_rate=0.001):
    """Create a train state for JAX model training."""
    params = model.init(jax.random.PRNGKey(0), jnp.ones([1, 224, 224, 3]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def train_jax_model(state, train_dataset, val_dataset, epochs=10):
    """Train a JAX model while tracking learning progression."""
    logger.info("Starting JAX training...")

    for epoch in range(epochs):
        for images, labels in tqdm(train_dataset, desc=f"Epoch {epoch+1}/{epochs}"):
            def loss_fn(params):
                logits = state.apply_fn({'params': params}, images)
                loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=jax.nn.one_hot(labels, 10)))
                return loss

            grads = jax.grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)

        logger.info(f"Epoch {epoch+1} complete.")

    return state