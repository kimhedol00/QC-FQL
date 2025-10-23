from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
import jax.tree_util # <<< CHANGED: jax.tree_util 임포트 추가

def get_size(data):
    """Return the size of the dataset."""
    # sizes = tree_map(lambda arr: len(arr), data)
    # return max(jax.tree_util.tree_leaves(sizes))
    # MODIFIED: data가 딕셔너리일 때도 올바르게 동작하도록 수정
    if isinstance(data, dict):
        sizes = tree_map(lambda arr: len(arr), data)
        return max(jax.tree_util.tree_leaves(sizes))
    return len(data)


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


def tree_map(f, tree):
    # If it's a list, apply the function recursively to each item
    if isinstance(tree, list):
        return [tree_map(f, item) for item in tree]
    
    # If it's a tuple, apply the function recursively to each item
    elif isinstance(tree, tuple):
        return tuple(tree_map(f, item) for item in tree)
    
    # If it's a dictionary, apply the function to both keys and values recursively
    elif isinstance(tree, dict):
        return {key: tree_map(f, value) for key, value in tree.items()}
    
    # Otherwise, apply the function to the element itself (base case)
    else:
        return f(tree)

@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=False, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            def _freeze(arr):
                if hasattr(arr, 'setflags'):
                    arr.setflags(write=False)
                return arr
            tree_map(_freeze, data)
            # tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        self.return_next_actions = False  # Whether to additionally return next actions; set outside the class.

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        if self.frame_stack is not None:
            # Stack frames.
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
            next_obs = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(tree_map(lambda arr: arr[cur_idxs], self['observations']))
                if i != self.frame_stack - 1:
                    next_obs.append(tree_map(lambda arr: arr[cur_idxs], self['observations']))
            next_obs.append(tree_map(lambda arr: arr[idxs], self['next_observations']))

            batch['observations'] = tree_map(lambda *args: np.concatenate(args, axis=-1), obs)
            batch['next_observations'] = tree_map(lambda *args: np.concatenate(args, axis=-1), *next_obs)
        if self.p_aug is not None:
            # Apply random-crop image augmentation.
            if np.random.rand() < self.p_aug:
                self.augment(batch, ['observations', 'next_observations'])
        return batch

    def sample_sequence(self, batch_size, sequence_length, discount):
        """
        MODIFIED: This function now correctly handles nested dictionaries for
        observations using tree_map.
        """
        idxs = np.random.randint(self.size - sequence_length + 1, size=batch_size)

        # Helper function to index arrays or map indexing over dictionaries
        def get_batch(data, indices):
            return tree_map(lambda arr: arr[indices], data)

        batch = get_batch(self._dict, idxs)

        # Pre-compute all required indices
        all_idxs = idxs[:, None] + np.arange(sequence_length)[None, :]
        all_idxs = all_idxs.flatten()

        full_sequence_batch = get_batch(self._dict, all_idxs)

        # Reshape each leaf in the observations pytree
        batch_observations = tree_map(
            lambda arr: arr.reshape(batch_size, sequence_length, *arr.shape[1:]),
            full_sequence_batch['observations']
        )
        batch_next_observations = tree_map(
            lambda arr: arr.reshape(batch_size, sequence_length, *arr.shape[1:]),
            full_sequence_batch['next_observations']
        )
        
        # Actions and other arrays are not nested
        batch_actions = full_sequence_batch['actions'].reshape(batch_size, sequence_length, *self['actions'].shape[1:])
        batch_rewards = full_sequence_batch['rewards'].reshape(batch_size, sequence_length, *self['rewards'].shape[1:])
        batch_masks = full_sequence_batch['masks'].reshape(batch_size, sequence_length, *self['masks'].shape[1:])
        batch_terminals = full_sequence_batch['terminals'].reshape(batch_size, sequence_length, *self['terminals'].shape[1:])

        # Calculate next_actions
        next_action_idxs = np.minimum(all_idxs + 1, self.size - 1)
        batch_next_actions = self['actions'][next_action_idxs].reshape(batch_size, sequence_length, *self['actions'].shape[1:])

        # (Reward/mask calculation logic remains the same)
        rewards = np.zeros((batch_size, sequence_length), dtype=float)
        masks = np.ones((batch_size, sequence_length), dtype=float)
        terminals = np.zeros((batch_size, sequence_length), dtype=float)
        valid = np.ones((batch_size, sequence_length), dtype=float)
        
        rewards[:, 0] = batch_rewards[:, 0].squeeze()
        masks[:, 0] = batch_masks[:, 0].squeeze()
        terminals[:, 0] = batch_terminals[:, 0].squeeze()
        
        discount_powers = discount ** np.arange(sequence_length)
        for i in range(1, sequence_length):
            rewards[:, i] = rewards[:, i-1] + batch_rewards[:, i].squeeze() * discount_powers[i]
            masks[:, i] = np.minimum(masks[:, i-1], batch_masks[:, i].squeeze())
            terminals[:, i] = np.maximum(terminals[:, i-1], batch_terminals[:, i].squeeze())
            valid[:, i] = 1.0 - terminals[:, i-1]
        
        return dict(
            observations=batch['observations'],
            full_observations=batch_observations,
            actions=batch_actions,
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=batch_next_observations,
            next_actions=batch_next_actions,
        )
        
    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0

def add_history(dataset, history_length):

    size = dataset.size
    (terminal_locs,) = np.nonzero(dataset['terminals'] > 0)
    initial_locs = np.concatenate([[0], terminal_locs[:-1] + 1])
    assert terminal_locs[-1] == size - 1

    idxs = np.arange(size)
    initial_state_idxs = initial_locs[np.searchsorted(initial_locs, idxs, side='right') - 1]
    obs_rets = []
    acts_rets = []
    for i in reversed(range(1, history_length)):
        cur_idxs = np.maximum(idxs - i, initial_state_idxs)
        outside = (idxs - i < initial_state_idxs)[..., None]
        obs_rets.append(tree_map(lambda arr: arr[cur_idxs] * (~outside) + jnp.zeros_like(arr[cur_idxs]) * outside, 
            dataset['observations']))
        acts_rets.append(tree_map(lambda arr: arr[cur_idxs] * (~outside) + jnp.zeros_like(arr[cur_idxs]) * outside, 
            dataset['actions']))
    observation_history, action_history = tree_map(lambda *args: np.stack(args, axis=-2), *obs_rets),\
        tree_map(lambda *args: np.stack(args, axis=-2), *acts_rets)

    dataset = Dataset(dataset.copy(dict(
        observation_history=observation_history,
        action_history=action_history)))
    
    return dataset


