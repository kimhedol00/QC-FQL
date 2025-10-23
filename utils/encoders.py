import functools
from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from utils.networks import MLP


class ResnetStack(nn.Module):
    """ResNet stack module."""

    num_features: int
    num_blocks: int
    max_pooling: bool = True

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME',
        )(x)

        if self.max_pooling:
            conv_out = nn.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding='SAME',
                strides=(2, 2),
            )

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)
            conv_out += block_input

        return conv_out


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False

    def setup(self):
        stack_sizes = self.stack_sizes
        self.stack_blocks = [
            ResnetStack(
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
            )
            for i in range(len(stack_sizes))
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = x.astype(jnp.float32) / 255.0

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)

        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm()(conv_out)
        out = conv_out.reshape((*x.shape[:-3], -1))

        out = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(out)

        return out

class RoboticsEncoder(nn.Module):
    """
    여러 개의 이미지 스트림과 상태 벡터를 동시에 처리하는 멀티모달 인코더.
    모든 이미지에 대해 CNN 파라미터를 공유하여 효율성을 높입니다.
    """
    image_encoder_cls: nn.Module = ImpalaEncoder
    state_hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, observations: dict, train: bool = True):
        # 파라미터를 공유하는 단일 이미지 인코더를 정의합니다.
        # shared_image_encoder = self.image_encoder_cls(name="shared_image_encoder")
        
        image_features_list = []
        # observations 딕셔너리에서 'image'로 시작하는 모든 키를 찾아 처리합니다.
        image_keys = sorted([k for k in observations.keys() if k.startswith('image')])
        for key in image_keys:
            image_encoder = self.image_encoder_cls(name=f"{key}_encoder")
            image_obs = observations[key]
            # image_features = shared_image_encoder(image_obs, train=train)
            image_features = image_encoder(image_obs, train=train)
            image_features_list.append(image_features)
        # 상태 벡터 처리 브랜치
        state_obs = observations['state']
        state_features = MLP(self.state_hidden_dims, layer_norm=True, activate_final=True)(state_obs)

        # 모든 특징 벡터 (여러 이미지 + 상태)를 하나로 결합합니다.
        all_features = image_features_list + [state_features]
        combined_features = jnp.concatenate(all_features, axis=-1)
        
        return combined_features


class RoboticsStateEncoder(nn.Module):
    """
    여러 개의 이미지 스트림과 상태 벡터를 동시에 처리하는 멀티모달 인코더.
    모든 이미지에 대해 CNN 파라미터를 공유하여 효율성을 높입니다.
    """
    image_encoder_cls: nn.Module = ImpalaEncoder
    state_hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, observations: dict, train: bool = True):
        state_obs = observations['state']
        
        return state_obs







encoder_modules = {
    'impala': ImpalaEncoder,
    'impala_debug': functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    'impala_small': functools.partial(ImpalaEncoder, num_blocks=1,),
    'impala_large': functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
    'robotics_multi_image': RoboticsEncoder,
    'robotics_no_image': RoboticsStateEncoder,
}

encoder_modules['robotics_multi_image_small'] = functools.partial(
    RoboticsEncoder, image_encoder_cls=encoder_modules['impala_small']
)
