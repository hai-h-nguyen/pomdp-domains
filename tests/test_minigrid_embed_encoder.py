import torch
import torch.nn as nn
import gym_minigrid.minigrid
from typing import cast


class MinigridEmbedFeatureExtractor(nn.Module):
    """
    Used for extracting features for Minigrid-like observations, based on
    https://github.com/allenai/allenact/blob/main/allenact_plugins/minigrid_plugin/minigrid_models.py
    """
    def __init__(self, obs_dim, object_embedding_dim=8):
        super(MinigridEmbedFeatureExtractor, self).__init__()

        assert (isinstance(obs_dim, tuple))

        agent_view_x, agent_view_y, view_channels = obs_dim
        assert agent_view_x == agent_view_y
        self.agent_view = agent_view_x
        self.view_channels = view_channels

        self.num_objects = (
            cast(
                int, max(map(abs, gym_minigrid.minigrid.OBJECT_TO_IDX.values()))  # type: ignore
            )
            + 1
        )
        self.num_colors = (
            cast(int, max(map(abs, gym_minigrid.minigrid.COLOR_TO_IDX.values())))  # type: ignore
            + 1
        )
        self.num_states = (
            cast(int, max(map(abs, gym_minigrid.minigrid.STATE_TO_IDX.values())))  # type: ignore
            + 1
        )

        self.num_channels = 0

        self.object_embedding_dim = object_embedding_dim

        if self.num_objects > 0:
            # Object embedding
            self.object_embedding = nn.Embedding(
                num_embeddings=self.num_objects, embedding_dim=self.object_embedding_dim
            )
            self.object_channel = self.num_channels
            self.num_channels += 1

        if self.num_colors > 0:
            # Same dimensionality used for colors and states
            self.color_embedding = nn.Embedding(
                num_embeddings=self.num_colors, embedding_dim=self.object_embedding_dim
            )
            self.color_channel = self.num_channels
            self.num_channels += 1

        if self.num_states > 0:
            self.state_embedding = nn.Embedding(
                num_embeddings=self.num_states, embedding_dim=self.object_embedding_dim
            )
            self.state_channel = self.num_channels
            self.num_channels += 1

    def forward(self, inputs):
        embed_list = []

        if self.num_objects > 0:
            ego_object_embeds = self.object_embedding(
                inputs[..., self.object_channel].long()
            )
            embed_list.append(ego_object_embeds)

        if self.num_colors > 0:
            ego_color_embeds = self.color_embedding(
                inputs[..., self.color_channel].long()
            )
            embed_list.append(ego_color_embeds)

        if self.num_states > 0:
            ego_state_embeds = self.state_embedding(
                inputs[..., self.state_channel].long()
            )
            embed_list.append(ego_state_embeds)

        ego_embeds = torch.cat(embed_list, dim=-1)

        print(ego_embeds.shape)

        return ego_embeds

    def get_output_size(self):
        return self.num_channels*self.object_embedding_dim

input = torch.rand((10, 7, 7, 3))
encoder = MinigridEmbedFeatureExtractor((7, 7, 7))

output = encoder(input)

print(output.shape, encoder.get_output_size())