import torch
import torch.nn as nn
from pyrutils.torch.modules.modules_mlp import build_mlp

class CAD120Baseline(nn.Module):
    def __init__(self, input_size: tuple, num_classes: tuple, hidden_size: int = 128, bidirectional: bool = True,
                 with_message_passing: bool = True, bias: bool = True):
        super(CAD120Baseline, self).__init__()
        human_input_size, object_input_size = input_size
        num_subactivities, num_affordances = num_classes
        self.with_message_passing = with_message_passing

        self.human_embedding_mlp = build_mlp([human_input_size, hidden_size], ['relu'], bias=bias)
        self.object_embedding_mlp = build_mlp([object_input_size, hidden_size], ['relu'], bias=bias)
        self.human_bd_rnn = nn.GRU(hidden_size, hidden_size, num_layers=1, bias=bias, batch_first=True,
                                   bidirectional=bidirectional)
        self.object_bd_rnn = nn.GRU(hidden_size, hidden_size, num_layers=1, bias=bias, batch_first=True,
                                    bidirectional=bidirectional)
        recognition_input_size = hidden_size
        if with_message_passing:
            recognition_input_size *= 2
        if bidirectional:
            recognition_input_size *= 2
        self.human_recognition_mlp = build_mlp([recognition_input_size, num_subactivities],
                                               [{'name': 'logsoftmax', 'dim': -1}], bias=bias)
        self.object_recognition_mlp = build_mlp([recognition_input_size, num_affordances],
                                                [{'name': 'logsoftmax', 'dim': -1}], bias=bias)

    def forward(self, x_human, x_objects, objects_mask):
        """Forward input tensors through CAD120Baseline.

        Arguments:
            x_human - Tensor of shape (batch_size, num_steps, num_humans, human_feature_size) containing frame-level
                features for the human(s) in the video.
            x_objects - Tensor of shape (batch_size, num_steps, num_objects, object_feature_size) containing
                frame-level features for the objects in the video.
            objects_mask - Binary tensor of shape (batch_size, num_objects) indicating whether an object is real or
                virtual. Virtual objects are used to enable batched operations.
        """
        # Initial Embeddings
        x_human, x_objects = self.human_embedding_mlp(x_human), self.object_embedding_mlp(x_objects)
        # Frame-level BiRNNs
        hx_hfr = self._process_frame_level_rnn(x_human, self.human_bd_rnn)
        hx_ofr = self._process_frame_level_rnn(x_objects, self.object_bd_rnn)
        # Objects -> Human message
        if self.with_message_passing:
            # Pool Objects
            objects_mask = torch.unsqueeze(torch.unsqueeze(objects_mask, dim=1), dim=-1)
            hx_ofm = hx_ofr * objects_mask
            hx_ofm = torch.sum(hx_ofm, dim=2, keepdim=True)
            num_real_objects = torch.clamp(torch.sum(objects_mask, dim=2, keepdim=True), min=1.0)
            hx_ofm = hx_ofm / num_real_objects
            # Merge objects with humans
            num_humans = x_human.size(2)
            hx_ofm = torch.repeat_interleave(hx_ofm, repeats=num_humans, dim=2)
            hx_h = torch.cat([hx_hfr, hx_ofm], dim=-1)
        else:
            hx_h = hx_hfr
        y_human_recognition = self.human_recognition_mlp(hx_h).permute(0, 3, 1, 2).contiguous()
        # Human -> Object message
        if self.with_message_passing:
            # Pool Humans
            hx_hfm = torch.sum(hx_hfr, dim=2, keepdim=True)
            # Merge humans with objects
            num_objects = x_objects.size(2)
            hx_hfm = torch.repeat_interleave(hx_hfm, repeats=num_objects, dim=2)
            hx_o = torch.cat([hx_ofr, hx_hfm], dim=-1)
        else:
            hx_o = hx_ofr
        y_object_recognition = self.object_recognition_mlp(hx_o).permute(0, 3, 1, 2).contiguous()
        return [y_human_recognition, y_object_recognition]

    @staticmethod
    def _process_frame_level_rnn(x, rnn):
        """Process frame-level RNNs.

        Arguments:
            x - Tensor of shape (batch_size, num_steps, num_entities, hidden_size) containing the frame-level input
                features of all entities.
            rnn - PyTorch RNN module to process every entity in x.
        Returns:
            A tensor of shape (batch_size, num_steps, num_entities, 2 * hidden_size) containing the frame-level hidden
            states of the input entities.
        """
        h_fr, num_entities = [], x.size(2)
        for e in range(num_entities):
            h_fe, _ = rnn(x[:, :, e])
            h_fr.append(h_fe)
        h_fr = torch.stack(h_fr, dim=2)
        return h_fr
