import torch
from torch import nn

from torch.nn.init import trunc_normal_

class ActionMatchLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.,
        bias = True,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, visual):
        q = k = v = self.norm1(x)
        att = self.cross_attn(q, visual, visual)[0]
        x = x + att
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class ActionMatch(nn.Module):
    def __init__(self, layers=2, embed_dim=64, alpha=0.1,):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.ModuleList([ActionMatchLayer(embed_dim, embed_dim//64) for _ in range(layers)])
        self.alpha = nn.Parameter(torch.ones(embed_dim) * alpha)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    
    def forward(self, text, visual):
        # B, N, C = visual.shape
        visual = self.norm(visual)
        for layer in self.decoder:
            text = layer(text, visual)
        # print("alpha: ", self.alpha)
        
        return self.alpha * text 
        # return self.alpha * text + text


if __name__ == '__main__':
    input_caption = torch.randn([8,172,512]).transpose(0,1)
    input_action = torch.randn([8,13,512]).transpose(0,1)
    CaptionPrompt = ActionMatch(embed_dim=512)
    time_step = input_caption.shape[0]
    output = []
    for t in range(time_step):
        output.append(CaptionPrompt(input_action, input_caption[t].unsqueeze(0)))
    output = torch.stack(output, dim=0)
    print(output.shape)
