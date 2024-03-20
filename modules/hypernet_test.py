import torch
import math
import numpy as np
import timm
import platform
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import resize
from PIL import Image

# 创建一个transform对象，包含了一系列的预处理操作
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像resize到模型需要的大小
    transforms.ToTensor(),  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # 对图像进行归一化，这里的均值和标准差是ImageNet数据集的均值和标准差
])


class PositionalEncoding(nn.Module):
    """
    这个PositionalEncoding类的输入维度该是(time, batch, channel)。
    如果输入是(batch, time, channel)的形式，可以使用torch.transpose或者torch.permute先将其转置为(time, batch, channel)的形式，
    然后再传入PositionalEncoding。
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class VisualImageEncoder(torch.nn.Module):
    def __init__(self):
        global checkpoint_path
        super(VisualImageEncoder, self).__init__()
        # 加载预训练的ViT-H模型
        # self.vit_encoder = timm.create_model('vit_huge_patch16_224', pretrained=True)
        # self.vit_encoder = timm.create_model('vit_base_patch16_224', pretrained=True)
        # TODO: vit encoder
        if platform.system().lower() == "linux":
            checkpoint_path = "projects/AIGC/models/liloras/vit_base_patch16_224/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"
        elif platform.system().lower() == "darwin":
            checkpoint_path = "model_zoo/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"
        self.vit_encoder: nn.Module = timm.create_model(model_name="vit_base_patch16_224",
                                                        checkpoint_path=checkpoint_path)

    def forward(self, x):
        """
        输入图像尺寸为224x224
        """
        features = self.vit_encoder.forward_features(x)  # 提取图片特征，一个(B,T,C)的3D张量
        return features


class WeightTransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(WeightTransformerDecoder, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layer = TransformerDecoderLayer(d_model, nhead)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers)
        self.weight_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model, bias=False)
        )

        # 使用均匀分布初始化全连接层的权重
        for name, param in self.transformer_decoder.named_parameters():
            if 'linear' in name and 'weight' in name:
                nn.init.uniform_(param.data)

        # 将偏置项初始化为0
        for name, param in self.transformer_decoder.named_parameters():
            if 'linear' in name and 'bias' in name:
                nn.init.zeros_(param.data)



    def forward(self, weight_embedding, face_embedding):
        """
        # 创建一个随机的weight_embedding和face_embedding
        weight_embedding = torch.rand(seq_length, batch_size, embedding_dim)
        face_embedding = torch.rand(seq_length, batch_size, embedding_dim)
        """
        # if self.src_mask is None or self.src_mask.size(0) != len(weight_embedding):
        #     device = weight_embedding.device
        #     mask = self._generate_square_subsequent_mask(len(weight_embedding)).to(device)
        #     self.src_mask = mask
        # hidden_embedding = self.transformer_decoder(pos_embedding, face_embedding, self.src_mask)

        pos_embedding = self.pos_encoder(weight_embedding)
        hidden_embedding = self.transformer_decoder(pos_embedding, face_embedding)
        output = self.weight_proj(hidden_embedding)
        return output


    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class HyperNetwork(torch.nn.Module):
    def __init__(
            self,
            ref_img_size: tuple[int] = (224, 224),
            rank: int = 1,
            down_dim: int = 128,
            up_dim: int = 64,
            weight_num: int = 128,
            iters: int = 4,
            train_encoder: bool = False):
        super(HyperNetwork, self).__init__()

        self.weight_dim = (down_dim + up_dim) * rank
        self.weight_num = weight_num
        self.iters = iters
        self.train_encoder = train_encoder
        self.ref_img_size = ref_img_size
        self.visual_image_encoder = VisualImageEncoder()
        self.weight_transformer_decoder = WeightTransformerDecoder(d_model=self.weight_dim, nhead=8, num_layers=4)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        print('Number of hypernetwork parameters: {:.2f}M'.format(total_params))

        # check encoder model shape and format
        test_input = torch.randn(1, 3, *ref_img_size)
        test_output = self.visual_image_encoder(test_input)
        if len(test_output.shape) == 3:
            # default shape in (B,T,C)
            pass
        elif len(test_output.shape) == 4:
            # B, C, H, W -> B, T, C
            test_output = test_output.view(1, test_output.size(1), -1).transpose(1, 2)
        else:
            raise ValueError("Output dimension must be 3 or 4")
        # 根据输出特征维度设置
        feature_dim = test_output.size(-1)
        self.feature_proj = nn.Linear(feature_dim, self.weight_dim, bias=False)

        if not train_encoder:
            # 设置visual_image_encoder为不可训练
            for param in self.visual_image_encoder.parameters():
                param.requires_grad = False
            # 设置visual_image_encoder为评估模式
            self.visual_image_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if not self.train_encoder:
            self.visual_image_encoder.eval()  # 确保visual_image_encoder始终在评估模式

    def train_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x):
        x = resize(x, self.ref_img_size, antialias=True)
        #  batch first
        image_features = self.visual_image_encoder(x)
        # print("image_features:", image_features, image_features.shape)
        face_embedding = self.feature_proj(image_features)
        # weight_embedding zero initialization
        weight_embedding = torch.zeros(face_embedding.size(0), self.weight_num, self.weight_dim,
                                       device=image_features.device)
        # batch first to time first
        face_embedding = face_embedding.permute(1, 0, 2)
        weight_embedding = weight_embedding.permute(1, 0, 2)
        # Iterative Prediction
        for i in range(self.iters):
            weight_embedding += self.weight_transformer_decoder(weight_embedding, face_embedding)
            # print("weight_embedding_%d"%i, weight_embedding)
        # print("weight_embedding Prediction", weight_embedding.shape)
        # time first to batch first
        weight_embedding = weight_embedding.permute(1, 0, 2)
        # weight = self.weight_proj(weight_embedding)
        print("weight:",weight_embedding, weight_embedding.shape)
        return weight_embedding

