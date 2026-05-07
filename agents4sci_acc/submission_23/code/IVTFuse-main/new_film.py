"""
Improved image‑fusion network for IR/VIS image fusion with text guidance.

This module implements a new network that follows the general structure of
the original IF‑FILM ``Net`` class but incorporates lightweight frequency‑aware
and hybrid pooling attention modules.  The network accepts an infrared
image, a visible image and an encoded text description and returns a
fused grayscale image.  The text features are assumed to be pre‑encoded
by an external encoder (e.g. BLIP); a simple 1‑D convolution projects
them to the hidden dimension used in the model.

The core idea is to enhance local feature extraction in each modality
encoder by inserting two small attention modules:

  * **FrequencyStripAttention** (FSA) splits the input feature into
    horizontal and vertical low/high‑frequency components and applies
    learnable weights to recombine them【393668295883076†L25-L46】.
  * **HybridPoolingAttention** (HPA) aggregates context along the
    horizontal and vertical axes using both average and max pooling to
    produce an attention mask【865671043685464†L15-L37】.  A group
    normalization and small convolutions mix the pooled features.

These modules are lightweight and add only a small number of parameters
relative to the baseline Restormer blocks.  The rest of the block
closely follows the structure of ``restormer_cablock`` in the original
repository: features from each modality are processed by a Restormer
block, converted to a sequence of tokens, refined via cross‑attention
with the text tokens, and then mapped back to the spatial domain.
The final fusion stage concatenates the outputs from the IR and VIS
branches and passes them through three Restormer layers and two
pointwise convolutions to produce a fused output.  A Sigmoid activation
ensures the fused image is in the [0, 1] range.

This network is designed to keep a similar parameter count to the
original IF‑FILM model and avoid heavy modules such as Mamba.  It
should train comfortably on a single RTX 4090 (≈24 GB VRAM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Import the TransformerBlock implementation used in the original FILM.
    # If the module is not found, the user should ensure that
    # ``net/restormer.py`` is available in the Python path.
    from net.restormer import TransformerBlock as Restormer
except Exception as e:
    raise ImportError(
        "Could not import Restormer. Ensure that net/restormer.py is available."  # noqa: E501
    ) from e


class FrequencyStripAttention(nn.Module):
    """Frequency Strip Attention (FSA).

    This module separates horizontal and vertical low‑frequency and
    high‑frequency components using average pooling.  Learnable
    parameters weight the contributions of low/high components before
    recombining the result.  See the original code for details【393668295883076†L25-L46】.
    """

    def __init__(self, k: int, kernel: int = 7) -> None:
        super().__init__()
        self.channel = k
        # Per‑channel learnable weights for vertical and horizontal
        # low/high‑frequency components.
        self.vert_low = nn.Parameter(torch.zeros(k, 1, 1))
        self.vert_high = nn.Parameter(torch.zeros(k, 1, 1))
        self.hori_low = nn.Parameter(torch.zeros(k, 1, 1))
        self.hori_high = nn.Parameter(torch.zeros(k, 1, 1))
        # Pooling operators to extract low‑frequency components along each
        # spatial dimension.
        self.vert_pool = nn.AvgPool2d(kernel_size=(kernel, 1), stride=1)
        self.hori_pool = nn.AvgPool2d(kernel_size=(1, kernel), stride=1)
        # Reflection padding to maintain spatial dimensions.
        pad_size = kernel // 2
        self.pad_vert = nn.ReflectionPad2d((0, 0, pad_size, pad_size))
        self.pad_hori = nn.ReflectionPad2d((pad_size, pad_size, 0, 0))
        # Blending coefficients.
        self.gamma = nn.Parameter(torch.zeros(k, 1, 1))
        self.beta = nn.Parameter(torch.ones(k, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Horizontal low‑frequency component via average pooling.
        hori_l = self.hori_pool(self.pad_hori(x))
        # Horizontal high‑frequency component is the residual.
        hori_h = x - hori_l
        # Combine horizontal low/high with learnable weights.
        hori_out = self.hori_low * hori_l + (self.hori_high + 1.0) * hori_h
        # Vertical low‑frequency component on the horizontally processed output.
        vert_l = self.vert_pool(self.pad_vert(hori_out))
        vert_h = hori_out - vert_l
        vert_out = self.vert_low * vert_l + (self.vert_high + 1.0) * vert_h
        # Blend the refined output with the original input.
        return x * self.beta + vert_out * self.gamma


class HybridPoolingAttention(nn.Module):
    """Hybrid Pooling Attention (HPA).

    This attention module aggregates horizontal and vertical context using
    both average and max pooling.  It splits the input channels into
    groups and computes an attention mask for each group, which is then
    applied to the corresponding features.  The implementation
    here follows the original code but uses a configurable number of
    groups (``factor``) to keep the model lightweight【865671043685464†L15-L37】.
    """

    def __init__(self, channels: int, factor: int = 4) -> None:
        super().__init__()
        self.groups = factor
        # Ensure that channels can be divided into the requested number of groups.
        assert channels % self.groups == 0, "channels must be divisible by factor"
        self.softmax = nn.Softmax(dim=-1)
        # Pooling operators.
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.map = nn.AdaptiveMaxPool2d((1, 1))
        # Average and max pooling along height (Y) and width (X).
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # average over width
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # average over height
        self.max_h = nn.AdaptiveMaxPool2d((None, 1))
        self.max_w = nn.AdaptiveMaxPool2d((1, None))
        # Normalisation and convolution layers per group.
        self.gn = nn.GroupNorm(num_groups=channels // self.groups,
                              num_channels=channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups,
                                 channels // self.groups,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups,
                                 channels // self.groups,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape to (batch * groups, channels_per_group, h, w).
        b, c, h, w = x.size()
        g = self.groups
        group_x = x.view(b * g, c // g, h, w)
        # Average pooling along height and width.
        x_h = self.pool_h(group_x)  # shape: (b*g, c//g, h, 1)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # shape: (b*g, c//g, w, 1)
        # Concatenate and learn a mapping.
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        # Split into height and width components.
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        # Compute average‑pooled attention mask.
        x1 = self.gn(group_x * torch.sigmoid(x_h) * torch.sigmoid(x_w.permute(0, 1, 3, 2)))
        # Additional convolution branch.
        x2 = self.conv3x3(group_x)
        # Max pooling branch for complementary features.
        y_h = self.max_h(group_x)
        y_w = self.max_w(group_x).permute(0, 1, 3, 2)
        yhw = self.conv1x1(torch.cat([y_h, y_w], dim=2))
        y_h, y_w = torch.split(yhw, [h, w], dim=2)
        y1 = self.gn(group_x * torch.sigmoid(y_h) * torch.sigmoid(y_w.permute(0, 1, 3, 2)))
        # Flatten spatial dimensions for attention weight computation.
        y11 = y1.view(b * g, c // g, -1)
        y12 = self.softmax(self.map(y1).view(b * g, 1, -1))
        x11 = x1.view(b * g, c // g, -1)
        x12 = self.softmax(self.agp(x1).view(b * g, 1, -1))
        x21 = x2.view(b * g, c // g, -1)
        x22 = self.softmax(self.agp(x2).view(b * g, 1, -1))
        # Combine weights from the average and max branches.
        weights = (torch.matmul(x12, y11) + torch.matmul(y12, x11)).view(b * g, 1, h, w)
        # Apply attention to the grouped features and reshape back.
        out = group_x * torch.sigmoid(weights)
        return out.view(b, c, h, w)


class CrossAttention(nn.Module):
    """A wrapper around PyTorch's ``nn.MultiheadAttention`` for cross‑attention.

    The inputs ``query``, ``key`` and ``value`` are expected to be of shape
    ``(batch, seq_len, embed_dim)``.  Internally the module transposes
    dimensions to match the expected shape of ``MultiheadAttention`` and
    returns the output in the same layout as the input.
    """

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          batch_first=True)

    def forward(self, query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        # PyTorch's MultiheadAttention with batch_first expects inputs of
        # shape (batch, seq_len, embed_dim).
        attn_output, _ = self.attn(query, key, value)
        return attn_output


class imagefeature2textfeature(nn.Module):
    """Convert a 2‑D image feature map to a sequence suitable for text interaction.

    A 1×1 convolution reduces (or expands) the input channels to
    ``mid_channel``.  The resulting feature map is then resampled to a
    fixed spatial size (288×384 in the original IF‑FILM) and reshaped
    into a sequence of length ``num_tokens = (288*384)/hidden_dim`` with
    token dimension equal to ``hidden_dim``【849992398940071†L117-L124】.
    """

    def __init__(self, in_channel: int, mid_channel: int, hidden_dim: int,
                 target_size=(288, 384)) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=mid_channel,
                              kernel_size=1)
        self.hidden_dim = hidden_dim
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, C, H, W)
        x = self.conv(x)
        # Resample to the canonical spatial size and reshape into (batch, seq_len, hidden_dim).
        x = F.interpolate(x, self.target_size, mode='nearest')
        batch_size = x.size(0)
        total_elems = x.numel() // batch_size
        seq_len = total_elems // self.hidden_dim
        return x.view(batch_size, seq_len, self.hidden_dim)


class text_preprocess(nn.Module):
    """Project pre‑encoded text features to the model's hidden dimension.

    The input text is assumed to be a tensor of shape (batch, seq_len, in_dim).
    A 1‑D convolution (implemented as ``nn.Conv1d``) maps the last
    dimension from ``in_dim`` to ``out_dim``.  This mirrors the
    behaviour of the original IF‑FILM where text features of size 768
    are projected to ``hidden_dim``【849992398940071†L160-L176】.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_dim,
                              out_channels=out_dim,
                              kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, in_dim)
        # Permute to (batch, in_dim, seq_len) for Conv1d and then back.
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class ImprovedCABlock(nn.Module):
    """A fusion block inspired by ``restormer_cablock`` with added FSA/HPA.

    This block processes infrared and visible features separately using
    convolution, non‑linearity, Frequency Strip Attention and Hybrid
    Pooling Attention.  Each branch then passes through a Restormer
    block to model local–global interactions.  The resulting feature
    maps are converted to token sequences for cross‑attention with the
    text.  Cross‑attention produces a weight vector which modulates the
    image tokens; the tokens are reshaped back to images and combined
    with the original features via a residual connection and a small
    convolution.  The output consists of updated IR and VIS feature maps.
    """

    def __init__(self,
                 input_channel: int = 1,
                 restormer_dim: int = 32,
                 num_heads: int = 8,
                 hidden_dim: int = 256,
                 image2text_dim: int = 32,
                 ffn_expansion_factor: int = 4,
                 bias: bool = False,
                 LayerNorm_type: str = 'WithBias',
                 fsa_kernel: int = 7,
                 hpa_factor: int = 4,
                 target_size: tuple = (288, 384),
                 use_fsa: bool = True,
                 use_hpa: bool = True) -> None:
        super().__init__()
        self.image2text_dim = image2text_dim
        # Convolution and non‑linearity for each modality.
        self.convA1 = nn.Conv2d(input_channel, restormer_dim, kernel_size=3,
                                stride=1, padding=1, bias=bias)
        self.preluA1 = nn.PReLU()
        self.convB1 = nn.Conv2d(input_channel, restormer_dim, kernel_size=3,
                                stride=1, padding=1, bias=bias)
        self.preluB1 = nn.PReLU()
        # FrequencyStripAttention and HybridPoolingAttention per modality.
        self.fsaA = (FrequencyStripAttention(restormer_dim, kernel=fsa_kernel)
                     if use_fsa else nn.Identity())
        self.fsaB = (FrequencyStripAttention(restormer_dim, kernel=fsa_kernel)
                     if use_fsa else nn.Identity())
        self.hpaA = (HybridPoolingAttention(restormer_dim, factor=hpa_factor)
                     if use_hpa else nn.Identity())
        self.hpaB = (HybridPoolingAttention(restormer_dim, factor=hpa_factor)
                     if use_hpa else nn.Identity())
        # Restormer blocks for each modality.
        self.restormerA = Restormer(restormer_dim,
                                    num_heads,
                                    ffn_expansion_factor,
                                    bias,
                                    LayerNorm_type)
        self.restormerB = Restormer(restormer_dim,
                                    num_heads,
                                    ffn_expansion_factor,
                                    bias,
                                    LayerNorm_type)
        # Convert image features to sequences of text tokens.
        self.imagef2textfA = imagefeature2textfeature(restormer_dim,
                                                      image2text_dim,
                                                      hidden_dim,
                                                      target_size=target_size)
        self.imagef2textfB = imagefeature2textfeature(restormer_dim,
                                                      image2text_dim,
                                                      hidden_dim,
                                                      target_size=target_size)
        # Cross‑attention modules for each modality.
        self.cross_attentionA = CrossAttention(embed_dim=hidden_dim,
                                               num_heads=num_heads)
        self.cross_attentionB = CrossAttention(embed_dim=hidden_dim,
                                               num_heads=num_heads)
        # 1×1 convolution to combine image and text features.
        self.convA2 = nn.Conv2d(image2text_dim, restormer_dim, kernel_size=1)
        self.preluA2 = nn.PReLU()
        self.convB2 = nn.Conv2d(image2text_dim, restormer_dim, kernel_size=1)
        self.preluB2 = nn.PReLU()
        # Final projection after concatenating original and text features.
        self.convA3 = nn.Conv2d(2 * restormer_dim, restormer_dim, kernel_size=1)
        self.preluA3 = nn.PReLU()
        self.convB3 = nn.Conv2d(2 * restormer_dim, restormer_dim, kernel_size=1)
        self.preluB3 = nn.PReLU()

    def forward(self,
                imageA: torch.Tensor,
                imageB: torch.Tensor,
                text: torch.Tensor) -> tuple:
        """Process IR and VIS features and text tokens.

        Args:
            imageA: infrared image tensor of shape (B, C, H, W).
            imageB: visible image tensor of shape (B, C, H, W).
            text: encoded text tensor of shape (B, seq_len, hidden_dim_in).

        Returns:
            A tuple ``(featA, featB)`` where each is a tensor of shape
            (B, restormer_dim, H, W).
        """
        # Apply convolution, non‑linearity, FSA and HPA to each modality.
        featA = self.preluA1(self.convA1(imageA))
        featA = self.fsaA(featA)
        featA = self.hpaA(featA)
        featA = self.restormerA(featA)

        featB = self.preluB1(self.convB1(imageB))
        featB = self.fsaB(featB)
        featB = self.hpaB(featB)
        featB = self.restormerB(featB)

        # Convert image features to token sequences.
        featA_tokens = self.imagef2textfA(featA)  # (B, seq_len, hidden_dim)
        featB_tokens = self.imagef2textfB(featB)  # (B, seq_len, hidden_dim)
        # Perform cross‑attention between text and image tokens.
        # Query: text (B, T_text, H), Key/Value: image tokens (B, T_img, H).
        caA = self.cross_attentionA(text, featA_tokens, featA_tokens)
        caB = self.cross_attentionB(text, featB_tokens, featB_tokens)
        # Aggregate text‑attention weights by averaging over the sequence.
        # Use adaptive average pooling to collapse the token dimension.
        caA_weights = F.adaptive_avg_pool1d(caA.permute(0, 2, 1), 1).permute(0, 2, 1)
        caA_weights = F.normalize(caA_weights, p=1, dim=2)
        caB_weights = F.adaptive_avg_pool1d(caB.permute(0, 2, 1), 1).permute(0, 2, 1)
        caB_weights = F.normalize(caB_weights, p=1, dim=2)
        # Reweight the image tokens using the attention weights.
        caA_tokens = (featA_tokens * caA_weights).view(featA.shape[0],
                                                      self.image2text_dim,
                                                      *self.imagef2textfA.target_size)
        caB_tokens = (featB_tokens * caB_weights).view(featB.shape[0],
                                                      self.image2text_dim,
                                                      *self.imagef2textfB.target_size)
        # Resize to match the spatial size of the current feature maps.
        # Note: at this stage featA and featB may have spatial dimensions
        # different from target_size if the input images are resized.
        H, W = featA.shape[2], featA.shape[3]
        caA_tokens = F.interpolate(caA_tokens, (H, W), mode='nearest')
        caB_tokens = F.interpolate(caB_tokens, (H, W), mode='nearest')
        # Project the reweighted text features back to restormer_dim channels.
        caA_proj = self.preluA2(self.convA2(caA_tokens))
        caB_proj = self.preluB2(self.convB2(caB_tokens))
        # Combine the original features with the text‑guided features.
        # The concatenation of the Restormer output and the text projection
        # resembles the structure of the original cablock【849992398940071†L117-L131】.
        outA = self.preluA3(self.convA3(torch.cat((featA, caA_proj + featA), dim=1)))
        outB = self.preluB3(self.convB3(torch.cat((featB, caB_proj + featB), dim=1)))
        return outA, outB


class Net(nn.Module):
    """Improved FILM network with FSA/HPA and cross‑attention.

    The network accepts two modality images (infrared and visible) and
    encoded text, processes them through three instances of
    ``ImprovedCABlock``, concatenates the resulting features and passes
    them through a sequence of Restormer and pointwise convolution
    layers to produce a fused output.  A Sigmoid activation ensures
    the output lies in [0, 1].
    """

    def __init__(self,
                 mid_channel: int = 32,
                 decoder_num_heads: int = 8,
                 ffn_factor: int = 4,
                 bias: bool = False,
                 LayerNorm_type: str = 'WithBias',
                 out_channel: int = 1,
                 hidden_dim: int = 256,
                 image2text_dim: int = 32,
                 fsa_kernel: int = 7,
                 hpa_factor: int = 4,
                 text_in_dim: int = 768,
                 target_size: tuple = (288, 384),
                 use_fsa: bool = True,
                 use_hpa: bool = True) -> None:
        super().__init__()
        # Project text features to the hidden dimension.
        self.text_process = text_preprocess(text_in_dim, hidden_dim)
        # Three fusion blocks for hierarchical processing.
        self.block1 = ImprovedCABlock(input_channel=1,
                                      restormer_dim=mid_channel,
                                      num_heads=decoder_num_heads,
                                      hidden_dim=hidden_dim,
                                      image2text_dim=image2text_dim,
                                      ffn_expansion_factor=ffn_factor,
                                      bias=bias,
                                      LayerNorm_type=LayerNorm_type,
                                      fsa_kernel=fsa_kernel,
                                      hpa_factor=hpa_factor,
                                      target_size=target_size,
                                      use_fsa=use_fsa,
                                      use_hpa=use_hpa)
        self.block2 = ImprovedCABlock(input_channel=mid_channel,
                                      restormer_dim=mid_channel,
                                      num_heads=decoder_num_heads,
                                      hidden_dim=hidden_dim,
                                      image2text_dim=image2text_dim,
                                      ffn_expansion_factor=ffn_factor,
                                      bias=bias,
                                      LayerNorm_type=LayerNorm_type,
                                      fsa_kernel=fsa_kernel,
                                      hpa_factor=hpa_factor,
                                      target_size=target_size,
                                      use_fsa=use_fsa,
                                      use_hpa=use_hpa)
        self.block3 = ImprovedCABlock(input_channel=mid_channel,
                                      restormer_dim=mid_channel,
                                      num_heads=decoder_num_heads,
                                      hidden_dim=hidden_dim,
                                      image2text_dim=image2text_dim,
                                      ffn_expansion_factor=ffn_factor,
                                      bias=bias,
                                      LayerNorm_type=LayerNorm_type,
                                      fsa_kernel=fsa_kernel,
                                      hpa_factor=hpa_factor,
                                      target_size=target_size,
                                      use_fsa=use_fsa,
                                      use_hpa=use_hpa)
        # Post‑fusion Restormer layers and convolutions.
        self.restormer1 = Restormer(2 * mid_channel,
                                    decoder_num_heads,
                                    ffn_factor,
                                    bias,
                                    LayerNorm_type)
        self.conv1 = nn.Conv2d(2 * mid_channel, mid_channel, kernel_size=1)
        self.restormer2 = Restormer(mid_channel,
                                    decoder_num_heads,
                                    ffn_factor,
                                    bias,
                                    LayerNorm_type)
        self.restormer3 = Restormer(mid_channel,
                                    decoder_num_heads,
                                    ffn_factor,
                                    bias,
                                    LayerNorm_type)
        self.conv2 = nn.Conv2d(mid_channel, out_channel, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self,
                imageA: torch.Tensor,
                imageB: torch.Tensor,
                text: torch.Tensor) -> torch.Tensor:
        """Forward pass for the improved fusion network.

        Args:
            imageA: infrared image tensor (B, 1, H, W).
            imageB: visible image tensor (B, 1, H, W).
            text: encoded text tensor (B, seq_len, text_in_dim).

        Returns:
            Fused image tensor of shape (B, 1, H, W).
        """
        # Project the pre‑encoded text to the model's hidden dimension.
        text_features = self.text_process(text)
        # Process through three improved fusion blocks.
        featA1, featB1 = self.block1(imageA, imageB, text_features)
        featA2, featB2 = self.block2(featA1, featB1, text_features)
        featA3, featB3 = self.block3(featA2, featB2, text_features)
        # Concatenate the final features from IR and VIS streams.
        fusion_feat = torch.cat((featA3, featB3), dim=1)
        # Post‑fusion refinement.
        fusion_feat = self.restormer1(fusion_feat)
        fusion_feat = self.conv1(fusion_feat)
        fusion_feat = self.restormer2(fusion_feat)
        fusion_feat = self.restormer3(fusion_feat)
        fusion_feat = self.conv2(fusion_feat)
        # Output fused image in [0, 1].
        return self.act(fusion_feat)