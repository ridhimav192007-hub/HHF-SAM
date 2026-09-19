import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from models.mhpm import MHPM
from models.phfd import PHFD
from models.hfm import HFM


# Adapter
class Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.down = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.up = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.down(x)
        x = self.relu(x)
        x = self.up(x)
        return x
# Adapter wrapper
class AdapterWrapper(nn.Module):
    def __init__(self, original_block, dim=768, hidden_dim=64):
        super().__init__()

        self.original_block = original_block
        self.adapter = Adapter(dim, hidden_dim)

    def forward(self, x):
        x = self.original_block(x)
        x = x + self.adapter(x)
        return x



# LoRA
class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank=4):
        super().__init__()

        self.lora_A = nn.Linear(input_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, output_dim, bias=False)

    def forward(self, x):
        return self.lora_B(self.lora_A(x))

    # LoRA wrapper for SAM Query and Value
class LoRAQKV(nn.Module):
    def __init__(self, original_qkv, dim=768, rank=4):
        super().__init__()

        self.original_qkv = original_qkv
        self.lora_q = LoRALayer(dim, dim, rank)
        self.lora_v = LoRALayer(dim, dim, rank)

    def forward(self, x):
        qkv = self.original_qkv(x)

        q_update = self.lora_q(x)
        v_update = self.lora_v(x)

        qkv[..., :768] += q_update
        qkv[..., 1536:2304] += v_update

        return qkv




# Load pretrained SAM-B
sam = sam_model_registry["vit_b"](
    checkpoint="checkpoints/sam_vit_b_01ec64.pth"
)

print("SAM-B loaded successfully!")
# Freeze original SAM parameters
for param in sam.parameters():
    param.requires_grad = False

print("Original SAM parameters frozen!")
# Check SAM-B Transformer blocks
print("Number of SAM encoder blocks:", len(sam.image_encoder.blocks))
# Inspect attention layer of first Transformer block
print(sam.image_encoder.blocks[0].attn)

# Attach LoRA to Q and V in all SAM encoder blocks
for block in sam.image_encoder.blocks:
    block.attn.qkv = LoRAQKV(block.attn.qkv)

print("LoRA attached to all SAM encoder blocks!")
# Check first block after attaching LoRA
print("New QKV layer:")
print(sam.image_encoder.blocks[0].attn.qkv)
print("First SAM Transformer block:")
print(sam.image_encoder.blocks[0])
# Attach Adapter to all SAM Transformer blocks
for i in range(len(sam.image_encoder.blocks)):
    sam.image_encoder.blocks[i] = AdapterWrapper(
        sam.image_encoder.blocks[i]
    )

print("Adapter attached to all SAM encoder blocks!")
# Capture features from SAM layers 3, 6, 9, and 12
features = {}

def save_feature(layer_number):
    def hook(module, input, output):
        features[layer_number] = output
    return hook

sam.image_encoder.blocks[2].register_forward_hook(save_feature(3))
sam.image_encoder.blocks[5].register_forward_hook(save_feature(6))
sam.image_encoder.blocks[8].register_forward_hook(save_feature(9))
sam.image_encoder.blocks[11].register_forward_hook(save_feature(12))

print("Hooks attached to layers 3, 6, 9, and 12!")
# HHF-SAM modules
mhpm = MHPM(channels=768)
hfm = HFM(channels=768)
phfd = PHFD(channels=768)