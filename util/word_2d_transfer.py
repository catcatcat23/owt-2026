"""Explicit supervised WORD 2D -> BTCV 3D initialization (not resume)."""
import math
import torch
import torch.nn.functional as F
from util.mae_transfer import sha256_file


def inflate_conv(weight, target_shape):
    if weight.ndim != 4 or len(target_shape) != 5:
        raise ValueError('Expected Conv2d -> Conv3d')
    depth = target_shape[2]
    result = weight.unsqueeze(2).repeat(1, 1, depth, 1, 1) / depth
    if tuple(result.shape) != tuple(target_shape):
        raise ValueError('Spatial/channel convolution mismatch')
    return result


def resize_canvas(weight, grid):
    side = math.isqrt(weight.shape[0])
    if side * side != weight.shape[0] or len(grid) != 3:
        raise ValueError('Expected square 2D AHER grid and 3D target grid')
    image = weight.T.reshape(1, weight.shape[1], side, side).float()
    image = F.interpolate(image, size=grid[-2:], mode='bilinear', align_corners=False)
    spatial = image[0].permute(1, 2, 0).reshape(-1, weight.shape[1])
    return spatial.repeat(grid[0], 1).to(weight.dtype)


def load_word_2d_transfer(model, path, args):
    checkpoint = torch.load(path, map_location='cpu')
    saved = checkpoint['args']
    saved = saved if isinstance(saved, dict) else vars(saved)
    if saved.get('dimension') != '2D' or args.dimension != '3D':
        raise ValueError('This transfer supports only 2D -> 3D')
    for key in ('backbone', 'token_factor', 'slot_tg_depth', 'slot_head_type',
                'slot_head_channels', 'fusion_mode', 'fusion_reference_count',
                'hsam_supervision', 'pixel_pe', 'query_refinement'):
        if saved.get(key) != getattr(args, key):
            raise ValueError('WORD transfer configuration mismatch: ' + key)
    source = {k: v for k, v in checkpoint['model'].items()
              if not k.startswith('perceptual_loss.')}
    # Slot names are encoded in the state keys; exact loading below also checks them.
    target = model.state_dict()
    grid = tuple(model.patch_embed.grid_size)
    copied, converted, initialized, used = [], [], [], set()
    positional = {'pos_embed_spatial', 'decoder_pos_embed_spatial'}
    zero_positions = {'pos_embed_temporal', 'decoder_pos_embed_temporal', 'decoder_pos_embed_class'}
    for key, value in target.items():
        if key in positional:
            # Native 512-grid sin/cos, not a stretched learned spatial table.
            initialized.append(key + ': native spatial sin/cos')
            continue
        if key in zero_positions:
            target[key] = torch.zeros_like(value)
            initialized.append(key + ': zero')
            continue
        if key in source and source[key].shape == value.shape:
            target[key] = source[key]; used.add(key); copied.append(key)
        elif key in source and value.ndim == 5 and source[key].ndim == 4:
            target[key] = inflate_conv(source[key], value.shape)
            used.add(key); converted.append(key + ': temporal average inflation')
        elif key.endswith('.aher.sp_linear1.weight') and key in source:
            target[key] = resize_canvas(source[key], grid)
            used.add(key); converted.append(key + ': spatial interpolation + time repetition')
        elif key in ('decoder_pred.0.weight', 'decoder_pred.0.bias'):
            origin = key.replace('decoder_pred.0.', 'decoder_pred.')
            target[key] = source[origin]; used.add(origin)
            converted.append(key + ': copy 2D reconstruction projection')
        elif key == 'decoder_pred.2.weight' and value.shape[0] == value.shape[1]:
            target[key] = torch.eye(value.shape[0], dtype=value.dtype)
            initialized.append(key + ': identity after existing 3D tanh')
        elif key == 'decoder_pred.2.bias':
            target[key] = torch.zeros_like(value)
            initialized.append(key + ': zero')
        else:
            raise ValueError('Unspecified WORD transfer tensor: ' + key)
        if target[key].shape != value.shape or not torch.isfinite(target[key]).all():
            raise ValueError('Invalid transferred tensor: ' + key)
    unused = set(source) - used - {'pos_embed'}
    if unused:
        raise ValueError('Unexpected unused WORD parameters: ' + str(sorted(unused)))
    model.load_state_dict(target, strict=True)
    return dict(path=str(path), sha256=sha256_file(path), source_epoch=checkpoint['epoch'],
                source_dataset=saved.get('dataset_protocol'), source_input_size=saved['input_size'],
                target_input_size=args.input_size, copied=copied, converted=converted,
                initialized=initialized, source_pos_embed='replaced by native 3D spatial sin/cos',
                exact=False, optimizer_restored=False, transfer='WORD supervised 2D -> BTCV 3D')
