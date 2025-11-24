import torch

""" 
    Custom collate function.
    We need to reorder the batch based on lengths of the sentences. 
    Since pytorch requires an ordered tensor for the lstm.
    lengths, sort_id = torch.Tensor([x for x in lengths]).sort(descending=True)
"""
# def custom_collate(batch):
#     output = {k: [] for k in batch[0].keys()}
    
#     # Group all values together as a list with the corresponding key 
#     for sample in batch:
#         for k in output.keys():
#             output[k].append(sample[k])
    
#     output['command_length'] = torch.LongTensor([c for c in output['command_length']])

#     # Sort the samples bases on command length
#     lengths, sort_id = output['command_length'].sort(descending=True)
#     sort_id = sort_id.tolist()
    
#     # Order all elements accordingly
#     output = {k: [v[i] for i in sort_id] for k, v in output.items()} 

#     # Stack as tensors
#     output = {k: torch.stack(v, 0).squeeze() for k, v in output.items()}    

#     return output 


# talk2car/baseline/utils/collate.py
# ...existing code...


def custom_collate(batch):
    """
    Collate for SBERT-style encoder:
      - keep 'command_text' as list[str] (SBERT expects raw strings)
      - stack tensor fields into batched tensors
      - ensure 'index' and 'rpn_gt' become 1D tensors of shape (B,)
    """
    out = {}

    # Keep command_text as list of strings for SBERT
    out['command_text'] = [sample['command_text'] for sample in batch]

    # Tensor fields to stack
    tensor_keys = ['image', 'rpn_image', 'rpn_bbox_lbrt', 'rpn_iou']

    for key in tensor_keys:
        out[key] = torch.stack([sample[key] for sample in batch], dim=0)

    # rpn_gt is stored as LongTensor([val]) per sample -> stack and squeeze to (B,)
    out['rpn_gt'] = torch.stack([sample['rpn_gt'] for sample in batch], dim=0).squeeze()
    # index: convert list of LongTensor([idx]) -> LongTensor(B,)
    out['index'] = torch.stack([sample['index'] for sample in batch], dim=0).squeeze()

    return out







