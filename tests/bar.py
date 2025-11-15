import torch
import torch.nn.functional as F

compression_attention_mask = torch.tensor([1], dtype=torch.int).repeat(2, 5)
print(compression_attention_mask)

a, b = torch.randint(1, 10, (1, 5, 32), dtype=float), torch.randint(1, 10, (1, 5, 32), dtype=float)
print(a, b)
print(F.cosine_similarity(a, b, dim=-1).mean())
print(torch.tensor([[1]]).repeat(2, 4))
print(torch.nn.functional.l1_loss(a, b, reduction="none"))
print(1e-2)

# if hybrid_alpha is not None:
#     labels = input_ids.clone()
#     labels[attention_mask == 0] = -100
#     # TODO: How to correctly slice compression_outputs?
#     ce_loss = F.cross_entropy(
#         compression_outputs.logits[:, num_compression_tokens - 1: -1].flatten(0, 1),
#         labels.flatten(),
#         reduction="mean",
#     )
#     for i in alignment_layer_indices:
#         target_hidden_states = outputs.hidden_states[i]  # [batch, sequence, hidden]
#         compression_hidden_states = compression_outputs.hidden_states[i][
#                                     :, :num_compression_tokens
#                                     ]  # [batch, mem, hidden]
#         if loss_type == "l2":
#             l2_loss = torch.mean(
#                 torch.mean(
#                     torch.sqrt(
#                         torch.sum(
#                             (target_hidden_states.unsqueeze(dim=0) - compression_hidden_states.unsqueeze(dim=2))
#                             ** 2,
#                             dim=-1,
#                         )
#                     ),
#                     dim=-1,
#                 )
#             )
#             loss = (1 - hybrid_alpha) * ce_loss + hybrid_alpha * l2_loss
#         elif loss_type == "l1":
#             l1_loss = torch.mean(
#                 torch.mean(
#                     torch.sum(
#                         (
#                             torch.abs(
#                                 target_hidden_states.unsqueeze(dim=0)
#                                 - compression_hidden_states.unsqueeze(dim=2)
#                             )
#                         ),
#                         dim=-1,
#                     ),
#                     dim=-1,
#                 )
#             )
#             loss = (1 - hybrid_alpha) * ce_loss + hybrid_alpha * l1_loss
#         elif loss_type == "cosine":
#             # TODO: Cosine distance
#             cosine = ...
#             loss = ce_loss + hybrid_alpha * (1.0 - cosine)
#         else:
#             raise ValueError(f"Unsupported loss_type: {self.args.loss_type}")
