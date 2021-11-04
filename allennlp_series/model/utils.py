import torch
import torch.nn.functional as F


def get_text_field_mask(
    text_field_tensors, num_wrapping_dims: int = 0
) -> torch.LongTensor:
    # if "mask" in text_field_tensors:
    print("text_field_tensors = ", text_field_tensors)
    return text_field_tensors["tokens"] != 0
    # return text_field_tensors["mask"]


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert (
        logits.dim() == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


# # Here is how to use this function for top-p sampling
# temperature = 1.0
# top_k = 0
# top_p = 0.9
#
# # Get logits with a forward pass in our model (input is pre-defined)
# logits = model(input)
#
# # Keep only the last token predictions of the first batch item (batch size 1), apply a temperature coefficient and filter
# logits = logits[0, -1, :] / temperature
# filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
#
# # Sample from the filtered distribution
# probabilities = F.softmax(filtered_logits, dim=-1)
# next_token = torch.multinomial(probabilities, 1)
