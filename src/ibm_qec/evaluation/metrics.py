import math

import torch

from ibm_qec.device import DEVICE

HIGHLIGHT_MAX_LIKELIHOOD_FACTOR = 1000.0


def _binomial_likelihood_bounds(k: int, n: int, highlight_factor: float = HIGHLIGHT_MAX_LIKELIHOOD_FACTOR):
    """Return lower/upper bounds on p where Bin(n,p) pdf drops by highlight_factor from the peak."""
    if n <= 0:
        return None, None

    if highlight_factor <= 1:
        raise ValueError("highlight_factor must be greater than 1 to define a meaningful interval")

    if k < 0 or k > n:
        raise ValueError("k must lie within [0, n]")

    if k == 0:
        upper = 1.0 - highlight_factor ** (-1.0 / n)
        upper = max(upper, 0.0)
        return 0.0, min(upper, 1.0)
    if k == n:
        lower = highlight_factor ** (-1.0 / n)
        lower = min(lower, 1.0)
        return max(lower, 0.0), 1.0

    p_hat = k / n

    log_comb = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

    def log_pdf(p: float) -> float:
        if p <= 0.0 or p >= 1.0:
            return float('-inf')
        return log_comb + k * math.log(p) + (n - k) * math.log(1.0 - p)

    log_peak = log_pdf(p_hat)
    target = log_peak - math.log(highlight_factor)

    lo, hi = 0.0, p_hat
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mid <= 0.0:
            break
        if log_pdf(mid) >= target:
            hi = mid
        else:
            lo = mid
    lower_bound = max(hi, 0.0)

    lo, hi = p_hat, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mid >= 1.0:
            break
        if log_pdf(mid) >= target:
            lo = mid
        else:
            hi = mid
    upper_bound = min(lo, 1.0)

    return lower_bound, upper_bound


def evaluate_decoder(model, data_loader):
    """
    Evaluates the decoder's performance, including LER and confidence metrics
    broken down by the initial logical state (|0> vs |1>).
    """
    model.eval()

    total_shots_0, total_shots_1 = 0, 0
    logical_errors_0, logical_errors_1 = 0, 0

    total_pre_conf_0, total_post_conf_0 = 0.0, 0.0
    total_pre_conf_1, total_post_conf_1 = 0.0, 0.0

    with torch.no_grad():
        for syndromes, system_graph, labels, initial_logical_states, final_measured_data in data_loader:
            syndromes = syndromes.to(DEVICE)
            system_graph = system_graph.to(DEVICE)
            final_measured_data = final_measured_data.to(DEVICE)
            initial_logical_states = initial_logical_states.to(DEVICE)

            D_current = final_measured_data.shape[1]
            if D_current == 0:
                continue

            pred_probs = model(syndromes, system_graph)
            pred_probs_x = pred_probs[:, 0, :D_current]
            correction_frame = (pred_probs_x > 0.5).long()
            corrected_data_bits = (final_measured_data + correction_frame) % 2

            pre_correction_ones = torch.sum(final_measured_data, dim=1)
            pre_correct_bit_count = torch.where(initial_logical_states == 1, pre_correction_ones,
                                                D_current - pre_correction_ones)

            post_correction_ones = torch.sum(corrected_data_bits, dim=1)
            post_correct_bit_count = torch.where(initial_logical_states == 1, post_correction_ones,
                                                 D_current - post_correction_ones)

            mask_initial_0 = (initial_logical_states == 0)
            mask_initial_1 = (initial_logical_states == 1)

            pre_conf_batch = pre_correct_bit_count.float() / D_current
            post_conf_batch = post_correct_bit_count.float() / D_current

            total_pre_conf_0 += torch.sum(pre_conf_batch[mask_initial_0]).item()
            total_post_conf_0 += torch.sum(post_conf_batch[mask_initial_0]).item()

            total_pre_conf_1 += torch.sum(pre_conf_batch[mask_initial_1]).item()
            total_post_conf_1 += torch.sum(post_conf_batch[mask_initial_1]).item()

            decoded_logical_state = torch.round(post_correction_ones.float() / D_current).long()
            is_logical_error = (decoded_logical_state != initial_logical_states)

            logical_errors_0 += torch.sum(is_logical_error & mask_initial_0).item()
            total_shots_0 += torch.sum(mask_initial_0).item()
            logical_errors_1 += torch.sum(is_logical_error & mask_initial_1).item()
            total_shots_1 += torch.sum(mask_initial_1).item()

    ler_0 = logical_errors_0 / total_shots_0 if total_shots_0 > 0 else 0
    ler_1 = logical_errors_1 / total_shots_1 if total_shots_1 > 0 else 0

    total_shots = total_shots_0 + total_shots_1
    accuracy = 1.0 - ((logical_errors_0 + logical_errors_1) / total_shots) if total_shots > 0 else 0

    avg_pre_conf_0 = total_pre_conf_0 / total_shots_0 if total_shots_0 > 0 else 0
    avg_post_conf_0 = total_post_conf_0 / total_shots_0 if total_shots_0 > 0 else 0

    avg_pre_conf_1 = total_pre_conf_1 / total_shots_1 if total_shots_1 > 0 else 0
    avg_post_conf_1 = total_post_conf_1 / total_shots_1 if total_shots_1 > 0 else 0

    ler_bounds_0 = _binomial_likelihood_bounds(logical_errors_0, total_shots_0) if total_shots_0 else (None, None)
    ler_bounds_1 = _binomial_likelihood_bounds(logical_errors_1, total_shots_1) if total_shots_1 else (None, None)
    ler_bounds_all = _binomial_likelihood_bounds(logical_errors_0 + logical_errors_1, total_shots) if total_shots else (None, None)

    return {
        "ler_0": ler_0,
        "ler_1": ler_1,
        "accuracy": accuracy,
        "avg_pre_conf_0": avg_pre_conf_0,
        "avg_post_conf_0": avg_post_conf_0,
        "avg_pre_conf_1": avg_pre_conf_1,
        "avg_post_conf_1": avg_post_conf_1,
        "ler_0_bound_low": ler_bounds_0[0],
        "ler_0_bound_high": ler_bounds_0[1],
        "ler_1_bound_low": ler_bounds_1[0],
        "ler_1_bound_high": ler_bounds_1[1],
        "overall_ler_bound_low": ler_bounds_all[0],
        "overall_ler_bound_high": ler_bounds_all[1],
        "shots_0": total_shots_0,
        "shots_1": total_shots_1,
        "shots_total": total_shots,
        "logical_errors_0": logical_errors_0,
        "logical_errors_1": logical_errors_1,
        "logical_errors_total": logical_errors_0 + logical_errors_1,
        "highlight_factor": HIGHLIGHT_MAX_LIKELIHOOD_FACTOR,
    }
