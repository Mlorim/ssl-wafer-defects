"""
losses.py

Consistency loss (Mean Teacher), Supervised Contrastive Loss (Khosla et al. 2020,
формула (2) из статьи Wei et al. про mean teacher), EMA update для teacher
модели, и VAE loss (ELBO) для статьи Wei et al. про latent vector representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def consistency_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """
    Consistency loss между student и teacher predictions на unlabeled данных.
    Используем MSE на softmax-вероятностях (классический вариант из
    Tarvainen & Valpola, 2017 — "Mean teachers are better role models").

    teacher_logits трактуются как target и не пропускают градиент назад
    (это должно быть уже обеспечено тем, что teacher forward идёт под
    torch.no_grad() на стороне метода, но detach() ставим для надёжности).
    """
    student_probs = F.softmax(student_logits, dim=1)
    teacher_probs = F.softmax(teacher_logits, dim=1).detach()
    return F.mse_loss(student_probs, teacher_probs)


def supcon_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised Contrastive Loss, формула (2) из статьи:

        L = sum_i [ -1/|P(i)| * sum_{p in P(i)} log( exp(f_i . f_p / tau) /
                                                        sum_{a in A(i)} exp(f_i . f_a / tau) ) ]

    embeddings: [B, D] L2-нормализованные эмбеддинги (уже нормализованы в ProjectionHead)
    labels:     [B] class labels (int64)

    Ожидается, что B содержит несколько views на каждый сэмпл (например,
    2*N, где каждая пара соседних индексов — это (view1, view2) одного
    объекта), но формула работает для любого батча, где positive pair
    определяется совпадением label (как описано в статье: "the supervised
    contrastive loss views them as a positive pair if they come from the
    same category").
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]

    labels = labels.contiguous().view(-1, 1)
    if labels.shape[0] != batch_size:
        raise ValueError("Число labels должно совпадать с числом embeddings")

    # mask[i,j] = 1 если label_i == label_j (включая i==j)
    mask = torch.eq(labels, labels.T).float().to(device)

    # similarity matrix, температурное масштабирование
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    # численная стабильность: вычитаем максимум по строке
    sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - sim_max.detach()

    # A(i) = все элементы кроме самого себя
    logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
    mask = mask * logits_mask  # P(i): positive pairs, исключая i==i

    exp_sim = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    # среднее log-prob по positive pairs для каждого i
    mask_sum = mask.sum(dim=1)
    # избегаем деления на 0 для сэмплов без positive pairs в батче (класс встретился 1 раз)
    valid = mask_sum > 0
    mean_log_prob_pos = torch.zeros(batch_size, device=device)
    mean_log_prob_pos[valid] = (mask[valid] * log_prob[valid]).sum(dim=1) / mask_sum[valid]

    loss = -mean_log_prob_pos[valid].mean() if valid.any() else torch.tensor(0.0, device=device)
    return loss


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1.0,
):
    """
    ELBO для WaferVAE (Wei et al., "Latent Vector Representation"):
    recon_loss (BCE, т.к. wafer map в [0, 1] и декодер заканчивается sigmoid)
    + kl_weight * KL(q(z|x) || N(0, 1)).

    Возвращает (total_loss, {"recon_loss":..., "kl_loss":...}) для логирования по фазам.
    """
    recon_loss = F.binary_cross_entropy(recon, target, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + kl_weight * kl_loss
    return total, {"recon_loss": recon_loss.item(), "kl_loss": kl_loss.item()}


def inverse_sqrt_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    w_c = 1/sqrt(n_c) (формула 14 статьи SemiWaferNet, weighted cross-entropy
    для дисбаланса классов). Нормализуется так, чтобы среднее по классам = 1 —
    это не указано в статье явно, но не даёт общему масштабу loss произвольно
    смещаться при пересчёте весов на каждом этапе (после добавления pseudo-label
    сэмплов состав классов меняется).
    """
    counts = torch.bincount(labels, minlength=num_classes).float().clamp(min=1)
    weights = 1.0 / torch.sqrt(counts)
    return weights / weights.mean()


def imq_kernel(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """Inverse multiquadratic kernel k(u,v) = c / (c + ||u-v||^2), см. MM-WAE, раздел 3.4.2."""
    dist_sq = torch.cdist(x, y, p=2).pow(2)
    return c / (c + dist_sq)


def mmd_loss(z: torch.Tensor, z_prior: torch.Tensor, c: float = None) -> torch.Tensor:
    """
    MMD между латентными векторами энкодера z и сэмплами из прайора z_prior
    (N(0,I)) с inverse multiquadratic kernel (формула статьи MM-WAE, раздел
    3.4.2): MMD = mean(k(z,z)) + mean(k(z_prior,z_prior)) - 2*mean(k(z,z_prior)).
    Диагональные члены (i==j) не исключаются — статья суммирует по всем парам
    i,j без исключений.

    c — константа IMQ kernel; статья не даёт числового значения, по умолчанию
    используется эвристика Tolstikhin et al. (2018, WAE-MMD): c = 2 * latent_dim.
    """
    if c is None:
        c = 2.0 * z.shape[1]
    k_zz = imq_kernel(z, z, c)
    k_pp = imq_kernel(z_prior, z_prior, c)
    k_zp = imq_kernel(z, z_prior, c)
    return k_zz.mean() + k_pp.mean() - 2 * k_zp.mean()


def class_frequency_weights(labels: torch.Tensor, num_classes: int, alpha: float = 1.5) -> torch.Tensor:
    """
    w_c = 1/log(alpha + pi_c), pi_c — доля класса c в переданном наборе меток
    (формула статьи MM-WAE, раздел 3.4.3). alpha статья задаёт только
    качественно ("alpha > 1 as a smoothing constant"), без точного значения —
    используется разумный дефолт 1.5. Нормализуется на среднее, чтобы общий
    масштаб classification loss не смещался произвольно.
    """
    counts = torch.bincount(labels, minlength=num_classes).float()
    pi = counts / counts.sum().clamp(min=1)
    weights = 1.0 / torch.log(alpha + pi).clamp(min=1e-6)
    return weights / weights.mean()


def modality_consistency_loss(gate_weights: torch.Tensor, branch_weights: torch.Tensor) -> torch.Tensor:
    """
    Delta = mean(gate_weights, по батчу) - branch_weights (формула статьи
    MM-WAE, раздел 3.4.4: w — per-sample gating веса энкодера, усредняются по
    батчу для сравнения с branch_weights — глобальным learnable вектором
    классификатора, одинаковым для всех сэмплов). L_cons = Var(Delta) + Std(Delta).
    """
    delta = gate_weights.mean(dim=0) - branch_weights.mean(dim=0)
    return delta.var(unbiased=False) + delta.std(unbiased=False)


@torch.no_grad()
def ema_update(student: nn.Module, teacher: nn.Module, alpha: float):
    """
    EMA обновление весов teacher модели по формуле (1) из статьи:

        theta_teacher(t) = alpha * theta_teacher(t-1) + (1 - alpha) * theta_student(t)
    """
    student_params = dict(student.named_parameters())
    for name, teacher_param in teacher.named_parameters():
        student_param = student_params[name]
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=(1.0 - alpha))

    # синхронизируем buffers (batchnorm running stats) напрямую копированием
    student_buffers = dict(student.named_buffers())
    for name, teacher_buffer in teacher.named_buffers():
        teacher_buffer.data.copy_(student_buffers[name].data)