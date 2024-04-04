import torch
import math
import einops

def fastmax_hack(q, k, v, p=1, mask=True):
    if not mask:

        # Normalize
        denum_term = 1
        q = q - torch.mean(q,dim = 3).unsqueeze(-1)
        k = k - torch.mean(k,dim = 3).unsqueeze(-1)
        qn = torch.linalg.norm(q, dim = 3)
        kn = torch.linalg.norm(k, dim = 3)
        q = q/torch.linalg.norm(qn, dim = 2, ord = float('inf')).unsqueeze(-1).unsqueeze(-1)
        k = k/torch.linalg.norm(kn, dim = 2, ord = float('inf')).unsqueeze(-1).unsqueeze(-1)


        first_term = torch.sum(v,-2)  # (b, h, d)
        second_term = torch.matmul(k.swapaxes(-2,-1),v)/denum_term  # (b, h, d, d)

        div1 = torch.ones([k.shape[0],k.shape[1],1,1], device=k.device)*k.shape[2] # (b, h, 1, 1)
        div2 = torch.sum(k,-2).unsqueeze(-1) # (b, h, d, 1)

        ans2 = torch.matmul(q,second_term)  # (b, h, n, d)
        div2 = torch.matmul(q,div2)/(denum_term) # (b, h, n, 1)

        ans = ans2 # (b, h, n, d)
        ans = torch.add(ans.permute(2,3,1,0) ,first_term.permute(2,1,0)).permute(3,2,0,1) # (b, h, n, d)
        div = div2 # (b, h, n, d)
        div = torch.add(div.permute(2,3,1,0) ,div1.permute(3,2,1,0)).permute(3,2,0,1) # (b, h, n, 1)
        ans = ans/div # (b, h, n, d)

        return ans


    else:
        # Normalize
        q = q - einops.reduce(q, "b h n d -> b h n ()", "mean")
        k = k - einops.reduce(k, "b h n d -> b h n ()", "mean")
        qn_norm = torch.sqrt(einops.einsum(q, q, "b h n d, b h n d -> b h n"))
        kn_norm = torch.sqrt(einops.einsum(k, k, "b h n d, b h n d -> b h n"))
        q = q / einops.reduce(qn_norm, "b h n -> b h () ()", "max")
        k = k / einops.reduce(kn_norm, "b h n -> b h () ()", "max")

        _, _, _, D = q.shape
        normalize_term = 1



        F = compute_F_masked(
            q, k, v, normalize_term, p
        )
        g = compute_g_masked(q, k, normalize_term, p)

        a = None
        o = F / einops.repeat(g, "b h n -> b h n d", d=D)
        output_tuple = o


        return output_tuple


def compute_F_masked(q, k, v, normalize_term, p, dropout_rate=0):
    """Computes numerator in factorized attention with mask

    For details on the formulas for computation, see Ryan's notes

    Args:
        q: (Query tensor) Shape (B, H, N, D)
        k: (Key tensor) Shape (B, H, N, D)
        v: (Value tensor) Shape (B, H, N, D)
        normalize_term: (Float) Term to divide score by for normalization
        p: (int) Degree of polynomial approximation to exponential function.
            Restricted to either 1 or 2.
        dropout_rate: (Float) Fraction of tensor to dropout. Between 0 and 1

    Returns:
        F: (Tensor) shape (B, H, N, D)
    """
    z1 = torch.cumsum(v, 2)
    F = z1
    
    kv = einops.einsum(k, v, "b h n m, b h n j -> b h n m j")
    z2 = torch.cumsum(kv, 2)
    F = F + einops.einsum(q, z2, "b h i m, b h i m j -> b h i j") / normalize_term

    if p == 2:
        kkv = einops.einsum(k, k, v, "b h n m, b h n l, b h n j -> b h n m l j")
        x3 = torch.cumsum(kkv, 2)
        F = F + einops.einsum(
            q, q, x3, "b h i m, b h i l, b h i m l j -> b h i j"
        ) / (2 * normalize_term**2)

    return F


def compute_g_masked(q, k, normalize_term, p):
    """Computes denominator in factorized attention with mask

    For details on the formulas for computation, see Ryan's notes

    Args:
        q: (Query tensor) Shape (B, H, N, D)
        k: (Key tensor) Shape (B, H, N, D)
        normalize_term: (Float) Term to divide score by for normalization
        p: (int) Degree of polynomial approximation to exponential function.
            Restricted to either 1 or 2.
        dropout_rate: (Float) Fraction of tensor to dropout. Between 0 and 1

    Returns:
        g: (Tensor) shape (B, H, N, D)
    """
    B, H, N, _ = q.shape

    z1 = torch.arange(N, device=q.device) + 1
    z1 = einops.repeat(z1, "n -> b h n", b=B, h=H)
    g = z1

    y2 = torch.cumsum(k, dim=2)
    z2 = einops.einsum(q, y2, "b h i m, b h i m -> b h i") / normalize_term
    g = g + z2

    if p == 2:
        x3 = einops.einsum(k, k, "b h n m, b h n l -> b h n m l")
        y3 = torch.cumsum(x3, dim=2)
        z3 = einops.einsum(q, q, y3, "b h i m, b h i l, b h i m l -> b h i") / (
            2 * normalize_term**2
        )
        g = g + z3

    return g