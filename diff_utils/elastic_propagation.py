
import torch

ROOF = 1.0e50
FLOOR = 1.0e-50
IPOWER_R = 50

class ElasticPropagationFn(torch.autograd.Function):

    @staticmethod
    def forward(
        B1: torch.Tensor,
        B2: torch.Tensor,
        B3: torch.Tensor,
        B4: torch.Tensor,
        rho: torch.Tensor,
        x: float,
        y_init: torch.Tensor,
        h_step: float,
        n_steps: int,
        loc_start: int,
        going_up: bool,
    ):
        y_out, y_history, i_power = _forward_impl(
            B1.detach(),
            B2.detach(),
            B3.detach(),
            B4.detach(),
            rho.detach(),
            x,
            y_init.detach(),
            h_step,
            n_steps,
            loc_start,
            going_up,
        )
        return y_out, y_history, torch.tensor(i_power, dtype=torch.int64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        B1, B2, B3, B4, rho, x, y_init, h_step, n_steps, loc_start, going_up = inputs
        y_out, y_history, i_power_t = output
        ctx.x = x
        ctx.h_step = h_step
        ctx.n_steps = n_steps
        ctx.loc_start = loc_start
        ctx.going_up = going_up
        ctx.save_for_backward(B1, B2, B3, B4, rho, y_init, y_history)
        ctx.mark_non_differentiable(y_history, i_power_t)

    @staticmethod
    def backward(ctx, grad_y_out, grad_y_history, grad_i_power):
        B1, B2, B3, B4, rho, y_init, y_history = ctx.saved_tensors

        grad_B1, grad_B2, grad_B3, grad_B4, grad_rho, grad_y_init = _backward_impl(
            grad_y_out,
            B1,
            B2,
            B3,
            B4,
            rho,
            ctx.x,
            y_history,
            ctx.h_step,
            ctx.n_steps,
            ctx.loc_start,
            ctx.going_up,
        )

        return (
            grad_B1,
            grad_B2,
            grad_B3,
            grad_B4,
            grad_rho,
            None,
            grad_y_init,
            None,
            None,
            None,
            None,
        )

def _forward_impl(B1, B2, B3, B4, rho, x, y_init, h_step, n_steps, loc_start, going_up):
    two_x = 2.0 * x
    two_h = 2.0 * h_step
    four_h_x = 4.0 * h_step * x
    dtype = y_init.dtype
    device = y_init.device

    y = y_init.clone()
    y_history = torch.empty(n_steps + 1, 5, dtype=dtype, device=device)
    y_history[0] = y

    i_power = 0
    j = loc_start + (n_steps if going_up else 0)

    xB3 = x * B3[j] - rho[j]

    z = torch.empty(5, dtype=dtype, device=device)
    z[0] = y[0] - 0.5 * (B1[j] * y[3] - B2[j] * y[4])
    z[1] = y[1] - 0.5 * (-rho[j] * y[3] - xB3 * y[4])
    z[2] = y[2] - 0.5 * (two_h * y[3] + B4[j] * y[4])
    z[3] = y[3] - 0.5 * (xB3 * y[0] + B2[j] * y[1] - two_x * B4[j] * y[2])
    z[4] = y[4] - 0.5 * (rho[j] * y[0] - B1[j] * y[1] - four_h_x * y[2])

    for step in range(n_steps):
        if going_up:
            j -= 1
        else:
            j += 1

        x_save = y.clone()
        y = z.clone()
        z = x_save

        xB3 = x * B3[j] - rho[j]

        z[0] = z[0] - (B1[j] * y[3] - B2[j] * y[4])
        z[1] = z[1] - (-rho[j] * y[3] - xB3 * y[4])
        z[2] = z[2] - (two_h * y[3] + B4[j] * y[4])
        z[3] = z[3] - (xB3 * y[0] + B2[j] * y[1] - two_x * B4[j] * y[2])
        z[4] = z[4] - (rho[j] * y[0] - B1[j] * y[1] - four_h_x * y[2])

        if step < n_steps - 1:
            if dtype in (torch.complex64, torch.complex128):
                scale_val = abs(z[1].real.item())
            else:
                scale_val = abs(z[1].item())
            if scale_val < FLOOR:
                z *= ROOF
                y *= ROOF
                i_power -= IPOWER_R
            elif scale_val > ROOF:
                z *= FLOOR
                y *= FLOOR
                i_power += IPOWER_R

        y_history[step + 1] = y

    return z.clone(), y_history, i_power

def _backward_impl(
    grad_y_out,
    B1,
    B2,
    B3,
    B4,
    rho,
    x,
    y_history,
    h_step,
    n_steps,
    loc_start,
    going_up,
):
    N = B1.shape[0]
    dtype = y_history.dtype
    device = y_history.device
    two_x = 2.0 * x
    two_h = 2.0 * h_step
    four_h_x = 4.0 * h_step * x

    grad_B1 = torch.zeros(N, dtype=dtype, device=device)
    grad_B2 = torch.zeros(N, dtype=dtype, device=device)
    grad_B3 = torch.zeros(N, dtype=dtype, device=device)
    grad_B4 = torch.zeros(N, dtype=dtype, device=device)
    grad_rho = torch.zeros(N, dtype=dtype, device=device)

    d_z = grad_y_out.clone()
    d_y = torch.zeros(5, dtype=dtype, device=device)

    j_sequence = []
    j = loc_start + (n_steps if going_up else 0)
    j_sequence.append(j)
    for step in range(n_steps):
        if going_up:
            j -= 1
        else:
            j += 1
        j_sequence.append(j)

    for step in range(n_steps - 1, -1, -1):
        j = j_sequence[step + 1]
        xB3 = x * B3[j] - rho[j]

        y_at_step = y_history[step + 1]

        d_y_from_z = torch.zeros(5, dtype=dtype, device=device)
        d_y_from_z[0] += -d_z[3] * xB3 - d_z[4] * rho[j]
        d_y_from_z[1] += -d_z[3] * B2[j] + d_z[4] * B1[j]
        d_y_from_z[2] += d_z[3] * two_x * B4[j] + d_z[4] * four_h_x
        d_y_from_z[3] += -d_z[0] * B1[j] + d_z[1] * rho[j] - d_z[2] * two_h
        d_y_from_z[4] += d_z[0] * B2[j] + d_z[1] * xB3 - d_z[2] * B4[j]

        grad_B1[j] += -d_z[0] * y_at_step[3] + d_z[4] * y_at_step[1]
        grad_B2[j] += d_z[0] * y_at_step[4] - d_z[3] * y_at_step[1]
        grad_B3[j] += -d_z[3] * x * y_at_step[0] - d_z[1] * x * y_at_step[4]
        grad_B4[j] += -d_z[2] * y_at_step[4] + d_z[3] * two_x * y_at_step[2]
        grad_rho[j] += d_z[1] * y_at_step[3] + d_z[1] * y_at_step[4]
        grad_rho[j] += -d_z[4] * y_at_step[0] + d_z[3] * y_at_step[0]

        new_d_z = d_y + d_y_from_z
        new_d_y = d_z.clone()
        d_z = new_d_z
        d_y = new_d_y

    j0 = j_sequence[0]
    xB3_0 = x * B3[j0] - rho[j0]
    y0 = y_history[0]

    d_y_init = d_y.clone()
    d_y_init[0] += d_z[0]
    d_y_init[3] += -0.5 * d_z[0] * B1[j0]
    d_y_init[4] += 0.5 * d_z[0] * B2[j0]
    d_y_init[1] += d_z[1]
    d_y_init[3] += 0.5 * d_z[1] * rho[j0]
    d_y_init[4] += 0.5 * d_z[1] * xB3_0
    d_y_init[2] += d_z[2]
    d_y_init[3] += -0.5 * d_z[2] * two_h
    d_y_init[4] += -0.5 * d_z[2] * B4[j0]
    d_y_init[3] += d_z[3]
    d_y_init[0] += -0.5 * d_z[3] * xB3_0
    d_y_init[1] += -0.5 * d_z[3] * B2[j0]
    d_y_init[2] += 0.5 * d_z[3] * two_x * B4[j0]
    d_y_init[4] += d_z[4]
    d_y_init[0] += -0.5 * d_z[4] * rho[j0]
    d_y_init[1] += 0.5 * d_z[4] * B1[j0]
    d_y_init[2] += 0.5 * d_z[4] * four_h_x

    grad_B1[j0] += -0.5 * d_z[0] * y0[3] + 0.5 * d_z[4] * y0[1]
    grad_B2[j0] += 0.5 * d_z[0] * y0[4] - 0.5 * d_z[3] * y0[1]
    grad_B4[j0] += -0.5 * d_z[2] * y0[4] + 0.5 * d_z[3] * two_x * y0[2]
    grad_B3[j0] += -0.5 * d_z[3] * x * y0[0] - 0.5 * d_z[1] * x * y0[4]
    grad_rho[j0] += 0.5 * d_z[1] * y0[3] + 0.5 * d_z[1] * xB3_0 / rho[j0]
    grad_rho[j0] += -0.5 * d_z[4] * y0[0]

    return grad_B1, grad_B2, grad_B3, grad_B4, grad_rho, d_y_init

def elastic_propagation(
    B1: torch.Tensor,
    B2: torch.Tensor,
    B3: torch.Tensor,
    B4: torch.Tensor,
    rho: torch.Tensor,
    x: float,
    y_init: torch.Tensor,
    h_step: float,
    n_steps: int,
    loc_start: int,
    going_up: bool = True,
) -> tuple[torch.Tensor, int]:
    y_out, _, i_power_t = ElasticPropagationFn.apply(
        B1,
        B2,
        B3,
        B4,
        rho,
        x,
        y_init,
        h_step,
        n_steps,
        loc_start,
        going_up,
    )
    return y_out, int(i_power_t.item())

__all__ = ["ElasticPropagationFn", "elastic_propagation"]
