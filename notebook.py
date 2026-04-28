# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "gdrive-fsspec==0.4.0",
#     "ipython==9.12.0",
#     "marimo>=0.23.2",
#     "matplotlib==3.10.8",
#     "numpy==2.4.4",
#     "pillow==12.2.0",
#     "scikit-learn==1.8.0",
#     "torch==2.11.0",
# ]
# ///

import marimo

__generated_with = "0.23.0"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_circles

    return make_circles, mo, nn, np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # An adventure through the geometry of noise

    Generative modelling is one of the most ubiquitous tools in machine learning. If you've used a tool to generate images, video, or text in the last few years, you've almost certainly interacted with a generative model.

    Many of them share a common trick: corrupt clean data with noise, then train a neural network to undo that corruption. Whether it's a DDPM removing noise step-by-step or Flow Matching learning a velocity field from noise to data, the model always needs to be told how much noise it's looking at. This is called noise conditioning.


    Recently, an important question is being asked: do we actually need the noise?

    Autonomous (or "blind") models drop the noise input entirely and learn a single function $f(x)$ that operates the same way regardless of how corrupted the input is. Surprisingly, this works! But why it works--the mechanism-- is far from intuitively obvious. If the model doesn't know the noise level, how does it know what to do?

    [The Geometry of Noise: Why Diffusion Models Don’t Need Noise Conditioning](https://www.alphaxiv.org/abs/2602.18428) [Sahraee-Ardakan et al.] is a recent paper from a team at Google that attempts to answer this question.

    This notebook is a visual and interactive walkthrough -- not just of their elegant proof, but also the field of diffusion and its evolution.

    I will assume a surface-level understanding of neural networks, mathematics, and Python. However, we will go through exactly as much as we need out of DDPMs, Flow Matching, EqMs (blind models), then tie it all together for the final proof.
    """)
    return


@app.cell(hide_code=True)
def _():
    from PIL import Image
    from IPython.display import display, Markdown, Image as IPImage

    display(Image.open("./public/image.png"))
    display(Markdown("**Figure 1.** Source: EveryPixel - https://journal.everypixel.com/ai-image-statistics"))
    return IPImage, display


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## But first, DDPMs

    Before we can understand why noise conditioning might be unnecessary, we need to understand what it is and why it was thought to be essential! In this chapter, we're going to build a DDPM from scratch on a toy 2D dataset. We'll also generate some samples, all on CPU.

    Denoising Diffusion Probabilistic Models (DDPMs) [Ho et al. 2020]() are one of the most influential solutions to the generative modelling problem, and power popular frameworks like StableDiffusion.

    Before we jump into any math to explain DDPMs, lets load in our dataset.

    ### The dataset
    We'll use a simple 2D distribution: concentric circles. This is a useful dataset because it lets us visualise everything directly (and to illustrate concentration effects). This is nothing fancy: just points arranged in 2 rings.

    ### The task
    Don't forget, our task here is: given samples from this distribution of points, learn to produce *new* points that land **on** these circles.
    """)
    return


@app.cell
def _(make_circles, plt, torch):
    def sample_data(n_samples=1000):
        """sample from two concentric circles"""
        x, labels = make_circles(n_samples=n_samples, noise=0.05, factor=0.5)
        x = torch.tensor(x, dtype=torch.float32)
        x = x * 2.0 # we scale the data so its variance is comparable to our noise schedule later
        return x, labels

    data, labels = sample_data(4000)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap="coolwarm", s=5, alpha=0.6)
    ax.set_aspect("equal")
    ax.set_title("Our data: concentric circles")
    plt.show()
    return ax, data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The forward process: adding noise
    The core idea of a DDPM is to define a *forward process* that gradually destroys data by adding Gaussian noise. Given a clean data point $x_0$, we produce a noisy version $x_t$ at timestep $t$:

    $$x_t = \sqrt{\hat\alpha_t x_0} + \sqrt{1-\hat\alpha_t}\epsilon, \epsilon \sim N(0, I)$$

    where $\hat\alpha_t$ is a cumulative product of a noise schedule that controls how quickly signal is destroyed. At $t = 0$, $\hat\alpha_0 \approx 1$ and $x_t \approx x_0$ (almost clean).

    At $t = T$, $\hat\alpha_T \approx 0$ and $x_t \approx \epsilon$ (almost pure noise)

    **Notation**: In case you're confused by some of the notation, here's an exhaustive list of the notation used and what it means:
    - $T$: final timestep
    - $\epsilon$: random noise sample drawn from a standard normal distribution
    - $\epsilon \sim N(0, I)$: noise has zero mean and unit variance; basically, the Gaussian distribution has a peak at 0 and a spread that isn't too narrow or too wide

    Run the cell below to see what a Gaussian distribution with zero mean and unit variance looks like.
    """)
    return


@app.cell
def _(np, plt):
    def gaussian(x, mu=0.0, sigma=1.0):
        """define the Gaussian function
        mu: float = mean
        sigma: float = variance
        """
        return (1.0 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

    x = np.linspace(-5, 5, 500)
    y = gaussian(x, mu=0.0, sigma=1.0)

    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("density")
    plt.title("Gaussian (Normal) Distribution")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That's it. The forward process is not learned! Its just a fixed way of corrupting data.

    Lets now define the noise schedule and apply it to our distribution of points.
    """)
    return


@app.cell
def _(torch):
    def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
        """linear schedule defined in Ho et al. 2020, original DDPM paper"""
        return torch.linspace(beta_start, beta_end, timesteps)

    T = 200 # total timesteps
    betas = linear_beta_schedule(T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0) # this was our alpha_hat_t variable in the equation

    def f_sample(x_0, t, noise=None):
        """forward process: sample x_t given x_0 and timestep t"""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = torch.sqrt(alphas_cumprod[t]).unsqueeze(-1)
        sqrt_one_minus = torch.sqrt(1.0 - alphas_cumprod[t]).unsqueeze(-1)

        return sqrt_alpha_bar * x_0 + sqrt_one_minus * noise, noise

    return T, alphas, alphas_cumprod, betas, f_sample


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lets visualise what this looks like. Run the cell below!
    """)
    return


@app.cell
def _(data, f_sample, plt, torch):
    fig_2, axes = plt.subplots(1, 5, figsize=(20, 4))
    timesteps_to_show = [0, 30, 80, 150, 199]

    for _ax, t_val in zip(axes, timesteps_to_show):
        t = torch.full((data.shape[0],), t_val, dtype=torch.long)
        x_t, _ = f_sample(data, t)
        _ax.scatter(x_t[:, 0], x_t[:, 1], s=3, alpha=0.5)
        _ax.set_xlim(-3, 3)
        _ax.set_ylim(-3, 3)
        _ax.set_aspect("equal")
        _ax.set_title(f"t = {t_val}")

    plt.suptitle("Forward process: data → noise", fontsize=14)
    plt.tight_layout()
    plt.show()
    return (t_val,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As you can see, the circles gradually dissolve into a blob of Gaussian noise. This is the forward process doing its job!

    The thing I'd like you to keep in mind as we proceed is that: at every intermediate step, the data has a *specific* noise level determined by that $\hat\alpha_t$ value. This is information the model will rely on later when we start generating samples.

    ### The model: a noise predictor

    The reverse process needs a neural network that, given a noisy input $x_t$ and the timestep $t$, predicts the noise $\epsilon$ that was added. For our 2D toy problem, we'll just train a simple multilayer perceptron (MLP).

    The timestep $t$ is encoded using sinusoidal embeddings (borrowed from the Transformer) so the network can distinguish clearly between different noise levels.

    The intuition here is that sinusoidal embeddings turn the timestep $t$ (or noise level) into a richer signal the network can use. Instead of feeding in a single number, we map $t$ into many values using sine an cosine waves at different frequencies.
    """)
    return


@app.cell
def _(nn, np, torch):
    class SinusoidalEmbedding(nn.Module):
        """encode the timestep t into a vector using sine/cosine functions"""

        def __init__(self, dim):
            super().__init__()
            self.dim = dim # total embedding dim

        def forward(self, t):
            # use half for cos and half for sin
            half = self.dim // 2

            freqs = torch.exp(
                -np.log(10000) * torch.arange(half, device=t.device) / half
            )
            # sequence of frequencies:
            # freqs[i] = exp(-log(10000) * i / half) = 10000^(-i / half)
            # exponentially decreasing frequencies across dims

            args = t[:, None].float() * freqs[None]
            return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    class DDPM(nn.Module):
        """mlp that predicts noise given (x_t, t)"""
        def __init__(self, data_dim=2, hidden_dim=128, time_dim=32):
            super().__init__()

            self.time_embed = SinusoidalEmbedding(time_dim)

            self.net = nn.Sequential(
                nn.Linear(data_dim + time_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, data_dim)
            )

        def forward(self, x, t):
            t_emb = self.time_embed(t)
            return self.net(torch.cat([x, t_emb], dim=-1))


    return DDPM, SinusoidalEmbedding


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Again, I will draw your attention to the fact that the model takes in both $x$ and $t$ as input. This is noise conditioning.

    ### Training the model
    The training loop is exceedingly simple! Each step:
    1. Sample a batch of clean data $x_0$
    2. Sample a random timestep $t$ for each sample
    3. Sample noise $\epsilon$ and create the noisy input $x_t$
    4. Predict the noise: $\hat\epsilon = f_0(x_t, t)$
    5. Minimise ${||\hat\epsilon - \epsilon||}^2$

    Add noise -> predict the noise that was added.
    """)
    return


@app.cell
def _(DDPM, T, data, f_sample, nn, np, plt, torch):
    model = DDPM()
    optim = torch.optim.Adam(model.parameters(), lr=2e-4) # we need an optimiser for our gradient descent
    num_steps = 15000

    losses = []

    for step in range(num_steps):
        # 1. sample a batch
        idx = torch.randint(0, len(data), (256,))
        _x_0 = data[idx]

        # 2. sample random timesteps
        t_step = torch.randint(0, T, (256,))

        # 3. add noise
        _x_t, noise = f_sample(_x_0, t_step)

        # 4. predict the noise
        noise_pred = model(_x_t, t_step)

        # 5. mean squared error loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.append(loss.item())

    plt.figure(figsize=(8, 3))
    plt.plot(losses, alpha=0.3)
    plt.plot(np.convolve(losses, np.ones(100)/100, mode="valid"), color="black")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.show()
    return model, num_steps


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that our model is trained, let's start sampling data from it.

    ### Sampling: the reverse process

    Sampling runs the forward process but in reverse. Starting from pure noise $x_T \sim N(0, I)$, we iteratively denoise using the model's predictions. At each step t, the model predicts the noise in $x_t$, and we take a step toward a cleaner version.

    The DDPM sampling update (from Ho et al. 2020) at each timestep is:
    $$x_{t-1} = 1/{\sqrt{\alpha_t}} (x_t - (\beta_t/{\sqrt{1-\hat\alpha_t}})\epsilon_0(x_t, t)) + \sigma_t z$$

    where $z \sim N(0, I)$ is fresh noise added at each step (except the last), and $sigma_t$ = $\sqrt{\beta_t}$.

    **Note**: The $\sigma_tz$ term comes from the original DDPM derivation. The forward process was stochastic, so the mathematically "correct" reverse is also stochastic. The $\sigma_t=\beta_t\sigma_t = \sqrt{\beta_t}$ ties the injected randomness to the noise schedule: more randomness early on (broad exploration), less later (precise denoising).

    But this isn't the only way to sample. Later work by Song et al. (2021) showed that there's an equivalent deterministic process — a probability flow ODE — that produces samples from the *same distribution* without injecting any noise during sampling. Diversity is preserved because each different starting point $x_0$ still maps to a different output. We'll revisit this distinction in the next chapter.
    """)
    return


@app.cell
def _(T, alphas, alphas_cumprod, betas, torch):
    # lets define how we will sample

    @torch.no_grad()
    def p_sample(model, x_t, t_idx):
        """single reverse step: x_t -> x_(t-1)"""

        t_tensor = torch.full((x_t.shape[0],), t_idx, dtype=torch.long)
        eps_pred = model(x_t, t_tensor)

        beta_t = betas[t_idx]
        alpha_t = alphas[t_idx]
        alpha_bar_t = alphas_cumprod[t_idx]

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred
        )

        if t_idx > 0:
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(beta_t) * noise

        return mean

    @torch.no_grad()
    def sample(model, n_samples=1000):
        """full reverse process: noise -> data."""
        x = torch.randn(n_samples, 2)
        trajectory = [x.clone()]

        for t_idx in reversed(range(T)):
            x = p_sample(model, x, t_idx)
            if t_idx % 30 == 0:
                trajectory.append(x.clone())

        return x, trajectory

    return (sample,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lets now generate samples and compare them to the real data.
    """)
    return


@app.cell
def _(data, model, plt, sample):
    samples, traj = sample(model, n_samples=2000)

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 5))
    _axes[0].scatter(data[:, 0], data[:, 1], s=3, alpha=0.5)
    _axes[0].set_title("Real data")
    _axes[0].set_xlim(-3, 3)
    _axes[0].set_ylim(-3, 3)
    _axes[0].set_aspect("equal")

    _axes[1].scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.5, color="orange")
    _axes[1].set_title("Generated samples")
    _axes[1].set_xlim(-3, 3)
    _axes[1].set_ylim(-3, 3)
    _axes[1].set_aspect("equal")

    plt.tight_layout()
    plt.show()
    return samples, traj


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Run the cell below to visualise the reverse trajectory and how samples evolve from noise to data!
    """)
    return


@app.cell
def _(plt, traj):
    _fig, _axes = plt.subplots(1, len(traj), figsize=(3 * len(traj), 3))
    for _i, (_ax, snap) in enumerate(zip(_axes, traj)):
        _ax.scatter(snap[:, 0], snap[:, 1], s=3, alpha=0.4)
        _ax.set_xlim(-3, 3)
        _ax.set_ylim(-3, 3)
        _ax.set_aspect("equal")
        _ax.set_title(f"Step {_i}")

    plt.suptitle("Reverse process: noise → data", fontsize=14)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Immediately, you'll notice that in 10 steps, we're getting very close to the data distribution. The fact that we're approximating order out of the disordered mess we start in is incredible! This is a good time to remind yourself that we essentially created a function that takes in random noise and outputs something meaningful.

    ### Takeaways before things get harder
    Three things to keep in mind as we move forward:
    1. The model receives $t$ at every step. It needs to know the noise level to calibrate its prediction. Predicting noise at $t=10$ is very different from predicting noise at $t=290$.
    2. Sampling requires running through all $T$ timesteps sequentially. Each step is one network evaluation, so generation costs scale linearly with $T$.
    3. The reverse process adds fresh stochasticity (remember our $\sigma_tz$) at each step. Our generation is effectively solving a *stochastic* process. The same starting noise can produce different outputs.

    **Notes on stochasticity**: That $\sigma_tz$ term we add at each reverse step is deceptively important. It defines the mathematical nature of the sampling process. Whenever you have a process that evolves over time *and* gets random kicks at each step, you're solving a Stochastic Differential Equation (SDE).

    The DDPM sampler is a discretised SDE where the model provides the deterministic "drift" direction and the $\sigma_tz$ term provides the random "diffusion".

    It turns out, though, that every SDE of this type has a corresponding Ordinary Differential Equation (ODE) that traces out the same overall distribution without any randomness. We can essentially get a deterministic mapping from noise to data without sacrificing diversity. The randomness is still in the initial sample, and the only randomness that gets removed is the trajectory randomness. What we gain, in return, is access to the full mathematical toolkit available to ODE solvers: adaptive step sizes, higher-order methods, and crucially, fewer network evaluations for pretty much the same quality.

    Flow Matching is one such generative method that skips the SDE entirely and directly learns the vector field of an ODE that transports noise to data. Let's explore this in more detail!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to match a flow
    In the last section, we ended with an observation: if sampling from a DDPM is really just solving an ODE, why bother with an SDE detour at all? You can just learn the ODE directly.

    That's the idea behind Flow Matching (Lipman et al., 2023)[link]

    ### From noise schedules to straight lines
    Recall the DDPM forward process: $x_t = \sqrt{\hat\alpha_t}x_0 + \sqrt{1-\hat\alpha_t} \epsilon$. The coefficients $\sqrt{\hat\alpha_t}$ and $\sqrt{1-\hat\alpha_t}$ follow a specific schedule that was carefully designed.

    The resulting paths through space, from data to noise, are actually curved! This curvature means the ODE solver needs many steps to track the trajectory accurately.

    To see what I mean by *curved*, run the code below.
    """)
    return


@app.cell
def _(np, plt):
    _x0 = np.array([2.0, 0.0])  # 1 clean point
    _eps = np.array([0.0, 2.0])  # 1 noise vector

    # simple linear schedule
    _T = 200
    _beta = np.linspace(1e-4, 0.02, _T)
    _alpha = 1.0 - _beta
    _alpha_hat = np.cumprod(_alpha)

    # generate traj
    _xs = []
    for _t in range(_T):
        ah = _alpha_hat[_t]
        _xt = np.sqrt(ah) * _x0 + np.sqrt(1 - ah) * _eps
        _xs.append(_xt)

    _xs = np.stack(_xs)

    plt.plot(_xs[:, 0], _xs[:, 1])
    plt.scatter(*_x0, label="x0 (data)")
    plt.scatter(*_eps, label="epsilon (noise)")
    plt.title("DDPM forward trajectory (curved path)")
    plt.legend()
    plt.axis("equal")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Flow matching takes a much simpler approach. Instead of a complex diffusion schedule, define the path from noise to data as a straight line:

    $$x_t = (1-t)\epsilon+tx_0$$

    At $t = 0$, you have pure noise. At $t = 1$, you have clean data $x_0$, and everything in between is linear interpolation. We're not having to track schedules or cumulative products anymore.

    If you remember high school physics, the velocity along this path would be how fast and in what direction a point is moving. This is also the derivative with respect to $t$:

    $$v=dx_t/dt = x_0 -\epsilon$$

    The velocity is constant along each path, and it points from noise toward data, with a magnitude equal to the distance between the two points. All our model needs to do now is learn the "velocity field."

    **Note**: you can think of a velocity field as a function that assigns a velocity vector to every point $x$ at every time $t$.

    ### The training objective
    Where previously we trained a DDPM to predict noise, we're now training a model to predict the velocity (approximate the velocity field). Let's quickly remind ourselves of the objective: we're still trying to learn how to sample from a data distribution. With FM, we're trying to learn a velocity field so we can move samples $x_t$ at time $t$ toward higher data density.

    The training objective is just as simple:

    $$L_{FM} = ||f_0(x_t, t) - (x_0 - \epsilon)||^2$$

    Here's the algorithm:
    1. Sample clean data $x_0$ and noise $\epsilon$
    2. Sample a random time $t \in [0,1]$
    3. Interpolate $x_t = (1-t)\epsilon + tx_0$
    4. Predict the velocity $\hat{v}=f_0(x_t, t)$
    5. Minimise $||\hat{v}-(x_0-\epsilon)||^2$

    Let's build this out in code!
    """)
    return


@app.cell
def _(SinusoidalEmbedding, nn, torch):
    def fm_sample_xt(x_0, eps, t):
        """linear interpolation between noise and data"""

        t = t.unsqueeze(-1)
        return (1-t) * eps + t * x_0

    class Flow(nn.Module):
        """mlp that predicts velocity given (x_t, t)"""
        def __init__(self, data_dim=2, hidden_dim=256, time_dim=64):
            super().__init__()
            self.time_embed = SinusoidalEmbedding(time_dim)
            self.net = nn.Sequential(
                nn.Linear(data_dim + time_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, data_dim)
            )

        def forward(self, x, t):
            t_emb = self.time_embed(t)
            return self.net(torch.cat([x, t_emb], dim=-1))

    return Flow, fm_sample_xt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You'll pretty quickly notice that our architecture is very similar to the DDPM we built before. The main change is what the model is predicting: velocity instead of noise.

    Incidentally, this is one of the reasons why Flow Matching quickly became the preferred option in much of the generative modelling research.
    """)
    return


@app.cell
def _(Flow, data, fm_sample_xt, nn, np, num_steps, plt, torch):
    fm_model = Flow()
    fm_optim = torch.optim.Adam(fm_model.parameters(), lr=2e-4)

    fm_losses = []

    for _step in range(num_steps):
        _idx = torch.randint(0, len(data), (512,))
        _x_0 = data[_idx]
        _eps = torch.randn_like(_x_0)

        # t is continuous in [0, 1], no discrete timesteps
        _t = torch.rand(512)

        _x_t = fm_sample_xt(_x_0, _eps, _t)
        target_velocity = _x_0 - _eps

        velocity_pred = fm_model(_x_t, _t)
        _loss = nn.functional.mse_loss(velocity_pred, target_velocity)

        fm_optim.zero_grad()
        _loss.backward()
        fm_optim.step()

        fm_losses.append(_loss.item())

    plt.figure(figsize=(8, 3))
    plt.plot(fm_losses, alpha=0.3)
    plt.plot(np.convolve(fm_losses, np.ones(100)/100, mode="valid"), color="black")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Flow Matching training loss")
    plt.show()
    return (fm_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sampling: solving the ODE

    Now that our model is trained, we can start sampling data from it. As a reminder: the ODE just describes the dynamical system you get when you follow the velocity field we just learned.

    So, when we're sampling data, we're "solving the ODE".

    There are many ways you can solve an ODE (sample the data, these phrases are synonymous as far as Flow Matching is concerned). The easiest way is Euler's method.

    **Notes on Euler's  method**:
    ODE solver, given by the equation:

    $$x_{t+\Delta t}=x_t+\Delta tv_0(x_t, t)$$

    In English:
    - take the current point $x_t$
    - evaluate the velocity $v_0(x_t, t)$
    - move a small step $\Delta t$ in that direction

    Simple! Here's that algorithm in code.
    """)
    return


@app.cell
def _(torch):
    @torch.no_grad()
    def fm_sample(model, n_samples=1000, steps=100):
        """generate samples by solving the ODE with Euler's method"""
        x = torch.randn(n_samples, 2)
        dt = 1.0 / steps
        trajectory = [x.clone()]

        for i in range(steps):
            t = torch.full((n_samples,), i * dt)
            v = model(x, t)
            x = x + v * dt
            if i % (steps // 10) == 0:
                trajectory.append(x.clone())

        trajectory.append(x.clone()) # we're storing the trajectory as we travel through the velocity field for observability
        return x, trajectory

    return (fm_sample,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compare this to DDPM sampling. Immediately, you can tell that its simpler. You're keeping track of a lot less, and the algorithm is conceptually pretty intuitive. You're travelling from noise to data on the same big field.

    Let's generate and compare the two:
    """)
    return


@app.cell
def _(data, fm_model, fm_sample, plt, samples):
    fm_samples, fm_trajectory = fm_sample(fm_model, n_samples=2000, steps=100)

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 5))
    _axes[0].scatter(data[:, 0], data[:, 1], s=3, alpha=0.5)
    _axes[0].set_title("Real data")
    _axes[1].scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.5, color="orange")
    _axes[1].set_title("DDPM samples (300 steps)")
    _axes[2].scatter(fm_samples[:, 0], fm_samples[:, 1], s=3, alpha=0.5, color="green")
    _axes[2].set_title("Flow Matching samples (100 steps)")
    for _ax in _axes:
        _ax.set_xlim(-3, 3)
        _ax.set_ylim(-3, 3)
        _ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()
    return (fm_samples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why straight lines matter
    The reason Flow Matching can get away with fewer steps for the same generative capability comes down to the geometry of the paths. Let's visualise individual trajectories to see the difference.
    """)
    return


@app.cell
def _(fm_model, plt, torch):
    @torch.no_grad()
    def fm_trajectories(model, n_samples=20, steps=100):
        """track individual sample paths during generation"""
        x = torch.randn(n_samples, 2)
        dt = 1.0 / steps
        paths = [x.clone()]
        for i in range(steps):
            t = torch.full((n_samples,), i * dt)
            v = model(x, t)
            x = x + v * dt
            paths.append(x.clone())
        return torch.stack(paths)  # (steps+1, n_samples, 2)

    _paths = fm_trajectories(fm_model, n_samples=30, steps=100)

    _fig, _ax = plt.subplots(figsize=(6,6))
    for i in range(_paths.shape[1]):
        _traj_i = _paths[:, i, :].numpy()
        _ax.plot(_traj_i[:, 0], _traj_i[:, 1], alpha=0.4, linewidth=0.8)
        _ax.scatter(_traj_i[0, 0], _traj_i[0, 1], s=10, color="gray", zorder=5)
        _ax.scatter(_traj_i[-1, 0], _traj_i[-1, 1], s=10, color="black", zorder=5)

    _ax.set_xlim(-3, 3)
    _ax.set_ylim(-3, 3)
    _ax.set_aspect("equal")
    _ax.set_title("Flow Matching sample trajectories (noise → data)")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As you can see, the points going from gray -> black are converging towards circles. This is the straight path I'm referring to in the journey from noise to data.

    Because the interpolation is linear, the learned velocity field produces near-straight trajectories. A simple Euler solver can follow a straight line with very few steps.

    Locally, the velocity field behaves like "move in this fixed direction toward data", which makes the overall dynamics much smoother.

    Put yourself in the numerical solver's shoes! Straight, consistent motion is easier to follow than bending and changing directions. The efficiency of FMs comes from the simpler dynamics, no injected noise during sampling, and fewer steps needed to reach a good solution.

    ### The velocity field
    Before we move on, let's just visualise this velocity field we keep talking about! Reminder that the model learned a function $f_0(x, t)$ that assigns a velocity vector to every point in space at every time $t$.
    """)
    return


@app.cell
def _(fm_model, plt, torch):
    @torch.no_grad()
    def plot_vector_field(model, t_val, ax, grid_range=2, n_grid=20):
        """plot the learned velocity field at a given time"""
        x_grid = torch.linspace(-grid_range, grid_range, n_grid)
        y_grid = torch.linspace(-grid_range, grid_range, n_grid)
        xx, yy = torch.meshgrid(x_grid, y_grid, indexing="xy")
        points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        t = torch.full((points.shape[0],), t_val)

        v = model(points, t)

        ax.quiver(
            points[:, 0], points[:, 1],
            v[:, 0], v[:, 1],
            alpha=0.6
        )
        ax.set_xlim(-grid_range, grid_range)
        ax.set_ylim(-grid_range, grid_range)
        ax.set_aspect("equal")
        ax.set_title(f"Velocity field at t = {t_val:.1f}")

    _fig, _axes = plt.subplots(1, 4, figsize=(20, 5))
    for _ax, _tval in zip(_axes, [0.0, 0.3, 0.7, 1.0]):
        plot_vector_field(fm_model, _tval, _ax)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At $t=0$ (pure noise), the field pushes mass inward toward where the data lives. As $t$ increases and points are closer to the data, the field becomes more structured, pushing points onto the two circle distribution. At $t=1$, the field should be at near-zero. Now, in our case, we can see that the outer circle has mostly been approximated by our velocity field, but we could probably train for longer to achieve the inner circle.

    Alas, this is what Flow Matching learns: a time-varying vector field. As you have guessed, FM models are still conditioned on something, *time* to be specific. The model needs to know what $t$ value it is at to give the right velocity. Remember that as we start exploring the last part of this notebook!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Geometry of noise
    So far, we built two generative models - a DDPM and a flow matching model - and both produce reasonable samples of our circles. I'm going to reiterate what they both have in common: they both receive the timestep t as an input.

    The DDPM noise predictor takes $(x_t, t)$. The Flow Matching velocity predictor $(x_t, t)$. Neither of them would work without knowing this noise level $t$. Well actually, would they?

    ### Same point, different jobs
    Let's consider a point $x_t = (0.5, 0.3)$. If the model is told $t = 0.05$, it knows this point is almost certainly near the data manifold and should make a small, precise correction. If told $t=0.95$, it knows the point is almost pure noise and should push aggressively toward the data. The same input leads to different outputs, distinguished through $t$ as the mediator.

    Let's visualise this with our trained FM model.
    """)
    return


@app.cell
def _(ax, data, fm_model, np, plt, torch):
    _t_point = torch.tensor([[0.5, 0.3]])

    _fig, _ax = plt.subplots(figsize=(6,6))
    _ax.scatter(data[:, 0], data[:, 1], s=2, alpha=0.2, color="gray", label="Data")
    _ax.scatter(0.5, 0.3, s=80, color="black", zorder=10, label="Query point")

    _colors = plt.cm.viridis(np.linspace(0, 1, 10))
    for j, _t_val in enumerate(np.linspace(0.0, 1.0, 10)):
        _t_tensor = torch.full((1,), _t_val)
        with torch.no_grad():
            v = fm_model(_t_point, _t_tensor)
        _ax.arrow(
            0.5, 0.3, v[0, 0].item() * 0.15, v[0, 1].item() * 0.15, head_width=0.04, head_length=0.02, fc=_colors[j], ec=_colors[j]
        )

    _sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0,1))
    plt.colorbar(_sm, ax=ax, label="t")
    _ax.set_xlim(-2, 2)
    _ax.set_ylim(-2, 2)
    _ax.set_aspect("equal")
    _ax.set_title("Same point, different t → different velocities")
    _ax.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The arrows fan out in different directions and magnitudes because the model has learned a different vector field for every noise level. A continuous family of functions indexed by $t$! That extra dimension of implicit knowledge is what noise conditioning buys us.

    ### What if we just remove $t$?
    Let's run the experiment where the model only sees $x_t$, given by the loss function:

    $$L_{blind}=||f_0(x_t) - (x_0 - \epsilon)||^2$$
    """)
    return


@app.cell
def _(
    T,
    alphas,
    alphas_cumprod,
    betas,
    data,
    fm_sample_xt,
    fm_samples,
    nn,
    plt,
    torch,
):
    class BlindFM(nn.Module):
        def __init__(self, data_dim=2, hidden_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(data_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, data_dim)
            )

        def forward(self, x):
            return self.net(x)


    class BlindDDPM(nn.Module):
        def __init__(self, data_dim=2, hidden_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(data_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, data_dim)
            )

        def forward(self, x):
            return self.net(x)

    @torch.no_grad()
    def blind_ddpm_sample(model, n_samples=2000):
        x = torch.randn(n_samples, 2)
        for t_idx in reversed(range(T)):
            eps_pred = model(x)
            beta_t = betas[t_idx]
            alpha_t = alphas[t_idx]
            alpha_bar_t = alphas_cumprod[t_idx]
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred
            )
            if t_idx > 0:
                x = mean + torch.sqrt(beta_t) * torch.randn_like(x)
            else:
                x = mean
        return x


    blind_ddpm_model = BlindDDPM()
    blind_ddpm_opt = torch.optim.Adam(blind_ddpm_model.parameters(), lr=3e-4)
    blind_ddpm_losses = []

    for _step in range(5000):
        _idx = torch.randint(0, len(data), (256,))
        _x_0 = data[_idx]
        _t = torch.randint(0, T, (256,))
        _noise = torch.randn_like(_x_0)
        _sqrt_ab = torch.sqrt(alphas_cumprod[_t]).unsqueeze(-1)
        _sqrt_1mab = torch.sqrt(1.0 - alphas_cumprod[_t]).unsqueeze(-1)
        _x_t = _sqrt_ab * _x_0 + _sqrt_1mab * _noise

        _pred = blind_ddpm_model(_x_t)
        _loss = nn.functional.mse_loss(_pred, _noise)

        blind_ddpm_opt.zero_grad()
        _loss.backward()
        blind_ddpm_opt.step()
        blind_ddpm_losses.append(_loss.item())

    blind_model = BlindFM()
    blind_opt = torch.optim.Adam(blind_model.parameters(), lr=3e-4)
    blind_losses = []

    for _step in range(5000):
        _idx = torch.randint(0, len(data), (256,))
        _x_0 = data[_idx]
        _eps = torch.randn_like(_x_0)
        _t = torch.rand(256)
        _x_t = fm_sample_xt(_x_0, _eps, _t)
        _target = _x_0 - _eps
        _pred = blind_model(_x_t)
        _loss = nn.functional.mse_loss(_pred, _target)
        blind_opt.zero_grad()
        _loss.backward()
        blind_opt.step()
        blind_losses.append(_loss.item())

    @torch.no_grad()
    def blind_sample(model, n_samples=1000, steps=100):
        x = torch.randn(n_samples, 2)
        dt = 1.0 / steps
        for i in range(steps):
            v = model(x)
            x = x + v * dt
        return x

    blind_samples = blind_sample(blind_model, n_samples=2000)
    blind_ddpm_samples = blind_ddpm_sample(blind_ddpm_model)

    _fig, _axes = plt.subplots(1, 4, figsize=(20, 5))
    _axes[0].scatter(data[:, 0], data[:, 1], s=3, alpha=0.5)
    _axes[0].set_title("Real data")
    _axes[1].scatter(fm_samples[:, 0], fm_samples[:, 1], s=3, alpha=0.5, color="green")
    _axes[1].set_title("FM Conditioned")
    _axes[2].scatter(blind_samples[:, 0], blind_samples[:, 1], s=3, alpha=0.5, color="red")
    _axes[2].set_title("FM Blind (velocity)")
    _axes[3].scatter(blind_ddpm_samples[:, 0], blind_ddpm_samples[:, 1], s=3, alpha=0.5, color="purple")
    _axes[3].set_title("DDPM Blind (noise pred)")
    for _ax in _axes:
        _ax.set_xlim(-3, 3)
        _ax.set_ylim(-3, 3)
        _ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In 2 dimensions, the FM Blind model is clearly struggling to approximate the data distribution. It can't tell noise levels apart, and produces a muddled vector field that is trying to be the average of all the time-conditional fields at once.

    So far, this confirms our intuition: noise conditioning is necessary if you want the model to learn where to go in a given step.

    But hold on! Our data is 2D and our ambient space is 2D. Generative modelling doesn't always look like that; a 256x256 image lives in a 196,608-dimensional space, while the actual images occupy a tiny low-dimensional "manifold" within it. This turns out to be quite important.

    **Note**: I introduced a few new terms above, so thought I'd define them for the curious reader!
    - Ambient space: the full vector space in which the data is represented. For real-valued data with $d$ coordinates, this is $\mathbb{R}^d$. It is the space the model operates in.
    - Why is a 256x256 image high-dimensional? A grayscale image has one scalar per pixel: $256 x 256 = 65,536$ coordinates, so it is a point in $\mathbb{R}^{65,536}$. An RGB image has 3 channels, so $256 x 256 x 3 = 196,608$, so it lies in $\mathbb{R}^{196,608}$ dimensions. This is basically the number of dimensions needed to describe the image.
    - Manifold: A manifold describes the (often lower-dimensional) space within the higher-dimensional space where points live. The simplest analogy may be the Earth being a 3D object, though the surface is 2D.

    ### Climbing into higher dimensions
    Let's embed our same circles into progressively higher-dimensional spaces using a *random orthogonal projection*. This is a way to map low-dimensional data into a higher-dimensional space while preserving its geometry (distances and angles).

    The data is still 2D in its intrinsic structure (we're dealing with two circles), but we'll just inflate our ambient space to 8, 32, then 128 dimensions. This is the setup described in the paper.

    **Note on random orthogonal projection**:
    Take a matrix $Q \in \mathbb{R}^{D x d}$ whose columns are *orthonormal* (i.e. $Q^TQ=I_d$). A set of vectors is orthonormal if each vector has length 1 (normalised), and different vectors are perpendicular (their dot product is 0).

    Then a point $x\in\mathbb{R}^d$ is mapped to:
    $$y=Qx \in \mathbb{R}^D$$

    "Random" here means that $Q$ is constructed by sampling a random matrix and orthonormalising its columns (e.g. using a method called QR decomposition). The result is that the original 2D structure is rotated and embedded into a higher-dim space without distortion.
    """)
    return


@app.cell
def _(data, fm_sample_xt, np, plt, t_val, torch):
    def embed_data(data_2d, D, seed=42):
        """embedding data to different dims"""
        rng = np.random.RandomState(seed)
        if D == 2:
            return data_2d, np.eye(2)
        M = np.linalg.qr(rng.randn(D,D))[0][:, :2] # random orthogonal projection
        embedded = data_2d.numpy() @ M.T
        return torch.tensor(embedded, dtype=torch.float32), M

    def project_back(samples_hd, M):
        return samples_hd.numpy() @ M

    dims = [2, 8, 32, 128]
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_samples = 2000

    _fig, _axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 4))

    for _ax, D in zip(_axes, dims):
        data_hd, _ = embed_data(data, D)
        for _t_val in t_values:
            _eps = torch.randn(len(data_hd), D)
            _t_tensor = torch.full((len(data_hd),), _t_val)
            _x_t = fm_sample_xt(data_hd, _eps, _t_tensor)
            norms = torch.norm(_x_t, dim=1).numpy()
            _ax.hist(norms, bins=50, alpha=0.5, density=True, label=f"t={t_val}")
        _ax.set_title(f"D = {D}")
        _ax.set_xlabel("||x_t||")
        _ax.legend(fontsize=8)

    plt.suptitle("As D grows, noise levels become geometrically separable", fontsize=14)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We don't even need to train a blind model to see why this works. The fact that lets the model succeed is a property of the data geometry itself and it's visible just by looking at how noisy observations distribute in different dimensions.

    We've already shown that our blind model struggles to approximate the data distribution, so the two natural questions which arise here are:
    1. Why does this happen at higher dimensions?
    2. Why does it only happen for velocity-parameterised models?

    These two questions are the crux of the paper, and the entire purpose of our exercise today. Pat yourself on the back, we're in the home stretch now!

    ### Why dimensionality fixes the ambiguity
    Let's define our setup concretely:
    - The data lies on a low-dimensional manifold (here, a set of 2D circles)
    - It is embedded in a higher-dimensional ambient space $\mathbb{R}^D$
    - We observe a noisy point $u = x + noise$, where the nose is Gaussian

    The key phenomenon is *concentration of measure*: in $D$ dimensions, a Gaussian noise vector almost always has norm $\approx \sqrt{D}$. So instead of being spread out, noise lives on a thin spherical shell.

    Now, lets decompose the noisy point:
    - A signal component (on the 2D manifold)
    - A noise component (in the remaining $D - 2$ dimensions)

    As $D$ grows, the noise component dominates the norm:
    $$||u||\approx (noise \space magnitude) \sim \sqrt{D-2}$$

    Basically, **in high dimensions, where a point lies (its distance from the origin) is almost entirely determined by how much noise it contains**.

    Different noise levels $t$ scale this magnitude, so each $t$ produces points that lie on a distinct shell in space. This is to say: the observation $u$ (specifically, its norm) reveals the noise level.

    Formally, from the paper, the sample distribution over noise levels collapses:

    \[
    p(t \mid u) \to \delta(t - \hat{t}(u))
    \]

    So, the model doesn't need to explicitly condition on $t$ anymore: the geometry of the high-dimensional space implicitly encodes it. I find this incredibly elegant!

    ### The marginal energy landscape
    So we've answered why the noise levels separate so cleanly in higher dimensions. Recall that the second question is why does this only work in velocity parametrisation models like Flow Matching?

    Earlier, we saw that sampling a blind DDPM where $t$ was not given was disastrous. But, would embedding the data into higher dimensions result in the same recovery that we see with FMs, in the DDPM?

    The answer, surprisingly, is no! The paper proves this rigorously: blind DDPMs structurally fail no matter how high the dimensions go.

    To see why, we need to look at what's happening underneath an autonomous model. The paper proves that any autonomous model -- whether predicting noise, signal, or velocity -- is implicitly performing gradient descent on a single time-invariant landscape called the *marginal energy landscape*. This landscape is described by:

    $$
    E_{marg}(u) = -log \int{p(u|t) p(t) dt}
    $$

    This is the negative log-likelihood of a noisy observation $u$, averaged over all possible noise levels. You can imagine this as a curved space where the model rolls downhill.

    The shape of this landscape has a super interesting property: clean data points sit at the bottom of infinitely deep energy wells. The energy gradient diverges as you approach the data manifold; mathematically, it blows up at rate $O(1/b(t))$ where $b(t)$ is the noise scale.

    This sounds catastrophic. A divergent gradient should make gradient descent impossible, because you're taking steps proportional to the gradient -> the gradient is enormous -> you overshoot wildly -> the trajectory blows up. So how do these models stay stable at all?

    **Note**: Before we continue, let's define some terms:
    - Negative log-likelihood. The likelihood $p(u)$ is just "how probable is the point $u$?" If it looks like a real data point, $p(u)$ is high; if it's pure noise, $p(u)$ is low. Taking the negative log flips this around: high probability becomes low value, low probability becomes high value.
    $−\log{p(u)}$ measures how "unusual" or "far from data" a point is. It turns a probability into something we can think of as a cost.
    - Why "energy"? This is borrowed from physics. In physics, a ball rolls toward states of low potential energy — it falls into valleys. We use the same metaphor here: real data points have low "energy" (they're stable, the model wants to land there), and noisy points have high "energy" (they're unstable, the model wants to move away). When the paper says the model performs gradient descent on the marginal energy, it literally means the same thing as a ball rolling downhill on a landscape where data points are the valleys. Calling it "energy" instead of "negative log-likelihood" is mostly just nicer language for the same idea.
    - Big-O notation: When we say $O(1/b(t))$, we're saying "this thing grows roughly like $1/b(t)$ as $b(t)$ gets small." It's a measure of how fast something scales.
    - Time-invariant: A time-invariant function is one that doesn't change based on tt
    t. The same input always gives the same output. Compare this to a Flow Matching velocity field $v(x,t)$, which gives different outputs at different times even for the same $x$. The marginal energy landscape is time-invariant: it's *one* landscape, fixed once and for all, and the model navigates it without any notion of where it is in time.
    """)
    return


@app.cell
def _(np, plt):
    # -----------------------------
    # Marginal energy landscape! You'll find a similar plot in Figure 1 of the paper
    # -----------------------------

    # Example "data manifold" points
    data_points = np.array([
        [-0.55,  0.25],
        [ 0.50,  0.45],
        [ 0.25, -0.55],
    ])

    n_grid = 180
    u1 = np.linspace(-1.0, 1.0, n_grid)
    u2 = np.linspace(-1.0, 1.0, n_grid)
    U1, U2 = np.meshgrid(u1, u2)
    grid = np.stack([U1, U2], axis=-1)

    sigmas = np.linspace(0.04, 0.55, 40)

    p = np.zeros(U1.shape)

    for sigma in sigmas:
        sigma2 = sigma ** 2

        for _x in data_points:
            diff = grid - _x
            dist2 = np.sum(diff ** 2, axis=-1)

            p += np.exp(-dist2 / (2 * sigma2)) / (2 * np.pi * sigma2)

    p = p / len(sigmas) / len(data_points)

    # Marginal energy: E_marg(u) = -log p(u)
    _eps = 1e-12
    E = -np.log(p + _eps)

    E = E - np.min(E)

    E_clip = np.clip(E, 0, np.percentile(E, 97))

    _fig = plt.figure(figsize=(13, 5))

    _ax1 = _fig.add_subplot(1, 2, 1, projection="3d")
    _surf = _ax1.plot_surface(
        U1, U2, E_clip,
        cmap="inferno",
        linewidth=0,
        antialiased=True,
        alpha=0.95,
    )

    _ax1.scatter(
        data_points[:, 0],
        data_points[:, 1],
        np.zeros(len(data_points)),
        marker="o",
        s=45,
        color="black",
        label="Data manifold",
    )

    _ax1.set_title("3D Energy Landscape")
    _ax1.set_xlabel("$u_1$")
    _ax1.set_ylabel("$u_2$")
    _ax1.set_zlabel("$E_{marg}(u)$")
    _ax1.view_init(elev=35, azim=-55)

    _ax2 = _fig.add_subplot(1, 2, 2)
    _contour = _ax2.contourf(U1, U2, E_clip, levels=40, cmap="inferno")
    _ax2.contour(U1, U2, E_clip, levels=15, colors="white", linewidths=0.4, alpha=0.5)

    _ax2.scatter(
        data_points[:, 0],
        data_points[:, 1],
        marker="*",
        s=90,
        color="black",
        label="Data manifold",
    )

    _ax2.set_title("Contour View")
    _ax2.set_xlabel("$u_1$")
    _ax2.set_ylabel("$u_2$")
    _ax2.set_aspect("equal")
    _ax2.legend(loc="upper right", fontsize=8)

    _cbar = _fig.colorbar(_contour, ax=_ax2, fraction=0.046, pad=0.04)
    _cbar.set_label("$E_{marg}(u) = -\\log p(u)$")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Riemannian gradient flow
    The paper's resolution to the problem of model stability is interesting. Autonomous models aren't doing plain gradient descent. They're doing Riemannian gradient descent, which is a form of gradient descent on a curved space where the local metric depends on local geometry.

    The metric in this setting is called the posterior noise variance: at any point u, this measures the models' effective uncertainty about which noise level produced this observation.

    Near the data manifold, this variance shrinks to zero. The metric becomes infinitesimally fine-grained, which has the effect of shrinking the effective step size in exactly the direction the energy is steepest!

    So basically, our divergent gradient and our posterior noise variance cancel. The product of these two effects is the actual update applied to the sample, and it stays bounded!

    But this preconditioning isn't automatic. It only emerges if the model is parametrised compatibly with Riemannian geometry.

    **Note**: What does Riemannian geometry mean? A Riemannian geometry is just a way of measuring distances and directions on a curved space, where the rules can change from place to place. For example, on flat ground, one step always covers one metre. On a Riemannian space, one step might cover one metre in one location and a tiny fraction of a metre in another. The rules are local. In our setting, this means that as we approach the data manifold, the step sizes shrink automatically.

    ### The bounded vector field condition
    The authors analyse 3 common parameterisations and compute something called the "effective gain" $v(t)$. You can think of $v(t)$ as an error amplifier: any small mistake the model makes in its prediction gets multiplied by $v(t)$ when applied to the sampling trajectory. If $v(t)$ stays small, errors stay small. Otherwise, even tiny errors blow up the trajectory.

    Below is how the parameterisations stack up in terms of effective gain.
    """)
    return


@app.cell
def _(IPImage, display):
    _img_path = "/__modal/volumes/vo-UDwxvUzNMlOqeAoa0J26N6/Screenshot 2026-04-27 at 5.36.16 pm.png"

    def _disp(path):
        display(IPImage(filename=path, width=900))

    _disp(_img_path)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Velocity prediction has a constant gain of 1. Errors stay errors and they don't get amplified. Noise prediction has a gain that explodes as you approach the data manifold, exactly where you need it to be calm. A small estimation error gets multiplied by our divergent metric from before and as $b(t) \rightarrow 0$, that multiplier goes to infinity.

    This is the bounded vector field condition: for stable generation, you need a parameterisation whose gain stays bounded all the way to the data manifold. Velocity prediction is the only common parameterisation that satisfies this cleanly.

    So, we can finally answer our second question: blind Flow Matching works because velocity prediction is the geometrically correct way to parameterise the flow. Blind DDPMs fail because noise prediction amplifies errors most catastrophically at the very moment the model is closest to producing a clean sample.

    ## Closing
    We started this notebook with DDPMs, models that learn to remove noise step by step, conditioned on the noise level $t$. Then we looked at Flow Matching and reframed the process as solving an ODE, where we learn a velocity field, still conditioned on $t$.

    Then, we ended up confirming rather elegantly that $t$ was never needed. In high-dimensional spaces, the noise level is geometrically encoded in the input itself, and a velocity parameterised model can recover it implicitly. The model rolls downhill on a marginal energy landscape, the Riemannian metric tames the singularity at the data manifold, and the bounded gain of velocity prediction keeps the trajectory stable. Schedule-free, time-invariant generation falls out for free.

    The next generation of generative models are being trained on this exact paradigm -- Equilibrium Matching and other blind models -- and we've just learned the intuition behind why it works.

    Until next time!
    """)
    return


if __name__ == "__main__":
    app.run()
