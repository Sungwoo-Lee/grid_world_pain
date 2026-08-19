"""Figures for the on-source decay-rule study."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np, json, pathlib

plt.rcParams.update({'figure.dpi': 130, 'font.size': 9, 'axes.titlesize': 9.5,
                     'savefig.bbox': 'tight', 'savefig.facecolor': 'white'})
TEAL, AMBER, PLUM, GREY = '#1f7a6c', '#c9852b', '#7b4b91', '#6e6579'
ONSRC = 2.0          # the constant in sensor.py:14
EPS = 0.001          # the guard threshold in sensor.py:14


def curve(d, g):
    return 1.0 / np.power(d, g)


# ------------------------------------------------------------------ FIG 0 ---
def fig0():
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5))
    g = 1.0

    ax = axes[0]
    d = np.linspace(0.35, 5.2, 400)
    ax.plot(d, curve(d, g), color=TEAL, lw=2.2, label='1 / distance   (the decay curve)')
    ax.axvspan(0.001, 1.0, color=GREY, alpha=.10)
    ax.text(0.52, 3.35, 'no grid cell can\nsit in here', ha='center', fontsize=8, color=GREY)
    ints = np.arange(1, 6)
    ax.plot(ints, curve(ints, g), 'o', ms=8, mfc=TEAL, mec='white', mew=1.2, zorder=5,
            label='where the sensor actually samples')
    ax.plot([0.5], [ONSRC], '*', ms=20, mfc=AMBER, mec='black', mew=.8, zorder=6,
            label='what the code returns ON the source')
    ax.annotate('the code says 2.0 here.\n1 / 0.5 = 2.0 — it lands\nexactly on the curve',
                xy=(0.62, ONSRC), xytext=(1.6, 1.45), fontsize=8.5, color='#8a5a12',
                arrowprops=dict(arrowstyle='->', color=AMBER, lw=1.4))
    ax.set_xlim(0, 5.3); ax.set_ylim(0, 4.0)
    ax.set_xlabel('distance from the source (cells)')
    ax.set_ylabel('smell strength')
    ax.set_title('γ = 1.0, the shipped setting — there is no jump')
    ax.legend(fontsize=7.6, loc='upper right', framealpha=.95); ax.grid(alpha=.22)

    ax = axes[1]
    dd = np.logspace(-4, 0.75, 700)
    code = np.where(dd < EPS, ONSRC, curve(np.maximum(dd, 1e-12), g))
    ax.plot(dd, code, color=PLUM, lw=2, label='what the code computes')
    ax.axvline(EPS, color=AMBER, lw=1.5, ls='--')
    ax.axvspan(1e-4, 1.0, color=GREY, alpha=.10)
    ax.annotate('THE discontinuity:\njust above 0.001 the value is 1000,\njust below it is 2.0',
                xy=(EPS, 60), xytext=(3e-3, 3.0), fontsize=8.5, color=PLUM,
                arrowprops=dict(arrowstyle='->', color=PLUM, lw=1.3))
    ax.plot(ints, curve(ints, g), 'o', ms=7, mfc=TEAL, mec='white', mew=1.1, zorder=5,
            label='reachable grid distances')
    ax.plot([1e-4], [ONSRC], '*', ms=17, mfc=AMBER, mec='black', mew=.7, zorder=6,
            label='distance exactly 0')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('distance from the source (cells, log scale)')
    ax.set_ylabel('smell strength (log)')
    ax.set_title('the same function, zoomed into the guard')
    ax.legend(fontsize=7.6, loc='lower left'); ax.grid(alpha=.22, which='both')

    fig.suptitle('FIG 0 — The rule contains a real discontinuity, and the grid can never reach it',
                 fontsize=11, y=1.03)
    fig.text(.5, -.09,
             'Left: at the shipped γ=1 the special-case value 2.0 is exactly what the curve gives at half a cell, so the readings a walking agent sees run smoothly through contact.\n'
             'Right: the code genuinely does jump — anywhere between distance 0.001 and 1 it returns enormous values, and below 0.001 it snaps to 2.0. But entities and cells both sit\n'
             'on integer coordinates, so the distance is only ever 0 or 1 or more. The grey band is unreachable. The discontinuity is real, and it is unreachable.',
             ha='center', fontsize=8.5)
    fig.savefig('fig0_where_the_jump_is.png'); plt.close(fig); print('ok fig0')


# ------------------------------------------------------------------ FIG 1 ---
def fig1():
    gammas = [0.5, 1.0, 2.0, 3.0]
    fig, axes = plt.subplots(1, len(gammas), figsize=(13.2, 3.9))
    for ax, g in zip(axes, gammas):
        d = np.linspace(0.35, 5.2, 400)
        ax.plot(d, curve(d, g), color=TEAL, lw=2)
        ints = np.arange(1, 6)
        ax.plot(ints, curve(ints, g), 'o', ms=6, mfc=TEAL, mec='white', mew=1)
        nat = curve(0.5, g)
        ax.plot([0.5], [ONSRC], '*', ms=17, mfc=AMBER, mec='black', mew=.7, zorder=6)
        ax.plot([0.5], [nat], 'o', ms=7, mfc='none', mec=PLUM, mew=1.8, zorder=6)
        if abs(nat - ONSRC) > 1e-9:
            ax.annotate('', xy=(0.5, nat), xytext=(0.5, ONSRC),
                        arrowprops=dict(arrowstyle='<->', color=PLUM, lw=1.5))
            ax.text(0.72, (nat+ONSRC)/2, f'off by\n{abs(nat-ONSRC):.2f}', fontsize=7.6, color=PLUM)
        ok = 'lands ON the curve' if abs(nat-ONSRC) < 1e-9 else 'MISSES the curve'
        ax.set_title(f"γ = {g:g}{'  (shipped)' if g == 1.0 else ''}\ncurve at ½ cell = {nat:.2f}"
                     f" · code says 2.00\n{ok}", fontsize=8.6)
        ax.set_ylim(0, max(4.6, nat*1.25)); ax.set_xlim(0, 5.3)
        ax.set_xlabel('distance (cells)'); ax.grid(alpha=.22)
        if g == gammas[0]:
            ax.set_ylabel('smell strength')
    fig.suptitle('FIG 1 — The constant 2.0 is only correct at one value of γ', fontsize=11, y=1.06)
    fig.text(.5, -.14,
             'The star is the hard-coded 2.0; the hollow circle is what the decay curve itself gives at half a cell. They coincide at γ=1 and nowhere else. At γ=2 the sensor reports\n'
             'HALF what its own curve implies for standing on a source; at γ=0.5 it reports 40% too much. The rule is not wrong today — it is wrong the moment anyone tunes γ,\n'
             'which the olfactory study explicitly contemplates as a way to buy near-field contrast.',
             ha='center', fontsize=8.5)
    fig.savefig('fig1_only_correct_at_one_gamma.png'); plt.close(fig); print('ok fig1')


# ------------------------------------------------------------------ FIG 2 ---
def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))
    for ax, g in zip(axes, (1.0, 2.0)):
        ds = [5, 4, 3, 2, 1]
        vals = [curve(d, g) for d in ds] + [ONSRC]
        labels = [f'{d}' for d in ds] + ['ON IT']
        cols = [TEAL]*5 + [AMBER]
        ax.bar(range(len(vals)), vals, color=cols, edgecolor='white', width=.68)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals)*.03, f'{v:.2f}', ha='center', fontsize=8)
        for i in range(len(vals)-1):
            r = vals[i+1]/vals[i]
            ax.annotate(f'×{r:.2f}', xy=(i+0.5, max(vals)*.80), ha='center',
                        fontsize=8.6, color=PLUM if abs(r-2.0) > .01 else '#8a5a12',
                        fontweight='bold' if g != 1.0 and i == len(vals)-2 else 'normal')
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels)
        ax.set_xlabel('distance from the source (cells)')
        ax.set_ylabel('reading at the agent’s own cell')
        verdict = ('final step matches the one before it —\nthe agent feels a smooth approach'
                   if g == 1.0 else
                   'the reading ACCELERATES (×4.00) then\nDECELERATES (×2.00) at contact')
        ax.set_title(f'γ = {g:g}{"  (shipped)" if g == 1.0 else ""}\n{verdict}', fontsize=9)
        ax.set_ylim(0, max(vals)*1.28)
    fig.suptitle('FIG 2 — What an agent walking onto food actually feels', fontsize=11, y=1.04)
    fig.text(.5, -.1,
             'Numbers above the bars are the readings; numbers between them are the step-to-step multiplier. At γ=1 the last two steps are both ×2.00 — contact is the natural\n'
             'continuation of the approach, and nothing about it is surprising. At γ=2 the approach builds to ×4.00 and then the contact step is only ×2.00: the smell gets stronger\n'
             'more slowly at the exact moment the agent arrives. THAT is the kink worth caring about, and it appears only when γ is changed.',
             ha='center', fontsize=8.5)
    fig.savefig('fig2_walking_onto_food.png'); plt.close(fig); print('ok fig2')


# ------------------------------------------------------------------ FIG 3 ---
def fig3():
    gs = np.linspace(0.4, 3.0, 300)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(gs, np.full_like(gs, ONSRC), color=AMBER, lw=2.4,
            label='A · keep the hard-coded 2.0   (today)')
    ax.plot(gs, curve(0.5, gs), color=TEAL, lw=2.4,
            label='B · half-cell floor:  1 / (½)^γ')
    ax.plot(gs, curve(1.0, gs), color=PLUM, lw=2.2, ls='--',
            label='C · one-cell floor:  1 / 1^γ = 1.0')
    ax.axvline(1.0, color='0.55', lw=1.2, ls=':')
    ax.plot([1.0], [2.0], 'o', ms=11, mfc='none', mec='black', mew=1.8, zorder=6)
    ax.annotate('A and B are bit-identical here,\nand only here',
                xy=(1.0, 2.0), xytext=(1.35, 3.4), fontsize=8.6,
                arrowprops=dict(arrowstyle='->', color='0.35', lw=1.2))
    ax.text(0.96, 1.15, 'shipped γ', fontsize=8, color='0.4', rotation=90, ha='right')
    ax.set_yscale('log'); ax.set_xlabel('decay power γ')
    ax.set_ylabel('value returned when standing ON a source')
    ax.set_title('FIG 3 — Three ways to define "standing on it", across γ', fontsize=10.5)
    ax.legend(fontsize=8.2, loc='upper left'); ax.grid(alpha=.22, which='both')
    fig.text(.5, -.13,
             'Option B says "treat being on the source as being half a cell away" — the grid\'s own resolution limit, and the same floor the visual blur needs for the same reason.\n'
             'It passes through exactly 2.0 at γ=1, so it is byte-identical to today under the shipped configuration, and it stays on the curve at every other γ where A does not.\n'
             'Option C removes the special case entirely by flooring at one cell, which is smooth but erases any distinction between standing on food and standing beside it.',
             ha='center', fontsize=8.5)
    fig.savefig('fig3_three_options.png'); plt.close(fig); print('ok fig3')


# ------------------------------------------------------------------ FIG 4 ---
def fig4():
    freq_p = pathlib.Path('frequency.json')
    freq = json.loads(freq_p.read_text()) if freq_p.exists() else None
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))

    ax = axes[0]
    food = np.array([-1., 0.])          # one cell north of the agent
    for r, lab in ((0, 'today'), (1, 'after the change')):
        pass
    cells1 = np.array([(dr, dc) for dr in range(-1, 2) for dc in range(-1, 2)
                       if abs(dr)+abs(dc) <= 1], float)
    for (rr, cc) in cells1:
        on = (rr, cc) == (food[0], food[1])
        ax.add_patch(Rectangle((cc-.46, -rr-.46), .92, .92,
                               facecolor=AMBER if on else '#dbe7e4',
                               edgecolor='0.7', lw=.7))
    ax.text(0, 1.06, '2.00', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(0, 0.06, '1.00', ha='center', va='center', fontsize=10)
    for (rr, cc) in [(0., 1.), (0., -1.), (1., 0.)]:
        ax.text(cc, -rr, f'{1/np.linalg.norm([rr-food[0], cc-food[1]]):.2f}',
                ha='center', va='center', fontsize=9, color='0.35')
    ax.plot(0, -0.30, marker='X', ms=10, mfc='#111', mec='white', mew=1)
    ax.plot(food[1], -food[0] - 0.30, 'o', ms=9, mfc='#4ad97a', mec='black', mew=.7)
    ax.text(1.35, 1.0, 'food sits in the\nnorth cell — that\nsampling point is\nON the source',
            fontsize=8.2, va='center')
    ax.set_xlim(-2.0, 3.4); ax.set_ylim(-2.0, 2.2)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('the new situation: a NEIGHBOUR cell on a source\n'
                 '(today only the centre cell could be)', fontsize=9)

    ax = axes[1]
    if freq:
        rs = [0, 1, 2]
        pct = [freq['pct'][str(r)] for r in rs]
        bars = ax.bar([f'range {r}\n({2*r*r+2*r+1} cells)' for r in rs], pct,
                      color=[GREY, TEAL, PLUM], edgecolor='white', width=.6)
        for b, v in zip(bars, pct):
            ax.text(b.get_x()+b.get_width()/2, v + max(pct)*.03, f'{v:.1f}%',
                    ha='center', fontsize=10, fontweight='bold')
        ax.set_ylabel('% of steps where the rule fires')
        ax.set_ylim(0, max(pct)*1.25)
        ax.set_title(f'measured on the real environment\n'
                     f'{freq["total_agent_steps"]:,} agent-steps, shipped config', fontsize=9)
        ax.grid(alpha=.22, axis='y')
    else:
        ax.text(.5, .5, 'run measure_frequency.py first', ha='center', transform=ax.transAxes)
        ax.axis('off')
    fig.suptitle('FIG 4 — How often this actually happens', fontsize=11, y=1.02)
    fig.text(.5, -.08,
             'Only entities with a non-zero chemical signature count: rocks and hiding predators ship all-zero signatures, so the 2.0 multiplies to nothing for them.\n'
             'The rule stops being a rare contact event and becomes a routine feature of the observation — which is what makes it worth deciding deliberately rather than inheriting.',
             ha='center', fontsize=8.5)
    fig.savefig('fig4_how_often.png'); plt.close(fig); print('ok fig4')


for f in (fig0, fig1, fig2, fig3, fig4):
    f()
