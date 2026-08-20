# Why the same code gives different answers

**Shareable page:** https://claude.ai/code/artifact/ad0fd4f9-2215-4a95-9e19-ac2e4bb1f8bc

## Purpose

This note explains why two runs of what looks like identical numerical code can produce
answers that differ in the last decimal place — and why running the *same compiled
program* twice never does. It is general knowledge about how computers store numbers, not
specific to this project, and it is written for a researcher who writes code but has never
had to think about floating-point representation.

It exists because the question came up concretely here: the trajectory-collection pipeline
found one column of environment values differing by about six hundredths of a millionth
between two runs. That was initially diagnosed as a bug in how episodes were batched. It
was not a bug at all — it was this. The project-specific record of that incident is in the
LLM wiki (`20260820_1606_reset_ulp_divergence_is_compiler_fusion`); this note is the
background knowledge that makes it obvious rather than alarming.

Every figure below was measured on NumPy 2.3.5 / Python 3.11.14 / x86-64 on 2026-08-20,
not quoted from a source. The code that produced them is at the end.

---

## The one idea

Mathematics has infinitely many numbers between 0 and 1. A computer has a fixed number of
bits, so it can store only a finite list of them. Everything else is rounded to the nearest
value on that list.

That is the whole subject. Every surprise below follows from it.

The gap between neighbouring storable values is called a **ULP** ("unit in the last
place"). It is not constant — it grows with magnitude, because the same handful of
significant digits has to cover ever larger numbers.

| near value | gap to the next storable `float32` |
|---|---|
| 0.001 | 0.000000000116 |
| 1.0 | 0.000000119 |
| 100 | 0.00000763 |
| 10 000 | 0.000977 |
| 1 000 000 | 0.0625 |
| 1 000 000 000 | **64.0** |

At a billion, a 32-bit float cannot distinguish 1 000 000 000 from 1 000 000 032.

---

## What each type gives you

| type | bits | decimal digits | largest value | whole numbers exact up to |
|---|---|---|---|---|
| `float64` | 64 | 15 | 1.8 × 10³⁰⁸ | 9 007 199 254 740 992 |
| `float32` | 32 | 6 | 3.4 × 10³⁸ | 16 777 216 |
| `float16` | 16 | 3 | 65 504 | 2 048 |
| `int64` | 64 | exact | 9.2 × 10¹⁸ | its whole range |
| `int32` | 32 | exact | 2 147 483 647 | its whole range |
| `int16` | 16 | exact | 32 767 | its whole range |
| `int8` | 8 | exact | 127 | its whole range |

A 16-bit float cannot count past 2 048 — ask for `2048 + 1` and you get 2 048 back. A
32-bit float stops counting whole numbers at about 16.8 million, which many real datasets
exceed.

---

## Experiment 1 — `0.1 + 0.2`

One tenth cannot be written exactly in binary, any more than one third can be written
exactly in decimal.

```
float64   0.1 + 0.2 = 0.30000000000000004441
          0.3       = 0.29999999999999998890   not equal

float32   0.1 + 0.2 = 0.30000001192092895508
          0.3       = 0.30000001192092895508   equal, by luck

float16   0.1 + 0.2 = 0.29980468750000000000
          0.3       = 0.30004882812500000000   not equal
```

Note the middle case. `float32` passes not because it is more careful but because its ticks
are coarse enough that both sides land on the same one. **Fewer digits made the equality
test pass.** A passing float comparison therefore proves less than it appears to.

---

## Experiment 2 — addition is not associative

School algebra says `(a+b)+c` equals `a+(b+c)`. On a computer they routinely differ,
because each step rounds. 500 000 random triples per type:

| type | triples where the two groupings disagree |
|---|---|
| `float64` | 22.7% |
| `float32` | 24.5% |
| `float16` | 24.5% |
| `int64` | **0%** |

About a quarter of the time, for every float type. This is normal behaviour, not an edge
case. Integers, being exact, never disagree.

A concrete case found by random search, in `float32`:

```
a = 0.16527635   b = 8.132703   c = 9.127556

(a + b) + c  =  17.425535202
a + (b + c)  =  17.425533295      difference 1.9e-06
```

### The consequence: summation order matters

The same 200 000 numbers, four orderings, `float32`, against a `float64` reference:

| order | result | error |
|---|---|---|
| reference (`float64`) | 99 777.806926 | — |
| front to back | 99 776.648438 | 1.158 |
| back to front | 99 778.015625 | 0.209 |
| smallest first | 99 779.093750 | 1.287 |
| pairwise (NumPy default) | 99 777.812500 | **0.006** |

Two lessons. The error is *not* negligible — over a whole unit on a sum of 100 000. And
how you add matters enormously: pairwise summation is roughly 200× more accurate than the
obvious loop, for free.

There is theory behind that. Worst-case summation error grows like `n` for the naive loop,
like `log n` for pairwise summation, and is *independent of n* for compensated (Kahan)
summation, which carries a running correction term. Higham is the standard treatment.

This is also why summing an array in parallel across many cores gives a different total
than summing it in order.

---

## Experiment 3 — the compiler may change your arithmetic

This is the one that catches people out, because the source code does not change.

Consider `mean + std * noise`. Two legitimate ways to compute it:

1. Multiply `std * noise`, **round**, add `mean`, **round**.
2. Do both in one *fused* instruction, keeping full precision in the middle, **round once**.

Compilers choose freely between them based on what is fastest given surrounding code.

```
noise = -0.8005198                                  (float32)

multiply, round, add, round  =  0.459844053
fused, round once            =  0.459844023
                                difference: 1 ULP
```

| type | draws differing | worst difference | in ULPs |
|---|---|---|---|
| `float64` | 0.0% | 0 | 0 |
| `float32` | 14.2% | 1.19e-07 | 2 |
| `float16` | 13.6% | 9.77e-04 | 2 |

### Which one is right?

The fused one. Computing the expression in exact rational arithmetic — infinite precision,
no rounding — and rounding once reproduces the fused answer exactly:

```
exact, infinite precision  =  0.459844031328
round once  (fused)        =  0.459844023      nearer the truth
round twice (separate)     =  0.459844053      further away
```

So the compiler is not corrupting your arithmetic; it is choosing the *more* accurate of
two valid readings.

### The rule the compiler follows

The C standard permits an expression to be **contracted** — intermediates kept at higher
precision — and scopes that permission to a single statement:

```c
float r = a*b + c;        // one statement  — may fuse, rounds once
float tmp = a*b;          // two statements — tmp is a real float,
float r   = tmp + c;      // so this rounds twice
```

**Splitting one line into two can change the answer.** So can the compiler inlining a
function, since inlining changes what counts as one expression.

Defaults differ by toolchain: GCC contracts aggressively (`-ffp-contract=fast`), Clang only
within a statement (`-ffp-contract=on`). GCC does not implement the standard's
`FP_CONTRACT` pragma; Clang does.

**An honest null result:** I tried to force this effect in a just-in-time compiled
framework by compiling one expression in two contexts. It did not reproduce — 0 of 20 000
values differed. You cannot summon it at will, which is exactly why it looks like a bug
when it does appear.

---

## So why does running the same program twice give the same answer?

Fifty runs of the same sum produced **one distinct result**. Bit-identical every time.

The resolution is the crux of the topic:

> **Each individual operation is completely deterministic. What varies is which sequence of
> operations your source code becomes.**

IEEE 754 requires addition, subtraction, multiplication, division, square root and fused
multiply-add to be *correctly rounded* — compute as if in infinite precision, then round.
The standard states that results are uniquely determined by the input values, the sequence
of operations, and the destination formats. All three are under your control.

That guarantee holds across Intel, AMD, ARM and RISC-V, and across C++, Fortran, Python,
Julia and R alike. When you get identical bits twice, a standard is making it happen.

But source code is not a sequence of operations — it is a *description* of one. The
compiler decides the actual instructions, and it has latitude. **One source, many valid
translations.** Each translation is perfectly deterministic; they are simply not all the
same translation.

So:

- **Same compiled program, same input** → identical, always, guaranteed.
- **Same source, compiled differently** → may differ in the last digit.

"Compiled differently" covers more than you would expect: a different optimisation level,
compiler version, processor, GPU instead of CPU, or maths library — or, in systems that
compile at runtime like JAX, PyTorch or Numba, *the same function used in a different
context within one program*. That last case is the sneaky one, because nothing visible to
you changed.

### Two caveats with teeth

**Transcendental functions are not covered.** `exp`, `log`, `sin`, `cos` are only
*recommended* to be correctly rounded, not required. Two maths libraries may return
different last digits for the same sine, so a library upgrade can move results even when
every operation you wrote was identical.

**Parallel reductions can vary run to run.** When the order of operations is decided at
runtime rather than compile time — GPU threads accumulating in whatever order they finish —
the order varies between runs, and since addition is not associative, so does the answer.
This is why some libraries offer a slower "deterministic mode" that forces a fixed order.

---

## Integers are exact — which is not the same as safe

Integers never round: 0 of 500 000 triples disagreed on grouping. For genuinely whole
quantities — counts, indices, identifiers — an integer type is exactly right.

Their failure mode is different, and worse in one respect: **silent, and enormous rather
than tiny.**

```
int8    127                       + 1  =  -128
int16   32 767                    + 1  =  -32 768
int32   2 147 483 647             + 1  =  -2 147 483 648
int64   9 223 372 036 854 775 807 + 1  =  -9 223 372 036 854 775 808

no error, no warning, no exception
```

A float that runs out of precision is wrong in the last digit. An integer that runs out of
range flips sign and magnitude entirely. The float degrades; the integer detonates.

---

## Practical rules

- **Never compare floats with `==`.** Use a tolerance suited to the magnitude.
- **Default to `float64` for anything analytical** — sums, statistics, accumulators,
  anything iterated. It is where all three experiments above came out clean.
- **`float32` is fine for measurements** whose real noise exceeds its precision. If a
  sensor is accurate to 1%, storing 7 digits stores 5 digits of noise.
- **`float16` is for bulk storage and network weights**, not arithmetic. Three digits;
  stops counting at 2 048; overflows at 65 504.
- **Integers for anything countable** — but check the range. `int32` overflows at 2.1
  billion, which real data reaches.
- **Sum with a good algorithm.** Pairwise beat the naive loop by ~200× here at no cost.
  NumPy already does this; a hand-written loop does not.
- **Do not demand bit-identical results across machines.** Compare against a tolerance.
  Bitwise reproducibility across hardware is a research problem, not something you can
  insist upon.

---

## The code

```python
import numpy as np
rng = np.random.default_rng(20260820)
N = 500_000

# How often does grouping change the answer?
for dt in (np.float64, np.float32, np.float16):
    A, B, C = ((rng.random(N) * 10).astype(dt) for _ in range(3))
    disagree = np.sum((A + B) + C != A + (B + C))
    print(dt.__name__, f"{100 * disagree / N:.1f}% disagree")

# Fused multiply-add vs multiply-then-add
mean, std = np.float32(0.7), np.float32(0.3)
noise = rng.standard_normal(N).astype(np.float32)
separate = (std * noise + mean).astype(np.float32)              # rounds twice
fused = (np.float64(std) * noise.astype(np.float64)
         + np.float64(mean)).astype(np.float32)                 # rounds once
print(f"{100 * np.sum(separate != fused) / N:.1f}% differ")

# Which is right? Exact rational arithmetic settles it.
from fractions import Fraction as F
a, b, c = np.float32(0.3), np.float32(-0.8005198), np.float32(0.7)
exact = F(float(a)) * F(float(b)) + F(float(c))
print("exact      ", float(exact))
print("round once ", np.float32(float(exact)))
print("round twice", np.float32(np.float32(a * b) + c))

# Summation order
x = rng.random(200_000).astype(np.float32)
ref = np.sum(x.astype(np.float64))
loop = np.float32(0)
for v in x:
    loop = np.float32(loop + v)
print("naive loop error", abs(loop - ref))
print("pairwise error  ", abs(np.add.reduce(x) - ref))

# Integer overflow is silent
print(np.int8(127) + np.int8(1))        # -128

# Where each float type stops counting whole numbers
for dt in (np.float64, np.float32, np.float16):
    limit = 2 ** (np.finfo(dt).nmant + 1)
    print(dt.__name__, limit, dt(limit) == dt(limit) + dt(1))
```

---

## References

- **David Goldberg**, "What Every Computer Scientist Should Know About Floating-Point
  Arithmetic", *ACM Computing Surveys* **23**(1), 5–48, 1991. Still the standard
  introduction.
- **IEEE 754**, Standard for Floating-Point Arithmetic (1985; current revision 2019). The
  source of the correct-rounding requirement, and therefore of run-to-run determinism.
- **Nicholas J. Higham**, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM,
  2002 — summation methods and error bounds, pp. 110–123. Also "The accuracy of floating
  point summation", *SIAM J. Sci. Comput.* **14**(4), 783–799, 1993.
- **ISO C** (C11 §6.5, `FP_CONTRACT`) for the contraction rule; GCC and Clang
  `-ffp-contract` documentation for the differing defaults.

---

*Measured 2026-08-20 on NumPy 2.3.5, Python 3.11.14, x86-64. Percentages come from 500 000
random samples per type; exact values shift slightly with the seed. "Exact" reference
values were computed with Python's `fractions.Fraction` (unlimited-precision rational
arithmetic), so they carry no rounding of their own.*
