# Damping model options

The simulator supports two damping models for the train structure:

## `stiffness` (default)

The damping matrix is proportional to stiffness only:

```
C = β K
```

This removes the mass-proportional term (α = 0), so rigid-body translation
modes (zero stiffness) are not damped. The coefficient β is chosen from the
target damping ratio ζ and a characteristic frequency:

```
β = 2 ζ / ω_n
```

By default, the lowest non-zero natural frequency is used. You can override it
by setting `damping_target` to a specific ωₙ in rad/s.

## `rayleigh_full`

The legacy model uses full Rayleigh damping:

```
C = α M + β K
```

This can damp rigid-body translation if α is large, so it is no longer the
default. Use it only when you explicitly want mass-proportional damping.

## Note on thesis vs. current model

The default damping model intentionally differs from the thesis-era Rayleigh
setup: the stiffness-proportional default avoids spurious deceleration of the
train before impact by not damping rigid-body translation modes. This keeps the
approach velocity physically meaningful when friction is absent.
