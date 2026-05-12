
## Contact state per mass patch

The dissipative wall-contact models now rely on a dedicated `ContactState` object
that tracks `active` and `v0_contact` per x-mass. This avoids the previous failure
mode where a later mass could inherit the default `v0_contact = 1 m/s`, inflating
`delta_dot / v0_contact` and creating a non-physical local force spike.

Engine convention:

```text
u_contact = q_x < 0 during wall penetration
du_contact = qdot_x
delta = -u_contact
delta_dot = -du_contact
```

Therefore `delta_dot > 0` during approach and `delta_dot < 0` during restitution.

Unknown contact model names now raise `ValueError` instead of silently falling
back to `anagnostopoulos`.
