# Ocean: models on the sphere

The models on a latitude band of the sphere, moved from somax. Each reproduces somax's tendencies, end states and diagnostics in float64, to a relative 1e-12 for `SphericalSWM` and 1e-9 for `SphericalQG`, whose streamfunction is solved to a tolerance (`packages/sesmax-ocean/tests/spherical/test_equal_to_somax.py`).

A run differs from the same run in somax in three ways. somax's T rows were half a cell too far south of the band it was given; here they are its cell centres. The cell areas are exact, where somax's were `R² cos φ Δλ Δφ`. The default radius and rotation rate are geophysix's (6,371,008.8 m and 7.292115e-5 1/s); somax used 6,371,200 m and 2π/86,164 s.

Fixed since, in `SphericalSWM`:

- It conserves mass. somax took no flux through the edge faces, so the cells along the edge had a wrong thickness tendency.

::: sesmax_ocean.spherical
