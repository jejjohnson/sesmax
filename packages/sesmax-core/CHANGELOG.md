# Changelog

## 0.1.0 (2026-11-05)


### Features

* **core:** domains, the IMEX solver helpers, scales, ScaledModel and StateAffine ([#40](https://github.com/jejjohnson/sesmax/issues/40)) ([ec68a5c](https://github.com/jejjohnson/sesmax/commit/ec68a5cbe5b560290fc46dab14f64bab25b131e0))
* **core:** forcing, spatial and temporal bases, and the generic forcing bank ([#43](https://github.com/jejjohnson/sesmax/issues/43)) ([a2258ea](https://github.com/jejjohnson/sesmax/commit/a2258ea8f5d2b21b1c022cf7fbe0970518662198))
* **core:** monitors, training rollouts and the ModelOp bridge ([#44](https://github.com/jejjohnson/sesmax/issues/44)) ([1543818](https://github.com/jejjohnson/sesmax/commit/15438185981176c89eab9d997df39f98824c5ef5))
* **core:** state and parameter types, the model base classes, guards and validators ([#37](https://github.com/jejjohnson/sesmax/issues/37)) ([102179b](https://github.com/jejjohnson/sesmax/commit/102179bcb08a1227042394fcf4c246c8b967e08b))
* **core:** the DA adapters behind [da] and the flowjax bridge behind [flows] ([#47](https://github.com/jejjohnson/sesmax/issues/47)) ([e17fb65](https://github.com/jejjohnson/sesmax/commit/e17fb653c357c2dc6974abe2e283ae768bde9651))
* **core:** the gfd module — stratification, modal transform, Helmholtz caches, Burger helpers ([#42](https://github.com/jejjohnson/sesmax/issues/42)) ([0b68f39](https://github.com/jejjohnson/sesmax/commit/0b68f3958c8ae86bd296d58756a32affde8051e9))
* **core:** the term algebra and OperatorTerm ([#39](https://github.com/jejjohnson/sesmax/issues/39)) ([d7810cb](https://github.com/jejjohnson/sesmax/commit/d7810cb93a6186a09102558cea64c0744d5f25ce))
* **core:** xarray I/O behind [xarray] and the checkpointer behind [checkpoint] ([#48](https://github.com/jejjohnson/sesmax/issues/48)) ([81d638b](https://github.com/jejjohnson/sesmax/commit/81d638bc5d3d441168a47c1c9b219a2c66203797))
* **da:** add WindowOperator, a control through a whole model run to the observations ([#97](https://github.com/jejjohnson/sesmax/issues/97)) ([314de57](https://github.com/jejjohnson/sesmax/commit/314de57f1acb1ecbbcf8989ef2c0f8ba9ac26ce7))
* **da:** let VardaxForwardModel forward an adjoint to the model ([#80](https://github.com/jejjohnson/sesmax/issues/80)) ([1859370](https://github.com/jejjohnson/sesmax/commit/185937010a091de52b67b4de2534e5ebcd4deb5c))
* **forcing:** add HatInTime, a piecewise-linear temporal basis ([#96](https://github.com/jejjohnson/sesmax/issues/96)) ([9f704e5](https://github.com/jejjohnson/sesmax/commit/9f704e5f8f2b05ff6f3ffd68285b9f921e6d330f))
* **model:** add Model.as_forward_model ([#94](https://github.com/jejjohnson/sesmax/issues/94)) ([dc0c54d](https://github.com/jejjohnson/sesmax/commit/dc0c54dbc6de899c769d6c97586744bd672cf36d))
* **solvers:** add stable_step, the summed advective and diffusive bound ([#93](https://github.com/jejjohnson/sesmax/issues/93)) ([518876b](https://github.com/jejjohnson/sesmax/commit/518876b9b114923732418995b49f9848973ab0b7))
* **testing:** periodic box and re-entrant channel for IncompressibleNS2D ([#63](https://github.com/jejjohnson/sesmax/issues/63)) ([128cbb2](https://github.com/jejjohnson/sesmax/commit/128cbb26db1bd7c89b3fa88026ab814a5ed90636))
* **testing:** the 1D PDE test problems ([#49](https://github.com/jejjohnson/sesmax/issues/49)) ([718f9f5](https://github.com/jejjohnson/sesmax/commit/718f9f552524e2ad64f3ac5c14a06e618bb7b90b))
* **testing:** the 1D shallow-water and 2D linear shallow-water models ([#51](https://github.com/jejjohnson/sesmax/issues/51)) ([b30c6a8](https://github.com/jejjohnson/sesmax/commit/b30c6a88c54dbde0eb46918ce6177dbb6a75b66e))
* **testing:** the 2D PDE test problems, the Burgers terms and Burgers2DOp ([#50](https://github.com/jejjohnson/sesmax/issues/50)) ([9c68871](https://github.com/jejjohnson/sesmax/commit/9c68871d948600e3ee521f5a0523f610abf2f1a3))
* **testing:** the Lorenz systems in sesmax_core.testing ([#46](https://github.com/jejjohnson/sesmax/issues/46)) ([9ede05c](https://github.com/jejjohnson/sesmax/commit/9ede05c01b72618cdd590eb41b4a483e9bbe6208))


### Bug Fixes

* **testing:** correct the nonlinear models' rate and their periodic edges ([#75](https://github.com/jejjohnson/sesmax/issues/75)) ([307bf9e](https://github.com/jejjohnson/sesmax/commit/307bf9e8e9597396493638b1192e939e66fb07fe))
* **testing:** put the cavity's wall vorticity on the wall nodes ([#58](https://github.com/jejjohnson/sesmax/issues/58)) ([7b9277d](https://github.com/jejjohnson/sesmax/commit/7b9277d7bb03110a8c53c8a1176e51c01a55ca87))

## Changelog
