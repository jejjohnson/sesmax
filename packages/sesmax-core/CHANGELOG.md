# Changelog

## [0.4.0](https://github.com/jejjohnson/sesmax/compare/sesmax-core-v0.3.1...sesmax-core-v0.4.0) (2026-11-28)


### ⚠ BREAKING CHANGES

* **experiment:** the package, with the monitors and the `Component` type ([#197](https://github.com/jejjohnson/sesmax/issues/197))

### Features

* **core:** public validators, Burger-number helpers and located-field helpers ([#163](https://github.com/jejjohnson/sesmax/issues/163)) ([9825cea](https://github.com/jejjohnson/sesmax/commit/9825ceae887c7cf1f50b763f290fc854ee253941)), closes [#153](https://github.com/jejjohnson/sesmax/issues/153)
* **experiment:** run specs, the component registry and a command ([#203](https://github.com/jejjohnson/sesmax/issues/203)) ([f3fd415](https://github.com/jejjohnson/sesmax/commit/f3fd415731b3bf9819803778d6e11034ef03f782))
* **experiment:** the checks of a run, and a spec's model as an operator ([#208](https://github.com/jejjohnson/sesmax/issues/208)) ([6ee6f1c](https://github.com/jejjohnson/sesmax/commit/6ee6f1c197876b5992717f05b6d4bc7434755d82))
* **experiment:** the package, with the monitors and the `Component` type ([#197](https://github.com/jejjohnson/sesmax/issues/197)) ([2dd5b83](https://github.com/jejjohnson/sesmax/commit/2dd5b83166a37cdd24fb829d815bd9dfcffed068))
* **ocean:** sesmax-ocean, with the resolution checks, the metrics and the forcing presets ([#164](https://github.com/jejjohnson/sesmax/issues/164)) ([2bf6494](https://github.com/jejjohnson/sesmax/commit/2bf6494436e2daa6b23e16ee6e8f428a2bc190eb)), closes [#154](https://github.com/jejjohnson/sesmax/issues/154)
* **ocean:** the scenarios, the model registry and the configs, registered as a component ([#210](https://github.com/jejjohnson/sesmax/issues/210)) ([b44fad1](https://github.com/jejjohnson/sesmax/commit/b44fad1e4434736902ba43dc1f6d6aa3557dc9ad))


### Bug Fixes

* **deps:** pipekit at pipekit-array-v0.3.0, gridax v0.3.4 and geophysix v0.0.10 ([#242](https://github.com/jejjohnson/sesmax/issues/242)) ([4420115](https://github.com/jejjohnson/sesmax/commit/44201150bb439442c9edc092aa6bdbdefc5c62d6))
* **deps:** pipekit-array 0.1.6, and DiffraxForwardModel from integrix ([#181](https://github.com/jejjohnson/sesmax/issues/181)) ([5c998e6](https://github.com/jejjohnson/sesmax/commit/5c998e6f8a3188875822ee95a32aa540f5411c5f))
* **deps:** pipekit-array 0.2.0 ([#204](https://github.com/jejjohnson/sesmax/issues/204)) ([1d41f4b](https://github.com/jejjohnson/sesmax/commit/1d41f4bf3aea9a60a7b954de2ac37673ecfb5110))
* **model:** compile `Model.integrate` once across calls ([#196](https://github.com/jejjohnson/sesmax/issues/196)) ([dfbda8a](https://github.com/jejjohnson/sesmax/commit/dfbda8a82624fc1013ef3d68439d801b753dc4dc))
* **model:** the states a run returns have their boundary conditions ([#201](https://github.com/jejjohnson/sesmax/issues/201)) ([91de391](https://github.com/jejjohnson/sesmax/commit/91de391485df8a67648669134c16c9955ad1d459))

## [0.3.1](https://github.com/jejjohnson/sesmax/compare/sesmax-core-v0.3.0...sesmax-core-v0.3.1) (2026-11-25)


### Bug Fixes

* **deps:** declare gaussx, which the elliptic inversions import ([#150](https://github.com/jejjohnson/sesmax/issues/150)) ([566b3c7](https://github.com/jejjohnson/sesmax/commit/566b3c760086b00cdd6aca65ce1c98063b98563e))

## [0.3.0](https://github.com/jejjohnson/sesmax/compare/sesmax-core-v0.2.0...sesmax-core-v0.3.0) (2026-11-24)


### ⚠ BREAKING CHANGES

* **deps:** move to gridax 42fe52d, GridOps and the functions in place of the operator classes ([#146](https://github.com/jejjohnson/sesmax/issues/146))

### Miscellaneous

* **deps:** move to gridax 42fe52d, GridOps and the functions in place of the operator classes ([#146](https://github.com/jejjohnson/sesmax/issues/146)) ([629103f](https://github.com/jejjohnson/sesmax/commit/629103f373d41f413d67705737c0ba8d92bc754e))

## [0.2.0](https://github.com/jejjohnson/sesmax/compare/sesmax-core-v0.1.0...sesmax-core-v0.2.0) (2026-11-23)


### ⚠ BREAKING CHANGES

* **core:** states of located fields, on gridax's canonical functions ([#138](https://github.com/jejjohnson/sesmax/issues/138))
* **deps:** sesmax-core moves to gridax with the jax and pipekit extras, at 5de277f ([#135](https://github.com/jejjohnson/sesmax/issues/135))

### Features

* **core:** forcing, vertical modes, relaxation and the barotropic filter, moved from gridax-jax ([#122](https://github.com/jejjohnson/sesmax/issues/122)) ([e40d702](https://github.com/jejjohnson/sesmax/commit/e40d702f8758b3a69f28008cc5503ec695cb6560))
* **core:** sampled forcing takes any functax 1-D scheme, and reports its step times ([#141](https://github.com/jejjohnson/sesmax/issues/141)) ([3c64811](https://github.com/jejjohnson/sesmax/commit/3c64811ed8d7bdb4d4300f3101de9eecd48fd608)), closes [#130](https://github.com/jejjohnson/sesmax/issues/130)
* **core:** solve_pde integrates a right-hand-side function with state and tendency corrections at every stage ([#128](https://github.com/jejjohnson/sesmax/issues/128)) ([a13567e](https://github.com/jejjohnson/sesmax/commit/a13567ecf46248e2dd0b8ba98b9d23af8f20e2e1)), closes [#127](https://github.com/jejjohnson/sesmax/issues/127)
* **core:** states of located fields, on gridax's canonical functions ([#138](https://github.com/jejjohnson/sesmax/issues/138)) ([65fda09](https://github.com/jejjohnson/sesmax/commit/65fda09f998003f87023f27ee9cd52660789da9f))
* **core:** the streamfunction, pressure and QG PV inversions move to sesmax_core.gfd, with their tests and notebooks ([#129](https://github.com/jejjohnson/sesmax/issues/129)) ([995adc8](https://github.com/jejjohnson/sesmax/commit/995adc8c92fb85f479ae8d68cfefdf79676d7328))


### Miscellaneous

* **deps:** sesmax-core moves to gridax with the jax and pipekit extras, at 5de277f ([#135](https://github.com/jejjohnson/sesmax/issues/135)) ([071d73a](https://github.com/jejjohnson/sesmax/commit/071d73a530027975b6a36370949bc798c53f2248))

## 0.1.0 (2026-11-10)


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
