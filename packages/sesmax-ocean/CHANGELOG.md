# Changelog

## 0.1.0 (2026-11-28)


### Features

* **experiment:** the checks of a run, and a spec's model as an operator ([#208](https://github.com/jejjohnson/sesmax/issues/208)) ([6ee6f1c](https://github.com/jejjohnson/sesmax/commit/6ee6f1c197876b5992717f05b6d4bc7434755d82))
* **ocean:** sesmax-ocean, with the resolution checks, the metrics and the forcing presets ([#164](https://github.com/jejjohnson/sesmax/issues/164)) ([2bf6494](https://github.com/jejjohnson/sesmax/commit/2bf6494436e2daa6b23e16ee6e8f428a2bc190eb)), closes [#154](https://github.com/jejjohnson/sesmax/issues/154)
* **ocean:** the barotropic and baroclinic QG models ([#165](https://github.com/jejjohnson/sesmax/issues/165)) ([5d99797](https://github.com/jejjohnson/sesmax/commit/5d997977839b53838fc22ee4465346a3647bc00b)), closes [#155](https://github.com/jejjohnson/sesmax/issues/155)
* **ocean:** the bundles of the basins, and the commands that fetch and build them ([#211](https://github.com/jejjohnson/sesmax/issues/211)) ([3f133df](https://github.com/jejjohnson/sesmax/commit/3f133dfeb2af5b8887a2dbfa704498cb6ec80d82))
* **ocean:** the scenarios, the model registry and the configs, registered as a component ([#210](https://github.com/jejjohnson/sesmax/issues/210)) ([b44fad1](https://github.com/jejjohnson/sesmax/commit/b44fad1e4434736902ba43dc1f6d6aa3557dc9ad))
* **ocean:** the shallow-water models and reparameterized QG ([#168](https://github.com/jejjohnson/sesmax/issues/168)) ([d34edd1](https://github.com/jejjohnson/sesmax/commit/d34edd10911c208f3c03cbb85fc333890f59d139))
* **ocean:** the spherical shallow-water and QG models ([#170](https://github.com/jejjohnson/sesmax/issues/170)) ([2e5ad7f](https://github.com/jejjohnson/sesmax/commit/2e5ad7f99603aff90cca85fc6e5cf576c4362985))


### Bug Fixes

* **deps:** pipekit at pipekit-array-v0.3.0, gridax v0.3.4 and geophysix v0.0.10 ([#242](https://github.com/jejjohnson/sesmax/issues/242)) ([4420115](https://github.com/jejjohnson/sesmax/commit/44201150bb439442c9edc092aa6bdbdefc5c62d6))
* **deps:** pipekit-array 0.1.6, and DiffraxForwardModel from integrix ([#181](https://github.com/jejjohnson/sesmax/issues/181)) ([5c998e6](https://github.com/jejjohnson/sesmax/commit/5c998e6f8a3188875822ee95a32aa540f5411c5f))
* **deps:** pipekit-array 0.2.0 ([#204](https://github.com/jejjohnson/sesmax/issues/204)) ([1d41f4b](https://github.com/jejjohnson/sesmax/commit/1d41f4bf3aea9a60a7b954de2ac37673ecfb5110))
* **model:** the states a run returns have their boundary conditions ([#201](https://github.com/jejjohnson/sesmax/issues/201)) ([91de391](https://github.com/jejjohnson/sesmax/commit/91de391485df8a67648669134c16c9955ad1d459))
* **ocean:** free slip on walls and coasts ([#182](https://github.com/jejjohnson/sesmax/issues/182)) ([9e353cf](https://github.com/jejjohnson/sesmax/commit/9e353cfe7a5460d9163c8b2859499042d7b2b621))
* **ocean:** hold q at zero on all four walls of the quasi-geostrophic models ([#213](https://github.com/jejjohnson/sesmax/issues/213)) ([a34e46c](https://github.com/jejjohnson/sesmax/commit/a34e46cbe597164ad37109472428366483c21e70))
* **ocean:** no tendency on the walls of the shallow-water models ([#184](https://github.com/jejjohnson/sesmax/issues/184)) ([b2b3259](https://github.com/jejjohnson/sesmax/commit/b2b32599bfc66ec9389dc0ecc1b29ad92c7353c6))
* **ocean:** the Coriolis parameter of `SphericalSWM` is at T ([#176](https://github.com/jejjohnson/sesmax/issues/176)) ([a677479](https://github.com/jejjohnson/sesmax/commit/a677479ae84043c06f979a4b43d2125b8a17eaa3))
* **ocean:** the momentum tendency at a periodic edge ([#174](https://github.com/jejjohnson/sesmax/issues/174)) ([f86087d](https://github.com/jejjohnson/sesmax/commit/f86087d8dab433f2ec1e5da219a6371032aabee8))
* **ocean:** the shallow-water models conserve mass ([#171](https://github.com/jejjohnson/sesmax/issues/171)) ([3ad89ac](https://github.com/jejjohnson/sesmax/commit/3ad89ac7064c87bc6d5e3d4267072aca5e543aa6))
* **ocean:** the velocity and the fluxes of `SphericalQG` at the edges of its band ([#177](https://github.com/jejjohnson/sesmax/issues/177)) ([aefe52f](https://github.com/jejjohnson/sesmax/commit/aefe52f7bb8f3ce332680e7b8879d4d5906f19d7))
* **ocean:** the viscous term along the edge of the box ([#175](https://github.com/jejjohnson/sesmax/issues/175)) ([b7e6ecc](https://github.com/jejjohnson/sesmax/commit/b7e6ecc735ccd766598d5bec2e9064c2caff7c20))
