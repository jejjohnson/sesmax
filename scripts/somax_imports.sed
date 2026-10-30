# Rewrite somax imports and names into sesmax-core (GNU sed).
#
# Used by the faithfulness check of every move from somax at 9cb19b3:
#   diff <(git -C ../somax show 9cb19b3:<src> | sed -f scripts/somax_imports.sed) <dest>
# prints only the changes a move may make. The renames come first so
# `SomaxModelOp` is not caught by `SomaxModel`; specific paths come before the
# generic `somax._src.` rule. `core/transforms.py` is split by hand (StateAffine
# to `_src/state_affine.py`, the modal pieces to `_src/gfd/modal.py`); the two
# transforms rules cover single-name imports, and a mixed import is edited by hand.
s/\bSomaxModelOp\b/ModelOp/g
s/\bSomaxForwardModel\b/VardaxForwardModel/g
s/\bSomaxDynamics\b/FilteraxDynamics/g
s/\bSomaxModel\b/Model/g
s/somax\._src\.core\.transforms import StateAffine/sesmax_core._src.state_affine import StateAffine/g
s/somax\._src\.core\.transforms import \(ModalTransform\|StratificationProfile\)/sesmax_core._src.gfd.modal import \1/g
s/somax\._src\.core\.helmholtz/sesmax_core._src.gfd.helmholtz/g
s/somax\._src\.core\./sesmax_core._src./g
s/somax\._src\.models\.lorenz/sesmax_core._src.testing.lorenz/g
s/somax\._src\.models\.pde/sesmax_core._src.testing.pde/g
s/somax\._src\.models\.swm\.\(linear_1d\|nonlinear_1d\|linear_2d\|linear_swm_terms\)/sesmax_core._src.testing.swm.\1/g
s/somax\._src\.models\.gfd_testcases/sesmax_core._src.testing.gfd_testcases/g
s/somax\._src\.models\._diagnostic_ops/sesmax_core._src._diagnostic_ops/g
s/somax\._src\.models\._nondim/sesmax_core._src._validation/g
s/somax\._src\./sesmax_core._src./g
s/\bsomax\.core\b/sesmax_core/g
s/\bsomax\.models\b/sesmax_core.testing/g
s/\bsomax\.\(da\|domain\|flows\|guards\|io\|monitor\|operators\|solvers\|train\)\b/sesmax_core.\1/g
s/^from somax import/from sesmax_core import/
