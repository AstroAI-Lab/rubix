from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
from beartype.typing import Any

from .api import ParamsTree

TransformTree = Mapping[str, Mapping[str, "ParameterTransform"]]


class ParameterTransform:
    """Base class for constrained parameter transforms."""

    def forward(self, unconstrained: Any) -> Any:
        """Map unconstrained values to constrained space."""
        raise NotImplementedError

    def inverse(self, constrained: Any) -> Any:
        """Map constrained values to unconstrained space."""
        raise NotImplementedError


@dataclass(frozen=True)
class IdentityTransform(ParameterTransform):
    """No-op transform."""

    def forward(self, unconstrained: Any) -> Any:
        return unconstrained

    def inverse(self, constrained: Any) -> Any:
        return constrained


@dataclass(frozen=True)
class SoftplusLowerBound(ParameterTransform):
    """Transform values to ``(lower, inf)`` via softplus."""

    lower: float
    eps: float = 1e-8

    def forward(self, unconstrained: Any) -> Any:
        return self.lower + jax.nn.softplus(unconstrained) + self.eps

    def inverse(self, constrained: Any) -> Any:
        shifted = constrained - self.lower - self.eps
        safe = jnp.maximum(shifted, jnp.asarray(self.eps, dtype=shifted.dtype))
        overflow_threshold = jnp.log(
            jnp.asarray(jnp.finfo(safe.dtype).max, dtype=safe.dtype)
        )
        return jnp.where(safe > overflow_threshold, safe, jnp.log(jnp.expm1(safe)))


@dataclass(frozen=True)
class SigmoidBounds(ParameterTransform):
    """Transform values to ``(lower, upper)`` via sigmoid."""

    lower: float
    upper: float
    eps: float = 1e-8

    def __post_init__(self):
        if self.upper <= self.lower:
            raise ValueError("upper must be strictly larger than lower")

    def forward(self, unconstrained: Any) -> Any:
        width = self.upper - self.lower
        return self.lower + width * jax.nn.sigmoid(unconstrained)

    def inverse(self, constrained: Any) -> Any:
        width = self.upper - self.lower
        x = (constrained - self.lower) / width
        x = jnp.clip(x, self.eps, 1.0 - self.eps)
        return jnp.log(x) - jnp.log1p(-x)


@dataclass(frozen=True)
class VelocityZBoundsTransform(ParameterTransform):
    """Transform that bounds only the line-of-sight velocity component.

    ``x`` and ``y`` velocity components are held fixed to ``fixed_xy`` in the
    forward pass. The unconstrained ``z`` channel is mapped to
    ``(lower_z, upper_z)`` via sigmoid.
    """

    lower_z: float
    upper_z: float
    fixed_xy: Any
    eps: float = 1e-8

    def __post_init__(self):
        if self.upper_z <= self.lower_z:
            raise ValueError("upper_z must be strictly larger than lower_z")

    def forward(self, unconstrained: Any) -> Any:
        unconstrained = jnp.asarray(unconstrained)
        fixed_xy = jnp.asarray(self.fixed_xy, dtype=unconstrained.dtype)
        width = self.upper_z - self.lower_z
        z = self.lower_z + width * jax.nn.sigmoid(unconstrained[:, 2])
        return jnp.stack([fixed_xy[:, 0], fixed_xy[:, 1], z], axis=1)

    def inverse(self, constrained: Any) -> Any:
        constrained = jnp.asarray(constrained)
        width = self.upper_z - self.lower_z
        z_scaled = (constrained[:, 2] - self.lower_z) / width
        z_scaled = jnp.clip(z_scaled, self.eps, 1.0 - self.eps)
        z_unconstrained = jnp.log(z_scaled) - jnp.log1p(-z_scaled)
        zeros_xy = jnp.zeros_like(constrained[:, :2])
        return jnp.concatenate([zeros_xy, z_unconstrained[:, None]], axis=1)


def apply_transforms(
    params: ParamsTree,
    transforms: TransformTree,
    direction: str = "forward",
) -> dict[str, dict[str, Any]]:
    """Apply transform tree to a parameter tree.

    Args:
        params (ParamsTree): Nested parameter dictionary.
        transforms (TransformTree): Nested transform dictionary with the same
            keys as ``params`` for transformed leaves.
        direction (str, optional): Either ``"forward"`` (unconstrained ->
            constrained) or ``"inverse"`` (constrained -> unconstrained).
            Defaults to ``"forward"``.

    Raises:
        ValueError: If ``direction`` is invalid.

    Returns:
        dict[str, dict[str, Any]]: Transformed parameter dictionary.
    """
    if direction not in {"forward", "inverse"}:
        raise ValueError("direction must be one of {'forward', 'inverse'}")

    transformed: dict[str, dict[str, Any]] = {}
    for component, fields in params.items():
        transformed[component] = {}
        component_transforms = transforms.get(component, {})
        for field, value in fields.items():
            transform = component_transforms.get(field, IdentityTransform())
            if direction == "forward":
                transformed[component][field] = transform.forward(value)
            else:
                transformed[component][field] = transform.inverse(value)

    return transformed


def inverse_transforms(
    params: ParamsTree,
    transforms: TransformTree,
) -> dict[str, dict[str, Any]]:
    """Apply inverse transforms to a parameter tree."""
    return apply_transforms(params=params, transforms=transforms, direction="inverse")


def build_age_metallicity_transforms(
    age_lower: float = 0.0,
    age_upper: float = 20.0,
    metallicity_lower: float = 0.0,
    metallicity_upper: float = 0.05,
) -> dict[str, dict[str, ParameterTransform]]:
    """Build default transform tree for age and metallicity optimization.

    Args:
        age_lower (float, optional): Lower age bound in Gyr. Defaults to 0.0.
        age_upper (float, optional): Upper age bound in Gyr. Defaults to 20.0.
        metallicity_lower (float, optional): Lower metallicity bound.
            Defaults to 0.0.
        metallicity_upper (float, optional): Upper metallicity bound.
            Defaults to 0.05.

    Returns:
        dict[str, dict[str, ParameterTransform]]: Transform tree for
            ``stars.age`` and ``stars.metallicity``.
    """
    return {
        "stars": {
            "age": SigmoidBounds(lower=age_lower, upper=age_upper),
            "metallicity": SigmoidBounds(
                lower=metallicity_lower,
                upper=metallicity_upper,
            ),
        }
    }


def build_age_metallicity_velocity_transforms(
    *,
    fixed_velocity_xy: Any,
    age_lower: float = 0.0,
    age_upper: float = 20.0,
    metallicity_lower: float = 0.0,
    metallicity_upper: float = 0.05,
    vz_lower: float = -400.0,
    vz_upper: float = 400.0,
) -> dict[str, dict[str, ParameterTransform]]:
    """Build transforms for age, metallicity, and velocity with bounded ``v_z``."""
    transforms = build_age_metallicity_transforms(
        age_lower=age_lower,
        age_upper=age_upper,
        metallicity_lower=metallicity_lower,
        metallicity_upper=metallicity_upper,
    )
    transforms["stars"]["velocity"] = VelocityZBoundsTransform(
        lower_z=vz_lower,
        upper_z=vz_upper,
        fixed_xy=fixed_velocity_xy,
    )
    return transforms
