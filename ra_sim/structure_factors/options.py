"""Options for raw structure-factor calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class StructureFactorOptions:
    scattering_table: str = "itc"
    anomalous_mode: str = "xraydb"
    debye_waller_mode: str = "cif"
    occupancy_mode: str = "cif"
    phase_sign: int = 1
    constant_factors: Mapping[str, complex] = field(default_factory=dict)

    @classmethod
    def package_default(cls) -> "StructureFactorOptions":
        return cls(
            scattering_table="itc",
            anomalous_mode="xraydb",
            debye_waller_mode="cif",
            occupancy_mode="cif",
        )

    @classmethod
    def vesta_cu_ka1(cls) -> "StructureFactorOptions":
        return cls(
            scattering_table="waaskirf",
            anomalous_mode="vesta_cu_ka1",
            debye_waller_mode="cif",
            occupancy_mode="cif",
        )

    def changed_from(self, other: "StructureFactorOptions") -> list[str]:
        names = (
            "scattering_table",
            "anomalous_mode",
            "debye_waller_mode",
            "occupancy_mode",
            "phase_sign",
            "constant_factors",
        )
        return [name for name in names if getattr(self, name) != getattr(other, name)]

    def to_dict(self) -> dict[str, object]:
        return {
            "scattering_table": self.scattering_table,
            "anomalous_mode": self.anomalous_mode,
            "debye_waller_mode": self.debye_waller_mode,
            "occupancy_mode": self.occupancy_mode,
            "phase_sign": self.phase_sign,
            "constant_factors": {
                key: [complex(value).real, complex(value).imag]
                for key, value in self.constant_factors.items()
            },
        }
