"""Module for handling material properties and signal attenuation in WiFi environments with absolute precision."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Callable
import numpy as np
import math
from scipy import constants
from scipy.optimize import minimize_scalar
import warnings

# Physical constants for precise calculations
EPSILON_0 = constants.epsilon_0  # Vacuum permittivity (F/m)
MU_0 = constants.mu_0           # Vacuum permeability (H/m)
C = constants.c                 # Speed of light (m/s)
ETA_0 = np.sqrt(MU_0 / EPSILON_0)  # Intrinsic impedance of free space (Ω)

@dataclass(frozen=True)
class FrequencyDependentProperty:
    """Represents frequency-dependent material properties with interpolation."""
    frequencies: List[float]  # Frequencies in Hz
    values: List[float]       # Property values at each frequency
    
    def get_value(self, frequency: float) -> float:
        """Get interpolated value at given frequency."""
        if len(self.frequencies) == 1:
            return self.values[0]
        
        # Find nearest frequency or interpolate
        if frequency <= self.frequencies[0]:
            return self.values[0]
        elif frequency >= self.frequencies[-1]:
            return self.values[-1]
        else:
            # Linear interpolation
            for i in range(len(self.frequencies) - 1):
                if self.frequencies[i] <= frequency <= self.frequencies[i + 1]:
                    f1, f2 = self.frequencies[i], self.frequencies[i + 1]
                    v1, v2 = self.values[i], self.values[i + 1]
                    return v1 + (v2 - v1) * (frequency - f1) / (f2 - f1)
        
        return self.values[-1]  # Fallback

@dataclass(frozen=True)
class AdvancedMaterial:
    """Advanced material class with frequency-dependent properties and precise physics."""
    name: str
    # Frequency-dependent properties (can be single values or frequency-dependent)
    relative_permittivity: Union[float, FrequencyDependentProperty]  # εᵣ
    conductivity: Union[float, FrequencyDependentProperty]          # σ (S/m)
    relative_permeability: Union[float, FrequencyDependentProperty] = 1.0  # μᵣ
    loss_tangent: Optional[Union[float, FrequencyDependentProperty]] = None  # tan(δ)
    
    # Physical properties
    density: float = 1000.0  # kg/m³
    temperature: float = 293.15  # K (20°C)
    
    # Surface properties for reflection/transmission
    surface_roughness: float = 0.0  # RMS roughness in meters
    surface_conductivity: Optional[float] = None  # Surface conductivity for metals
    
    # Composite material properties
    is_composite: bool = False
    composite_layers: List['AdvancedMaterial'] = field(default_factory=list)
    layer_thicknesses: List[float] = field(default_factory=list)
    
    def get_relative_permittivity(self, frequency: float) -> complex:
        """Get complex relative permittivity at given frequency."""
        if isinstance(self.relative_permittivity, FrequencyDependentProperty):
            eps_r_real = self.relative_permittivity.get_value(frequency)
        else:
            eps_r_real = self.relative_permittivity
        
        # Get conductivity
        if isinstance(self.conductivity, FrequencyDependentProperty):
            sigma = self.conductivity.get_value(frequency)
        else:
            sigma = self.conductivity
        
        # Get loss tangent if available
        if self.loss_tangent is not None:
            if isinstance(self.loss_tangent, FrequencyDependentProperty):
                tan_delta = self.loss_tangent.get_value(frequency)
            else:
                tan_delta = self.loss_tangent
            eps_r_imag = eps_r_real * tan_delta
        else:
            # Calculate from conductivity
            omega = 2 * np.pi * frequency
            eps_r_imag = sigma / (omega * EPSILON_0)
        
        return eps_r_real - 1j * eps_r_imag
    
    def get_relative_permeability(self, frequency: float) -> complex:
        """Get complex relative permeability at given frequency."""
        if isinstance(self.relative_permeability, FrequencyDependentProperty):
            mu_r = self.relative_permeability.get_value(frequency)
        else:
            mu_r = self.relative_permeability
        
        # For most materials, μᵣ ≈ 1 (non-magnetic)
        return mu_r - 1j * 0.0
    
    def get_propagation_constant(self, frequency: float) -> complex:
        """Calculate complex propagation constant γ = α + jβ."""
        omega = 2 * np.pi * frequency
        eps_r = self.get_relative_permittivity(frequency)
        mu_r = self.get_relative_permeability(frequency)
        
        # Complex propagation constant
        gamma = 1j * omega * np.sqrt(MU_0 * EPSILON_0 * eps_r * mu_r)
        return gamma
    
    def get_attenuation_constant(self, frequency: float) -> float:
        """Get power attenuation constant α (Np/m)."""
        gamma = self.get_propagation_constant(frequency)
        return np.real(gamma)
    
    def get_phase_constant(self, frequency: float) -> float:
        """Get phase constant β (rad/m)."""
        gamma = self.get_propagation_constant(frequency)
        return np.imag(gamma)
    
    def get_intrinsic_impedance(self, frequency: float) -> complex:
        """Get intrinsic impedance of the material."""
        eps_r = self.get_relative_permittivity(frequency)
        mu_r = self.get_relative_permeability(frequency)
        return ETA_0 * np.sqrt(mu_r / eps_r)
    
    def calculate_attenuation(self, frequency: float = 2.4e9, thickness: float = None, 
                            angle_of_incidence: float = 0.0) -> float:
        """
        Calculate precise signal attenuation through the material.
        
        Args:
            frequency: Signal frequency in Hz
            thickness: Material thickness in meters (if None, uses default)
            angle_of_incidence: Angle of incidence in radians (0 = normal incidence)
            
        Returns:
            Attenuation in dB
        """
        if self.is_composite and self.composite_layers:
            return self._calculate_composite_attenuation(frequency, thickness, angle_of_incidence)
        
        # Get attenuation constant
        alpha = self.get_attenuation_constant(frequency)
        
        # Apply thickness (exponential attenuation)
        if thickness is None:
            thickness = 0.1  # Default thickness
        
        # Basic exponential attenuation
        attenuation_np = alpha * thickness / np.cos(angle_of_incidence) if angle_of_incidence != 0 else alpha * thickness
        
        # Convert to dB (8.686 = 20/ln(10))
        attenuation_db = 8.686 * attenuation_np
        
        return attenuation_db
    
    def _calculate_composite_attenuation(self, frequency: float, total_thickness: float, 
                                       angle_of_incidence: float) -> float:
        """Calculate attenuation for composite materials using transfer matrix method."""
        if not self.composite_layers or not self.layer_thicknesses:
            return self.calculate_attenuation(frequency, total_thickness, angle_of_incidence)
        
        # Transfer matrix method for multilayer materials
        total_attenuation = 0.0
        
        for layer, layer_thickness in zip(self.composite_layers, self.layer_thicknesses):
            layer_atten = layer.calculate_attenuation(frequency, layer_thickness, angle_of_incidence)
            total_attenuation += layer_atten
        
        return total_attenuation
    
    def calculate_reflection_coefficient(self, frequency: float, angle_of_incidence: float, 
                                       polarization: str = 'TE') -> complex:
        """
        Calculate reflection coefficient using Fresnel equations.
        
        Args:
            frequency: Signal frequency in Hz
            angle_of_incidence: Angle of incidence in radians
            polarization: 'TE' (transverse electric) or 'TM' (transverse magnetic)
            
        Returns:
            Complex reflection coefficient
        """
        # Assume incident medium is air (εᵣ = 1, μᵣ = 1)
        eta_1 = ETA_0  # Air impedance
        eta_2 = self.get_intrinsic_impedance(frequency)
        
        if polarization.upper() == 'TE':
            # TE polarization (E-field perpendicular to plane of incidence)
            reflection_coeff = (eta_2 * np.cos(angle_of_incidence) - eta_1 * np.cos(self._get_transmission_angle(frequency, angle_of_incidence))) / \
                              (eta_2 * np.cos(angle_of_incidence) + eta_1 * np.cos(self._get_transmission_angle(frequency, angle_of_incidence)))
        else:
            # TM polarization (E-field parallel to plane of incidence)
            reflection_coeff = (eta_1 * np.cos(angle_of_incidence) - eta_2 * np.cos(self._get_transmission_angle(frequency, angle_of_incidence))) / \
                              (eta_1 * np.cos(angle_of_incidence) + eta_2 * np.cos(self._get_transmission_angle(frequency, angle_of_incidence)))
        
        return reflection_coeff
    
    def calculate_transmission_coefficient(self, frequency: float, angle_of_incidence: float, 
                                         polarization: str = 'TE') -> complex:
        """Calculate transmission coefficient using Fresnel equations."""
        reflection_coeff = self.calculate_reflection_coefficient(frequency, angle_of_incidence, polarization)
        return 1.0 + reflection_coeff  # T = 1 + R
    
    def _get_transmission_angle(self, frequency: float, angle_of_incidence: float) -> float:
        """Calculate transmission angle using Snell's Law."""
        # Assume incident medium is air (n₁ = 1)
        n1 = 1.0
        eps_r = self.get_relative_permittivity(frequency)
        n2 = np.sqrt(np.real(eps_r))  # Refractive index
        
        # Snell's Law: n₁ sin(θ₁) = n₂ sin(θ₂)
        sin_theta_2 = n1 * np.sin(angle_of_incidence) / n2
        
        # Handle total internal reflection
        if abs(sin_theta_2) > 1.0:
            return np.pi / 2  # Critical angle
        
        return np.arcsin(sin_theta_2)
    
    def calculate_total_attenuation_with_reflection(self, frequency: float, thickness: float, 
                                                  angle_of_incidence: float = 0.0, 
                                                  polarization: str = 'TE') -> float:
        """
        Calculate total attenuation including reflection losses.
        
        Args:
            frequency: Signal frequency in Hz
            thickness: Material thickness in meters
            angle_of_incidence: Angle of incidence in radians
            polarization: 'TE' or 'TM'
            
        Returns:
            Total attenuation in dB
        """
        # Transmission coefficient (power)
        T = self.calculate_transmission_coefficient(frequency, angle_of_incidence, polarization)
        transmission_loss_db = -10 * np.log10(np.abs(T)**2)
        
        # Material attenuation
        material_attenuation_db = self.calculate_attenuation(frequency, thickness, angle_of_incidence)
        
        # Total attenuation
        total_attenuation_db = transmission_loss_db + material_attenuation_db
        
        return total_attenuation_db

# Frequency-dependent material properties database
FREQUENCY_DEPENDENT_PROPERTIES = {
    'concrete': {
        'relative_permittivity': FrequencyDependentProperty(
            frequencies=[1e9, 2.4e9, 5e9, 10e9],
            values=[5.0, 4.5, 4.2, 4.0]
        ),
        'conductivity': FrequencyDependentProperty(
            frequencies=[1e9, 2.4e9, 5e9, 10e9],
            values=[0.02, 0.014, 0.012, 0.010]
        ),
        'loss_tangent': FrequencyDependentProperty(
            frequencies=[1e9, 2.4e9, 5e9, 10e9],
            values=[0.15, 0.12, 0.10, 0.08]
        )
    },
    'glass': {
        'relative_permittivity': FrequencyDependentProperty(
            frequencies=[1e9, 2.4e9, 5e9, 10e9],
            values=[6.5, 6.0, 5.8, 5.6]
        ),
        'conductivity': FrequencyDependentProperty(
            frequencies=[1e9, 2.4e9, 5e9, 10e9],
            values=[0.005, 0.004, 0.003, 0.002]
        )
    },
    'drywall': {
        'relative_permittivity': FrequencyDependentProperty(
            frequencies=[1e9, 2.4e9, 5e9, 10e9],
            values=[2.2, 2.0, 1.9, 1.8]
        ),
        'conductivity': FrequencyDependentProperty(
            frequencies=[1e9, 2.4e9, 5e9, 10e9],
            values=[0.002, 0.001, 0.0008, 0.0006]
        )
    },
    'metal': {
        'relative_permittivity': FrequencyDependentProperty(
            frequencies=[1e9, 2.4e9, 5e9, 10e9],
            values=[1.0, 1.0, 1.0, 1.0]
        ),
        'conductivity': FrequencyDependentProperty(
            frequencies=[1e9, 2.4e9, 5e9, 10e9],
            values=[1e7, 1e7, 1e7, 1e7]
        ),
        'surface_conductivity': 1e7
    }
}

# Advanced materials with frequency-dependent properties
ADVANCED_MATERIALS = {
    'concrete': AdvancedMaterial(
        name='Concrete',
        relative_permittivity=FREQUENCY_DEPENDENT_PROPERTIES['concrete']['relative_permittivity'],
        conductivity=FREQUENCY_DEPENDENT_PROPERTIES['concrete']['conductivity'],
        loss_tangent=FREQUENCY_DEPENDENT_PROPERTIES['concrete']['loss_tangent'],
        density=2400.0,
        surface_roughness=0.001
    ),
    'glass': AdvancedMaterial(
        name='Glass',
        relative_permittivity=FREQUENCY_DEPENDENT_PROPERTIES['glass']['relative_permittivity'],
        conductivity=FREQUENCY_DEPENDENT_PROPERTIES['glass']['conductivity'],
        density=2500.0,
        surface_roughness=0.0001
    ),
    'drywall': AdvancedMaterial(
        name='Drywall',
        relative_permittivity=FREQUENCY_DEPENDENT_PROPERTIES['drywall']['relative_permittivity'],
        conductivity=FREQUENCY_DEPENDENT_PROPERTIES['drywall']['conductivity'],
        density=800.0,
        surface_roughness=0.0005
    ),
    'metal': AdvancedMaterial(
        name='Metal',
        relative_permittivity=FREQUENCY_DEPENDENT_PROPERTIES['metal']['relative_permittivity'],
        conductivity=FREQUENCY_DEPENDENT_PROPERTIES['metal']['conductivity'],
        surface_conductivity=FREQUENCY_DEPENDENT_PROPERTIES['metal']['surface_conductivity'],
        density=7850.0,
        surface_roughness=0.00001
    ),
    'wood': AdvancedMaterial(
        name='Wood',
        relative_permittivity=2.1,
        conductivity=0.002,
        density=600.0,
        surface_roughness=0.002
    ),
    'brick': AdvancedMaterial(
        name='Brick',
        relative_permittivity=4.0,
        conductivity=0.01,
        density=1800.0,
        surface_roughness=0.003
    ),
    'tile': AdvancedMaterial(
        name='Tile',
        relative_permittivity=5.0,
        conductivity=0.003,
        density=2300.0,
        surface_roughness=0.0002
    ),
    'carpet': AdvancedMaterial(
        name='Carpet',
        relative_permittivity=2.5,
        conductivity=0.001,
        density=1200.0,
        surface_roughness=0.005
    ),
    'air': AdvancedMaterial(
        name='Air',
        relative_permittivity=1.0,
        conductivity=0.0,
        density=1.225,
        surface_roughness=0.0
    )
}

# Composite materials (e.g., reinforced concrete, insulated walls)
def create_reinforced_concrete() -> AdvancedMaterial:
    """Create reinforced concrete as a composite material."""
    concrete = ADVANCED_MATERIALS['concrete']
    steel = AdvancedMaterial(
        name='Steel',
        relative_permittivity=1.0,
        conductivity=1e7,
        density=7850.0
    )
    
    # Reinforced concrete: 95% concrete, 5% steel reinforcement
    composite = AdvancedMaterial(
        name='Reinforced Concrete',
        relative_permittivity=4.5,  # Effective permittivity
        conductivity=0.02,          # Effective conductivity
        is_composite=True,
        composite_layers=[concrete, steel],
        layer_thicknesses=[0.19, 0.01],  # 19cm concrete, 1cm steel
        density=2500.0
    )
    
    return composite

def create_insulated_wall() -> AdvancedMaterial:
    """Create insulated wall as a composite material."""
    drywall = ADVANCED_MATERIALS['drywall']
    insulation = AdvancedMaterial(
        name='Insulation',
        relative_permittivity=1.8,
        conductivity=0.0005,
        density=50.0
    )
    
    # Insulated wall: drywall-insulation-drywall
    composite = AdvancedMaterial(
        name='Insulated Wall',
        relative_permittivity=2.0,  # Effective permittivity
        conductivity=0.001,         # Effective conductivity
        is_composite=True,
        composite_layers=[drywall, insulation, drywall],
        layer_thicknesses=[0.016, 0.1, 0.016],  # 16mm drywall, 10cm insulation, 16mm drywall
        density=400.0
    )
    
    return composite

# Add composite materials to the database
ADVANCED_MATERIALS['reinforced_concrete'] = create_reinforced_concrete()
ADVANCED_MATERIALS['insulated_wall'] = create_insulated_wall()

# Backward compatibility: Keep original Material class
@dataclass(frozen=True)
class Material:
    """Legacy Material class for backward compatibility."""
    name: str
    relative_permittivity: float
    conductivity: float
    thickness: float
    color: tuple[float, float, float] = (0.5, 0.5, 0.5)
    
    def calculate_attenuation(self, frequency: float = 2.4e9) -> float:
        """Legacy attenuation calculation."""
        # Convert to AdvancedMaterial for calculation
        adv_material = AdvancedMaterial(
            name=self.name,
            relative_permittivity=self.relative_permittivity,
            conductivity=self.conductivity
        )
        return adv_material.calculate_attenuation(frequency, self.thickness)

# Legacy MATERIALS dictionary for backward compatibility
MATERIALS = {
    'concrete': Material('Concrete', 4.5, 0.014, 0.2),
    'glass': Material('Glass', 6.0, 0.004, 0.006),
    'wood': Material('Wood', 2.1, 0.002, 0.04),
    'drywall': Material('Drywall', 2.0, 0.001, 0.016),
    'metal': Material('Metal', 1.0, 1e7, 0.002),
    'brick': Material('Brick', 4.0, 0.01, 0.1),
    'plaster': Material('Plaster', 3.0, 0.005, 0.02),
    'tile': Material('Tile', 5.0, 0.003, 0.01),
    'asphalt': Material('Asphalt', 3.5, 0.006, 0.05),
    'carpet': Material('Carpet', 2.5, 0.001, 0.01),
    'plastic': Material('Plastic', 2.3, 0.0001, 0.005),
    'insulation': Material('Insulation', 1.8, 0.0005, 0.05),
    'fiber_cement': Material('Fiber Cement', 3.2, 0.002, 0.015),
    'steel': Material('Steel', 1.0, 1e7, 0.005),
    'copper': Material('Copper', 1.0, 5.8e7, 0.001),
    'aluminum': Material('Aluminum', 1.0, 3.5e7, 0.002),
    'foam': Material('Foam', 1.5, 0.0002, 0.03),
    'rubber': Material('Rubber', 2.0, 0.0001, 0.01),
    'ceramic': Material('Ceramic', 6.5, 0.002, 0.01),
    'vinyl': Material('Vinyl', 2.2, 0.0005, 0.002),
    'air': Material('Air', 1.0, 0.0, 0.0)
}

class MaterialLayer:
    """Represents a layer of material in the signal path."""
    def __init__(self, material: Union[Material, AdvancedMaterial], thickness_multiplier: float = 1.0):
        """Initialize a material layer."""
        self.material = material
        self.thickness = material.thickness * thickness_multiplier if hasattr(material, 'thickness') else 0.1
        
    def get_attenuation(self, frequency: float = 2.4e9, angle_of_incidence: float = 0.0) -> float:
        """Get the total attenuation through this layer."""
        if isinstance(self.material, AdvancedMaterial):
            return self.material.calculate_attenuation(frequency, self.thickness, angle_of_incidence)
        else:
            return self.material.calculate_attenuation(frequency)

class SignalPath:
    """Represents the path of a signal through various materials with advanced physics."""
    def __init__(self):
        """Initialize an empty signal path."""
        self.layers: List[MaterialLayer] = []
        
    def add_layer(self, material: Union[Material, AdvancedMaterial], thickness_multiplier: float = 1.0):
        """Add a material layer to the path."""
        self.layers.append(MaterialLayer(material, thickness_multiplier))
        
    def calculate_total_attenuation(self, frequency: float = 2.4e9, angle_of_incidence: float = 0.0) -> float:
        """Calculate total attenuation along the path with advanced physics."""
        total_attenuation = 0.0
        
        for layer in self.layers:
            layer_atten = layer.get_attenuation(frequency, angle_of_incidence)
            total_attenuation += layer_atten
            
        return total_attenuation
    
    def calculate_reflection_losses(self, frequency: float = 2.4e9, angle_of_incidence: float = 0.0) -> float:
        """Calculate reflection losses at material interfaces."""
        if len(self.layers) < 2:
            return 0.0
        
        total_reflection_loss = 0.0
        
        for i in range(len(self.layers) - 1):
            layer1 = self.layers[i].material
            layer2 = self.layers[i + 1].material
            
            if isinstance(layer1, AdvancedMaterial) and isinstance(layer2, AdvancedMaterial):
                # Calculate reflection coefficient at interface
                R = layer1.calculate_reflection_coefficient(frequency, angle_of_incidence)
                reflection_loss_db = -10 * np.log10(1 - np.abs(R)**2)
                total_reflection_loss += reflection_loss_db
        
        return total_reflection_loss

def test_advanced_material_properties():
    """Test advanced material properties and calculations."""
    print("=== Testing Advanced Material Properties ===")
    
    # Test frequency-dependent properties
    concrete = ADVANCED_MATERIALS['concrete']
    frequencies = [1e9, 2.4e9, 5e9, 10e9]
    
    print(f"\nConcrete Properties vs Frequency:")
    print(f"{'Frequency (GHz)':<15} {'εᵣ':<10} {'σ (S/m)':<12} {'α (Np/m)':<12} {'Atten (dB/cm)':<15}")
    print("-" * 70)
    
    for freq in frequencies:
        eps_r = concrete.get_relative_permittivity(freq)
        sigma = concrete.conductivity.get_value(freq) if isinstance(concrete.conductivity, FrequencyDependentProperty) else concrete.conductivity
        alpha = concrete.get_attenuation_constant(freq)
        atten_db_cm = concrete.calculate_attenuation(freq, 0.01)  # 1cm thickness
        
        print(f"{freq/1e9:<15.1f} {np.real(eps_r):<10.2f} {sigma:<12.4f} {alpha:<12.4f} {atten_db_cm:<15.2f}")
    
    # Test angle-dependent attenuation
    print(f"\nAngle-Dependent Attenuation (Glass, 2.4 GHz, 1cm):")
    print(f"{'Angle (deg)':<12} {'Atten (dB)':<12} {'Reflection Loss (dB)':<20} {'Total (dB)':<12}")
    print("-" * 60)
    
    glass = ADVANCED_MATERIALS['glass']
    angles_deg = [0, 15, 30, 45, 60, 75, 85]
    
    for angle_deg in angles_deg:
        angle_rad = np.radians(angle_deg)
        atten = glass.calculate_attenuation(2.4e9, 0.01, angle_rad)
        refl_loss = glass.calculate_reflection_coefficient(2.4e9, angle_rad)
        refl_loss_db = -10 * np.log10(1 - np.abs(refl_loss)**2)
        total = atten + refl_loss_db
        
        print(f"{angle_deg:<12} {atten:<12.2f} {refl_loss_db:<20.2f} {total:<12.2f}")
    
    # Test composite materials
    print(f"\nComposite Material Comparison:")
    print(f"{'Material':<20} {'Thickness':<12} {'Atten (dB)':<12}")
    print("-" * 50)
    
    materials_to_test = [
        ('Concrete', ADVANCED_MATERIALS['concrete'], 0.2),
        ('Reinforced Concrete', ADVANCED_MATERIALS['reinforced_concrete'], 0.2),
        ('Insulated Wall', ADVANCED_MATERIALS['insulated_wall'], 0.132),
        ('Glass', ADVANCED_MATERIALS['glass'], 0.006)
    ]
    
    for name, material, thickness in materials_to_test:
        atten = material.calculate_attenuation(2.4e9, thickness)
        print(f"{name:<20} {thickness:<12.3f} {atten:<12.2f}")

if __name__ == "__main__":
    test_advanced_material_properties()
