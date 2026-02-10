"""Custom exceptions for python_magnetcooling"""


class MagnetCoolingError(Exception):
    """Base exception for python_magnetcooling"""
    pass


class WaterPropertiesError(MagnetCoolingError):
    """Error in water properties calculation"""
    pass


class CorrelationError(MagnetCoolingError):
    """Error in heat transfer correlation"""
    pass


class FrictionError(MagnetCoolingError):
    """Error in friction factor calculation"""
    pass


class ValidationError(MagnetCoolingError):
    """Input validation error"""
    pass


class ConvergenceError(MagnetCoolingError):
    """Iterative solver did not converge"""
    pass


class HeatExchangerError(MagnetCoolingError):
    """Base exception for heat exchanger calculations"""
    pass


class InvalidNTUError(HeatExchangerError):
    """Invalid NTU value in heat exchanger calculation"""
    
    def __init__(self, ntu_value: float, tci: float, thi: float, 
                 pci: float, phi: float, debitc: float, debith: float):
        self.ntu_value = ntu_value
        self.tci = tci
        self.thi = thi
        self.pci = pci
        self.phi = phi
        self.debitc = debitc
        self.debith = debith
        super().__init__(
            f"Invalid NTU value: {ntu_value}\n"
            f"Conditions: Tci={tci}°C, Thi={thi}°C, Pci={pci} bar, Phi={phi} bar, "
            f"Debitc={debitc} m³/h, Debith={debith} l/s"
        )


class InvalidHeatTransferError(HeatExchangerError):
    """Invalid heat transfer rate in heat exchanger calculation"""
    
    def __init__(self, q_value: float, tci: float, thi: float,
                 pci: float, phi: float, debitc: float, debith: float):
        self.q_value = q_value
        self.tci = tci
        self.thi = thi
        self.pci = pci
        self.phi = phi
        self.debitc = debitc
        self.debith = debith
        super().__init__(
            f"Invalid heat transfer rate: {q_value} W\n"
            f"Conditions: Tci={tci}°C, Thi={thi}°C, Pci={pci} bar, Phi={phi} bar, "
            f"Debitc={debitc} m³/h, Debith={debith} l/s"
        )


class InvalidTemperatureError(HeatExchangerError):
    """Invalid temperature value in heat exchanger calculation"""
    
    def __init__(self, temp_name: str, temp_value: float, h: float,
                 tci: float, thi: float, pci: float, phi: float,
                 debitc: float, debith: float):
        self.temp_name = temp_name
        self.temp_value = temp_value
        self.h = h
        self.tci = tci
        self.thi = thi
        self.pci = pci
        self.phi = phi
        self.debitc = debitc
        self.debith = debith
        super().__init__(
            f"Invalid temperature {temp_name}: {temp_value}\n"
            f"Heat transfer coefficient: h={h} W/m²/K\n"
            f"Conditions: Tci={tci}°C, Thi={thi}°C, Pci={pci} bar, Phi={phi} bar, "
            f"Debitc={debitc} m³/h, Debith={debith} l/s"
        )
