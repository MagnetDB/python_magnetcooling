"""
Test waterflow_factory module - demonstrates WaterFlow object creation from various sources.
"""

import pytest
from python_magnetcooling.waterflow_factory import (
    from_flow_params,
    from_database_record,
    from_fitted_data,
    create_default,
)
from python_magnetcooling.waterflow import WaterFlow


def test_from_flow_params():
    """Test creating WaterFlow from flow_params dictionary (as saved by compute method)"""
    # This is the format produced by the compute() method in examples/flow_params.py
    params = {
        "Vp0": {"value": 1000, "unit": "rpm"},
        "Vpmax": {"value": 2840, "unit": "rpm"},
        "F0": {"value": 0, "unit": "l/s"},
        "Fmax": {"value": 140, "unit": "l/s"},
        "Pmax": {"value": 22, "unit": "bar"},
        "Pmin": {"value": 4, "unit": "bar"},
        "BP": {"value": 4, "unit": "bar"},
        "Imax": {"value": 28000, "unit": "A"},
    }
    
    flow = from_flow_params(params)
    
    assert isinstance(flow, WaterFlow)
    assert flow.pump_speed_min == 1000
    assert flow.pump_speed_max == 2840
    assert flow.flow_min == 0
    assert flow.flow_max == 140
    assert flow.pressure_max == 22
    assert flow.pressure_min == 4
    assert flow.pressure_back == 4
    assert flow.current_max == 28000


def test_from_flow_params_with_pout():
    """Test that both BP and Pout keys work for back pressure"""
    params = {
        "Vp0": {"value": 1000, "unit": "rpm"},
        "Vpmax": {"value": 2840, "unit": "rpm"},
        "F0": {"value": 0, "unit": "l/s"},
        "Fmax": {"value": 140, "unit": "l/s"},
        "Pmax": {"value": 22, "unit": "bar"},
        "Pmin": {"value": 4, "unit": "bar"},
        "Pout": {"value": 5, "unit": "bar"},  # Using Pout instead of BP
        "Imax": {"value": 28000, "unit": "A"},
    }
    
    flow = from_flow_params(params)
    assert flow.pressure_back == 5


def test_from_database_record_with_mapping():
    """Test creating WaterFlow from database record with custom key mapping"""
    # Simulated database record with custom field names
    record = {
        "min_pump_rpm": 1000,
        "max_pump_rpm": 2840,
        "min_flow_rate": 0,
        "max_flow_rate": 140,
        "max_pressure": 22,
        "min_pressure": 4,
        "back_pressure": 4,
        "max_current": 28000,
    }
    
    mapping = {
        "Vp0": "min_pump_rpm",
        "Vpmax": "max_pump_rpm",
        "F0": "min_flow_rate",
        "Fmax": "max_flow_rate",
        "Pmax": "max_pressure",
        "Pmin": "min_pressure",
        "BP": "back_pressure",
        "Imax": "max_current",
    }
    
    flow = from_database_record(record, mapping)
    
    assert flow.pump_speed_min == 1000
    assert flow.pump_speed_max == 2840
    assert flow.flow_max == 140


def test_from_database_record_standard_format():
    """Test creating WaterFlow from database record in standard format"""
    # Database record already in flow_params format
    record = {
        "Vp0": {"value": 1000},
        "Vpmax": {"value": 2840},
        "F0": {"value": 0},
        "Fmax": {"value": 140},
        "Pmax": {"value": 22},
        "Pmin": {"value": 4},
        "BP": {"value": 4},
        "Imax": {"value": 28000},
    }
    
    flow = from_database_record(record)
    assert flow.current_max == 28000


def test_create_default():
    """Test creating WaterFlow with default values"""
    flow = create_default()
    
    assert isinstance(flow, WaterFlow)
    assert flow.pump_speed_min == 1000
    assert flow.pump_speed_max == 2840
    assert flow.current_max == 28000


def test_from_fitted_data():
    """Test creating WaterFlow from fitted curve parameters"""
    # These would come from curve fitting experimental data
    pump_speed_fit = (2840, 1000)  # (Vpmax, Vp0)
    flow_rate_fit = (0, 140)       # (F0, Fmax)
    pressure_fit = (4, 22)         # (Pmin, Pmax)
    back_pressure = 4.0
    max_current = 28000
    
    flow = from_fitted_data(
        pump_speed_fit,
        flow_rate_fit,
        pressure_fit,
        back_pressure,
        max_current
    )
    
    assert flow.pump_speed_max == 2840
    assert flow.pump_speed_min == 1000
    assert flow.flow_max == 140
    assert flow.pressure_max == 22
    assert flow.pressure_min == 4


def test_waterflow_methods_work():
    """Verify that WaterFlow objects created by factory work correctly"""
    params = {
        "Vp0": {"value": 1000, "unit": "rpm"},
        "Vpmax": {"value": 2840, "unit": "rpm"},
        "F0": {"value": 0, "unit": "l/s"},
        "Fmax": {"value": 140, "unit": "l/s"},
        "Pmax": {"value": 22, "unit": "bar"},
        "Pmin": {"value": 4, "unit": "bar"},
        "BP": {"value": 4, "unit": "bar"},
        "Imax": {"value": 28000, "unit": "A"},
    }
    
    flow = from_flow_params(params)
    
    # Test the WaterFlow methods
    current = 20000  # A
    
    speed = flow.pump_speed(current)
    assert speed > 0
    
    flow_rate = flow.flow_rate(current)
    assert flow_rate > 0
    
    pressure = flow.pressure(current)
    assert pressure > 0
    
    pressure_drop = flow.pressure_drop(current)
    assert pressure_drop >= 0
    
    velocity = flow.velocity(current, 1e-4)  # 1e-4 m²
    assert velocity > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
