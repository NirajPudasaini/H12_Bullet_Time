# Copyright (c) 2022-2025, The Isaac Lab Project Developers

"""Sensor implementations."""

from .capacitive_sensor import CapacitiveSensor
from .capacitive_sensor_cfg import CapacitiveSensorCfg
from .capacitive_sensor_data import CapacitiveSensorData

from .tof_sensor import TofSensor
from .tof_sensor_cfg import TofSensorCfg
from .tof_sensor_data import TofSensorData

from .cone_sensor import ConeSensor
from .cone_sensor_cfg import ConeSensorCfg

# Canonical aliases: shape-based naming
FieldSensor = CapacitiveSensor
FieldSensorCfg = CapacitiveSensorCfg
FieldSensorData = CapacitiveSensorData

RaySensor = TofSensor
RaySensorCfg = TofSensorCfg
RaySensorData = TofSensorData

ConeSensorData = CapacitiveSensorData

__all__ = [
    "CapacitiveSensor", "CapacitiveSensorCfg", "CapacitiveSensorData",
    "TofSensor", "TofSensorCfg", "TofSensorData",
    "ConeSensor", "ConeSensorCfg", "ConeSensorData",
    "FieldSensor", "FieldSensorCfg", "FieldSensorData",
    "RaySensor", "RaySensorCfg", "RaySensorData",
]
