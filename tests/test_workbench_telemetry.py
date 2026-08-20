from local_llm_control.workbench import _parse_mactop_metrics


def test_parse_mactop_metrics_filters_bogus_m1_gpu_sensors():
    text = """
mactop_power_watts{component="gpu"} 12.5
mactop_power_watts{component="cpu"} 1.25
mactop_power_watts{component="system"} 18.0
mactop_power_watts{component="total"} 31.75
mactop_gpu_freq_mhz 972
mactop_gpu_usage_percent 99.5
mactop_gpu_temp_celsius 0
mactop_temp_sensor_celsius{key="Tg05",name="GPU"} 9.2
mactop_temp_sensor_celsius{key="TRD0",name="GPU D0"} 61.0
mactop_temp_sensor_celsius{key="TRD1",name="GPU D1"} 63.0
mactop_fan_rpm{fan_id="0",fan_name="Fan 0"} 2000
mactop_fan_rpm{fan_id="1",fan_name="Fan 1"} 2200
mactop_thermal_state 1
"""
    telemetry = _parse_mactop_metrics(text)
    assert telemetry["available"] is True
    assert telemetry["gpu_power_w"] == 12.5
    assert telemetry["gpu_temp_c"] == 62.0
    assert telemetry["gpu_temp_max_c"] == 63.0
    assert telemetry["fan_rpm"] == 2100
    assert telemetry["thermal_state"] == "Fair"
