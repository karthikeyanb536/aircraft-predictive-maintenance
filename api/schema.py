from pydantic import BaseModel, Field

class SensorInput(BaseModel):
    cycle: float = Field(..., description="Current engine cycle number", example=50)
    op_setting_1: float = Field(..., description="Operational setting 1", example=0.0023)
    op_setting_2: float = Field(..., description="Operational setting 2", example=0.0003)
    op_setting_3: float = Field(..., description="Operational setting 3", example=100.0)
    sensor_2: float = Field(..., description="Fan inlet temperature", example=642.0)
    sensor_3: float = Field(..., description="LPC outlet temperature", example=1583.0)
    sensor_4: float = Field(..., description="HPC outlet temperature", example=1396.0)
    sensor_7: float = Field(..., description="Fan inlet pressure", example=554.0)
    sensor_9: float = Field(..., description="Physical fan speed", example=9065.0)
    sensor_11: float = Field(..., description="Bypass ratio", example=47.5)
    sensor_12: float = Field(..., description="Burner fuel-air ratio", example=521.0)
    sensor_14: float = Field(..., description="HPT coolant bleed", example=8140.0)
    sensor_17: float = Field(..., description="Turbine inlet temperature", example=392.0)
    sensor_20: float = Field(..., description="Bypass ratio 2", example=38.8)
    sensor_21: float = Field(..., description="Fan speed ratio", example=23.2)

class PredictionOutput(BaseModel):
    predicted_rul: float = Field(..., description="Predicted Remaining Useful Life in cycles")
    status: str = Field(..., description="Engine health status")
    confidence: str = Field(..., description="Prediction confidence level")