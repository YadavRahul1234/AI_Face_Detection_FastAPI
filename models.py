from pydantic import BaseModel
from typing import Optional, List

class RegisterRequest(BaseModel):
    name: str
    image: str  # base64 encoded image



class AttendanceResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict] = None

class VisitorApprovalRequest(BaseModel):
    decision: str  # "approve" or "reject"

class AttendanceRecord(BaseModel):
    id: int
    name: str
    date: str
    time: str
    timestamp: str

class VisitorRecord(BaseModel):
    id: int
    name: Optional[str]
    person_to_meet: Optional[str]
    status: str
    image_path: str
    timestamp: str

class UpdateEmployeeRequest(BaseModel):
    name: str

class CreateVisitorRequest(BaseModel):
    name: str
    person_to_meet: str
    image: str  # base64 encoded image

class UpdateVisitorRequest(BaseModel):
    name: Optional[str] = None
    person_to_meet: Optional[str] = None
    status: Optional[str] = None
