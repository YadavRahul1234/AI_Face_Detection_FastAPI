from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from sqlalchemy.orm import Session
import os
import shutil
from datetime import datetime
import json
import cv2
import numpy as np
from typing import List
from database import get_db, Employee, Attendance, Visitor
from models import RegisterRequest, AttendanceResponse, VisitorApprovalRequest, AttendanceRecord, VisitorRecord, UpdateEmployeeRequest, CreateVisitorRequest, UpdateVisitorRequest
from face_utils import encode_face, match_face, load_encodings_from_db, decode_base64_image

app = FastAPI(title="Face Recognition Attendance System")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/register", response_model=AttendanceResponse)
async def register_employee(request: RegisterRequest, db: Session = Depends(get_db)):
    # Decode base64 image
    image = decode_base64_image(request.image)

    # Save image
    file_path = os.path.join(UPLOAD_DIR, f"{request.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    cv2.imwrite(file_path, image)

    # Encode face
    encoding = encode_face(image)
    if encoding is None:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="No face detected in the image")

    # Store in DB
    encoding_json = json.dumps(encoding.tolist())
    db_employee = Employee(name=request.name, face_encoding=encoding_json, image_path=file_path)
    db.add(db_employee)
    db.commit()
    db.refresh(db_employee)

    return AttendanceResponse(
        status="success",
        message=f"Employee {request.name} registered successfully",
        data={"employee_id": db_employee.id}
    )

@app.post("/attendance", response_model=AttendanceResponse)
async def mark_attendance(image: str = Form(...), db: Session = Depends(get_db)):
    # Decode base64 image
    try:
        image = decode_base64_image(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # Encode face
    encoding = encode_face(image)
    if encoding is None:
        raise HTTPException(status_code=400, detail="No face detected in the image")

    # Get all known encodings for employees and visitors
    employees = db.query(Employee).all()
    visitors = db.query(Visitor).filter(Visitor.face_encoding.isnot(None)).all()

    employee_encodings = load_encodings_from_db([emp.face_encoding for emp in employees])
    visitor_encodings = load_encodings_from_db([vis.face_encoding for vis in visitors])

    employee_names = [emp.name for emp in employees]
    visitor_ids = [vis.id for vis in visitors]

    # Match against employees first
    matched_employee, emp_distance = match_face(encoding, employee_encodings)

    if matched_employee:
        name = employee_names[int(matched_employee.split('_')[1])]
        # Mark attendance
        now = datetime.now()
        db_attendance = Attendance(
            name=name,
            date=now.strftime("%Y-%m-%d"),
            time=now.strftime("%H:%M:%S")
        )
        db.add(db_attendance)
        db.commit()
        return AttendanceResponse(
            status="success",
            message=f"Attendance marked for {name}",
            data={"name": name, "date": db_attendance.date, "time": db_attendance.time}
        )

    # Match against visitors
    matched_visitor, vis_distance = match_face(encoding, visitor_encodings)

    if matched_visitor:
        visitor_id = visitor_ids[int(matched_visitor.split('_')[1])]
        visitor = db.query(Visitor).filter(Visitor.id == visitor_id).first()
        return AttendanceResponse(
            status="success",
            message=f"Returning visitor {visitor.name} detected",
            data={"visitor_id": visitor_id, "name": visitor.name, "status": visitor.status}
        )
    else:
        # Save as new visitor
        file_path = os.path.join(UPLOAD_DIR, f"visitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(file_path, image)
        db_visitor = Visitor(image_path=file_path)
        db.add(db_visitor)
        db.commit()
        db.refresh(db_visitor)
        return AttendanceResponse(
            status="success",
            message="Unknown Face Detected",
            data={"visitor_id": db_visitor.id, "status": "visitor"}
        )

@app.get("/attendance/all", response_model=List[AttendanceRecord])
async def get_attendance_records(db: Session = Depends(get_db)):
    records = db.query(Attendance).all()
    return [
        AttendanceRecord(
            id=record.id,
            name=record.name,
            date=record.date,
            time=record.time,
            timestamp=record.timestamp.isoformat()
        ) for record in records
    ]

@app.put("/visitor/approve/{visitor_id}", response_model=AttendanceResponse)
async def approve_visitor(
    visitor_id: int,
    request: VisitorApprovalRequest,
    db: Session = Depends(get_db)
):
    visitor = db.query(Visitor).filter(Visitor.id == visitor_id).first()
    if not visitor:
        raise HTTPException(status_code=404, detail="Visitor not found")

    if request.decision.lower() == "approve":
        visitor.status = "Approved"
        message = f"Visitor {visitor_id} approved"
    elif request.decision.lower() == "reject":
        visitor.status = "Rejected"
        message = f"Visitor {visitor_id} rejected"
    else:
        raise HTTPException(status_code=400, detail="Invalid decision")

    db.commit()
    return AttendanceResponse(status="success", message=message)

@app.get("/visitors", response_model=List[VisitorRecord])
async def get_visitors(db: Session = Depends(get_db)):
    visitors = db.query(Visitor).all()
    return [
        VisitorRecord(
            id=v.id,
            name=v.name,
            person_to_meet=v.person_to_meet,
            status=v.status,
            image_path=v.image_path,
            timestamp=v.timestamp.isoformat()
        ) for v in visitors
    ]

@app.get("/employees", response_model=List[dict])
async def get_employees(db: Session = Depends(get_db)):
    employees = db.query(Employee).all()
    return [
        {
            "id": emp.id,
            "name": emp.name,
            "image_path": emp.image_path
        } for emp in employees
    ]

@app.put("/employee/{employee_id}", response_model=AttendanceResponse)
async def update_employee(
    employee_id: int,
    request: UpdateEmployeeRequest,
    db: Session = Depends(get_db)
):
    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    employee.name = request.name
    db.commit()
    return AttendanceResponse(
        status="success",
        message=f"Employee {employee_id} updated successfully",
        data={"employee_id": employee_id, "name": request.name}
    )

@app.delete("/employee/{employee_id}", response_model=AttendanceResponse)
async def delete_employee(employee_id: int, db: Session = Depends(get_db)):
    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    # Remove image file if exists
    if os.path.exists(employee.image_path):
        os.remove(employee.image_path)

    db.delete(employee)
    db.commit()
    return AttendanceResponse(
        status="success",
        message=f"Employee {employee_id} deleted successfully"
    )

@app.post("/visitor", response_model=AttendanceResponse)
async def create_visitor(request: CreateVisitorRequest, db: Session = Depends(get_db)):
    # Decode base64 image
    image = decode_base64_image(request.image)

    # Save image
    file_path = os.path.join(UPLOAD_DIR, f"visitor_{request.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    cv2.imwrite(file_path, image)

    # Encode face
    encoding = encode_face(image)
    if encoding is None:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="No face detected in the image")

    # Store in DB
    encoding_json = json.dumps(encoding.tolist())
    db_visitor = Visitor(name=request.name, person_to_meet=request.person_to_meet, image_path=file_path, face_encoding=encoding_json)
    db.add(db_visitor)
    db.commit()
    db.refresh(db_visitor)

    return AttendanceResponse(
        status="success",
        message=f"Visitor {request.name} created successfully",
        data={"visitor_id": db_visitor.id}
    )

@app.put("/visitor/{visitor_id}", response_model=AttendanceResponse)
async def update_visitor(
    visitor_id: int,
    request: UpdateVisitorRequest,
    db: Session = Depends(get_db)
):
    visitor = db.query(Visitor).filter(Visitor.id == visitor_id).first()
    if not visitor:
        raise HTTPException(status_code=404, detail="Visitor not found")

    if request.name is not None:
        visitor.name = request.name
    if request.person_to_meet is not None:
        visitor.person_to_meet = request.person_to_meet
    if request.status is not None:
        visitor.status = request.status

    db.commit()
    return AttendanceResponse(
        status="success",
        message=f"Visitor {visitor_id} updated successfully"
    )

@app.delete("/visitor/{visitor_id}", response_model=AttendanceResponse)
async def delete_visitor(visitor_id: int, db: Session = Depends(get_db)):
    visitor = db.query(Visitor).filter(Visitor.id == visitor_id).first()
    if not visitor:
        raise HTTPException(status_code=404, detail="Visitor not found")

    # Remove image file if exists
    if os.path.exists(visitor.image_path):
        os.remove(visitor.image_path)

    db.delete(visitor)
    db.commit()
    return AttendanceResponse(
        status="success",
        message=f"Visitor {visitor_id} deleted successfully"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
