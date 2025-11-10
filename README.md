# Face Recognition Attendance System

A FastAPI-based face recognition attendance system that allows registering employees, marking attendance, and managing visitors using facial recognition.

## Features

- **Employee Registration**: Register employees with their face images.
- **Attendance Marking**: Automatically mark attendance by recognizing faces.
- **Visitor Management**: Detect and manage unknown visitors.
- **Face Recognition**: Uses dlib and OpenCV for accurate face detection and recognition.
- **Database Integration**: Uses SQLite with SQLAlchemy for data persistence.

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd FastApi
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the required dlib models and place them in the `models/` directory:
   - `dlib_face_recognition_resnet_model_v1.dat`
   - `shape_predictor_68_face_landmarks.dat`

## Usage

1. Run the application:

   ```bash
   uvicorn main:app --reload
   ```

2. Access the API documentation at `http://127.0.0.1:8000/docs`

## API Endpoints

### Employee Management

- `POST /register`: Register a new employee with name and base64-encoded image.
- `GET /employees`: Retrieve all employees.
- `PUT /employee/{employee_id}`: Update employee information.
- `DELETE /employee/{employee_id}`: Delete an employee.

### Attendance

- `POST /attendance`: Mark attendance by uploading a base64-encoded image.
- `GET /attendance/all`: Get all attendance records.

### Visitor Management

- `POST /visitor`: Create a new visitor with details and image.
- `GET /visitors`: Retrieve all visitors.
- `PUT /visitor/approve/{visitor_id}`: Approve or reject a visitor.
- `PUT /visitor/{visitor_id}`: Update visitor information.
- `DELETE /visitor/{visitor_id}`: Delete a visitor.

## Models

### Request Models

- `RegisterRequest`: name (str), image (base64 str)
- `AttendanceResponse`: status (str), message (str), data (dict)
- `VisitorApprovalRequest`: decision (str: "approve" or "reject")
- `CreateVisitorRequest`: name (str), person_to_meet (str), image (base64 str)
- `UpdateEmployeeRequest`: name (str)
- `UpdateVisitorRequest`: name (str, optional), person_to_meet (str, optional), status (str, optional)

### Response Models

- `AttendanceRecord`: id (int), name (str), date (str), time (str), timestamp (str)
- `VisitorRecord`: id (int), name (str), person_to_meet (str), status (str), image_path (str), timestamp (str)

## Database Schema

- **Employee**: id, name, face_encoding (JSON), image_path
- **Attendance**: id, name, date, time, timestamp
- **Visitor**: id, name, person_to_meet, status, image_path, face_encoding (JSON), timestamp

## Dependencies

- FastAPI: Web framework
- Uvicorn: ASGI server
- Dlib: Face recognition library
- OpenCV: Image processing
- NumPy: Numerical computations
- SQLAlchemy: ORM for database
- Pydantic: Data validation
- Python-multipart: Multipart form data handling

## Configuration

- Database: SQLite (`attendance.db`)
- Upload Directory: `uploads/`
- Models Directory: `models/`

## Error Handling

The API includes proper error handling for:

- Invalid image data
- No face detected
- Database errors
- Missing resources

## License

[Add your license here]
