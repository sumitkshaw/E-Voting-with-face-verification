import face_recognition
import numpy as np
import cv2
import os

def Recognizer():
    """Recognizes a face using OpenCV and face_recognition."""
    
    video = cv2.VideoCapture(0)
    
    # Ensure the camera is opened
    if not video.isOpened():
        raise RuntimeError("Error: Unable to access the webcam.")
    
    known_face_encodings = []
    known_face_names = []

    # Load stored images
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, "static", "profile_pics")

    names = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith(('jpg', 'png')):
                path = os.path.join(root, file)
                img = face_recognition.load_image_file(path)

                # Convert to RGB to avoid "Unsupported Image Type" errors
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                label = file
                if label == 'default.jpg':
                    continue
                
                encodings = face_recognition.face_encodings(img)
                if encodings:  # Only add if a face is detected
                    img_encoding = encodings[0]
                    known_face_names.append(label)
                    known_face_encodings.append(img_encoding)

    face_locations = []
    face_encodings = []

    # Start webcam processing
    frame_count = 0
    max_frames = 50  # Approx. 5 seconds at 10 FPS

    while frame_count < max_frames:  
        check, frame = video.read()
        
        if not check:
            print("Error: Failed to read frame from webcam.")
            break

        # Convert to RGB (Fixes "Unsupported Image Type" error)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            if len(matches) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    face_names.append(name)
                    if name not in names:
                        names.append(name)

        # Display results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow("Face Recognition Panel", frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):  # Press 'q' to exit early
            break
        
        frame_count += 1  # Increment frame count

    video.release()
    cv2.destroyAllWindows()
    return names
