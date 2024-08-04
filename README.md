## Real-Time-Hand-Sign-Recognition-System



### 'datacollection.py'
This script is responsible for collecting and preparing the dataset used for training the hand sign recognition model. The main tasks include detecting hands, cropping the hand regions, and saving the cropped images under appropriate labels.
- **Hand Detection**: Capture frames, detect hands, and get bounding box coordinates.
- **Hand Cropping**: Crop and resize hand regions.
- **Labeling**: Prompt user for labels and save images under appropriate directories.
  
### 'test.py'
This script is designed to detect and classify hand signs in real-time, displaying the predicted label on the bounding box around the detected hand.
- **Hand Detection**: Capture frames and detect hands.
- **Classification**: Crop hand regions, preprocess, and classify using the model.
- **Display**: Draw bounding boxes and labels on detected hands.

### Applications
- **Sign Language Translation**: Facilitate communication for the hearing impaired by translating sign language into text or speech.
- **Gesture Control**: Enable gesture-based control systems for devices and applications.
- **Interactive Systems**: Enhance human-computer interaction in gaming, virtual reality, and other fields.
### Future Work
- Expand the dataset to include more hand signs and variations.
- Improve model robustness against different backgrounds and lighting conditions.
- Explore the integration of multimodal inputs, such as combining hand signs with facial expressions for more comprehensive recognition systems.
