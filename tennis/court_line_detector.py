import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import cv2
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import logging
from tennis.constants import RESNET50_PRE_TRAINED_MODEL_IMAGE_SIZE, COURT_KEYPOINTS, REFERENCE_KEYPOINTS
from tennis.utils import line_intersection, merge_lines, get_homography_matrix, transform_coordinates

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CourtLineDetector:
    
    def __init__(self):
        self.court_keypoints_path = 'analysis/court_keypoints.csv'

    def detect_keypoints(self, image: np.ndarray, model_path: str) -> None:
        predicted_keypoints = self._predict_keypoints(image, model_path)
        refined_predicted_keypoints = self._refine_keypoints(image, predicted_keypoints)
        homographied_keypoints = self._homograph_keypoints(refined_predicted_keypoints)
        refined_homographied_keypoints = self._refine_keypoints(image, homographied_keypoints)
        court_keypoints = self._homograph_keypoints(refined_homographied_keypoints)
        df = pd.DataFrame({
            'predicted_keypoints': predicted_keypoints,
            'refined_predicted_keypoints': refined_predicted_keypoints,
            'homographied_keypoints': homographied_keypoints,
            'refined_homographied_keypoints': refined_homographied_keypoints,
            'court_keypoints': court_keypoints
        })
        df.to_csv(self.court_keypoints_path, index=False)

    def _predict_keypoints(self, image: np.ndarray, model_path: str) -> List[Tuple[int, int]]:
        # Set up device
        device = 'mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # Modify the last layer to output 14 keypoints (x, y coordinates)
        model.fc = torch.nn.Linear(model.fc.in_features, COURT_KEYPOINTS * 2)
        # Load the trained weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((RESNET50_PRE_TRAINED_MODEL_IMAGE_SIZE, RESNET50_PRE_TRAINED_MODEL_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Convert BGR to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image_rgb is of shape (h, w, 3)
        original_h, original_w = image_rgb.shape[:2]
        
        # Preprocess the image
        try:
            # After the transform, the image tensor will be of shape (3, 224, 224)
            # unsequeeze(0) adds a batch dimension, so the shape becomes (1, 3, 224, 224)
            # and the model expects input of shape (batch_size, 3, 224, 224)
            image_tensor = transform(image_rgb).unsqueeze(0).to(device) # type: ignore
        except Exception as e:
            logger.error(f"Error during image transformation: {e}")
            raise
            
        # Run inference
        # Disable gradient calculation in PyTorch to save memory and computation
        # since we are only doing inference
        with torch.no_grad():
            outputs = model(image_tensor)
        
        # Process outputs
        # The output tensor will be of shape (1, 28), squeeze() removes the batch 
        # dimension and the shape becomes (28,), numpy() converts the tensor to 
        # a numpy array which can only be done on CPU
        keypoints = outputs.squeeze().cpu().numpy()
        
        # Scale keypoints to original image size
        keypoints[::2] *= original_w / RESNET50_PRE_TRAINED_MODEL_IMAGE_SIZE
        keypoints[1::2] *= original_h / RESNET50_PRE_TRAINED_MODEL_IMAGE_SIZE
        
        # Convert to list of (x, y) tuples
        return [(int(keypoints[i]), int(keypoints[i + 1])) for i in range(0, len(keypoints), 2)]

    def _refine_keypoints(self, image: np.ndarray, keypoints: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        refined_keypoints = []
        # Skip specific indices since they are most likely behind the net
        skip_indices = {8, 9, 12}
        
        for idx, keypoint in enumerate(keypoints):
            if idx not in skip_indices:
                refined = self._refine_keypoint(image, keypoint)
                refined_keypoints.append(refined)
            else:
                refined_keypoints.append(None)

        return refined_keypoints
        
    def _refine_keypoint(self, image: np.ndarray, keypoint: Tuple[int, int], 
                         crop_size: int = 40) -> Optional[Tuple[int, int]]:
        # Validate inputs
        if keypoint is None:
            return None
            
        refined_x, refined_y = keypoint
        height, width = image.shape[:2]
        
        # Define crop boundaries with bounds checking
        x_min = max(refined_x - crop_size, 0)
        x_max = min(refined_x + crop_size, width)
        y_min = max(refined_y - crop_size, 0)
        y_max = min(refined_y + crop_size, height)
        
        # Validate crop size
        if x_max <= x_min or y_max <= y_min:
            logger.warning(f"Invalid crop dimensions: x=[{x_min},{x_max}], y=[{y_min},{y_max}]")
            return None

        # Create crop
        img_crop = image[y_min:y_max, x_min:x_max]
        
        # Detect lines in the crop
        lines = self._detect_lines(img_crop)
        
        # Refine using line intersections
        if len(lines) < 2:
            return None
        
        lines = merge_lines(lines)
        if len(lines) != 2:
            return None

        intersection = line_intersection(lines[0], lines[1])
        if not intersection:
            return None
        
        new_x, new_y = intersection
        # Validate intersection is within crop
        crop_h, crop_w = img_crop.shape[:2]
        if 0 <= new_x < crop_w and 0 <= new_y < crop_h:
            # Map back to original image coordinates
            refined_x = x_min + new_x
            refined_y = y_min + new_y
            return (refined_x, refined_y)
        else:
            return None

    def _detect_lines(self, image: np.ndarray) -> List[np.ndarray]:
        if image.size == 0:
            logger.warning("Empty image provided to detect_lines")
            return []
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold to create binary image
        _, binary = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the image, doesn't work
        # kernel = np.ones((3, 3), np.uint8)
        # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Detect lines
        # The result will be in shape (N, 1, 4): 
        # [[[10,  20, 100,  20]], [[50,  30,  50, 150]], [[25,  75,  75, 125]]]
        lines_result = cv2.HoughLinesP(
            binary, 
            rho=1,
            theta=np.pi/180, 
            threshold=30, 
            minLineLength=10, 
            maxLineGap=30
        )
        
        # Process results
        if lines_result is None:
            return []

        # Lines will be in shape (N, 4):             
        # [[10,  20, 100,  20], [50,  30,  50, 150], [25,  75,  75, 125]]
        lines = np.squeeze(lines_result)

        # Handle different shapes of results
        if len(lines.shape) == 1:
            # Only one line detected: [25,  75,  75, 125]
            return [lines]
        elif len(lines.shape) == 2:
            return lines # type: ignore
        else:
            return []

    def _homograph_keypoints(self, given_keypoints: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        homography_matrix = get_homography_matrix(REFERENCE_KEYPOINTS, given_keypoints)
        # Apply homography to reference keypoints
        homographied_keypoints = transform_coordinates(REFERENCE_KEYPOINTS, homography_matrix)
        # Preserve detected keypoints where available
        for i in range(len(homographied_keypoints)):
            if given_keypoints[i] is not None:
                homographied_keypoints[i] = given_keypoints[i]
                
        return homographied_keypoints

    def load_court_keypoints(self) -> List[Tuple[int, int]]:
        df = pd.read_csv(self.court_keypoints_path)
        court_keypoints = df['court_keypoints'].tolist()
        court_keypoints = [eval(x) for x in court_keypoints]
        return court_keypoints
    
    def draw_keypoints(self, image: np.ndarray, court_keypoints: List[Tuple[int, int]]) -> np.ndarray:
        # Create a copy to avoid modifying the original
        result = image.copy()
        
        # Draw each keypoint
        for keypoint in court_keypoints:
            if keypoint is not None:
                cv2.circle(result, keypoint, 5, (0, 255, 255), -1)
                
        return result
    
    def draw(self, video_frames: List[np.ndarray]) -> List[np.ndarray]:
        court_keypints = self.load_court_keypoints()

        output_frames = []
        
        # Process each frame
        for frame in video_frames:
            frame_with_keypoints = self.draw_keypoints(frame, court_keypints)
            output_frames.append(frame_with_keypoints)
            
        return output_frames
