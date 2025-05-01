import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import cv2
import pandas as pd
import numpy as np
from sympy import Line
import sympy
from scipy.spatial import distance
from typing import List, Tuple, Optional
import logging
from .constants import RESNET50_PRE_TRAINED_MODEL_IMAGE_SIZE, COURT_KEYPOINTS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CourtLineDetector:
    """
    Detects and refines court line keypoints in images and videos.
    
    This class uses a pre-trained ResNet50 model to predict court line keypoints,
    refines them using traditional computer vision techniques, and provides
    visualization tools.
    """
    
    def __init__(self):
        """
        Initialize the court line detector with a pre-trained model.
        
        Args:
            model_path: Path to the pre-trained model weights
        """
        self.court_keypoints_path = 'analysis/court_keypoints.csv'

        # Initialize keypoint storage
        self.predicted_keypoints = None
        self.refined_predicted_keypoints = []
        self.homographied_keypoints = None
        self.refined_homographied_keypoints = []
        self.court_keypoints = None
        
    def _get_device(self) -> torch.device:
        """Determine the best available device for computation."""
        if torch.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def _initialize_model(self, model_path: str) -> torch.nn.Module:
        """Initialize and load the ResNet50 model with custom output layer."""
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # Modify the last layer to output 14 keypoints (x, y coordinates)
        model.fc = torch.nn.Linear(model.fc.in_features, COURT_KEYPOINTS * 2)
        
        try:
            # Load the trained weights
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
        model.to(self.device)
        # model.eval()  # Set to evaluation mode, doesn't work
        return model
    
    def _setup_transforms(self) -> transforms.Compose:
        """Setup image transformation pipeline for model input."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((RESNET50_PRE_TRAINED_MODEL_IMAGE_SIZE, RESNET50_PRE_TRAINED_MODEL_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_keypoints(self, image: np.ndarray, model_path: str) -> None:
        """
        Predict keypoints in the given image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of predicted keypoints as (x, y) tuples
        """
        # Set up device
        self.device = self._get_device()
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._initialize_model(model_path)
        logger.info("Initialised model")
        
        # Setup image transformation pipeline
        self.transform = self._setup_transforms()
        
        # Convert BGR to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image_rgb is of shape (h, w, 3)
        original_h, original_w = image_rgb.shape[:2]
        
        # Preprocess the image
        try:
            # After the transform, the image tensor will be of shape (3, 224, 224)
            # unsequeeze(0) adds a batch dimension, so the shape becomes (1, 3, 224, 224)
            # and the model expects input of shape (batch_size, 3, 224, 224)
            image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device) # type: ignore
        except Exception as e:
            logger.error(f"Error during image transformation: {e}")
            raise
            
        # Run inference
        # Disable gradient calculation in PyTorch to save memory and computation
        # since we are only doing inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # Process outputs
        # The output tensor will be of shape (1, 28), squeeze() removes the batch 
        # dimension and the shape becomes (28,), numpy() converts the tensor to 
        # a numpy array which can only be done on CPU
        keypoints = outputs.squeeze().cpu().numpy()
        
        # Scale keypoints to original image size
        keypoints[::2] *= original_w / RESNET50_PRE_TRAINED_MODEL_IMAGE_SIZE
        keypoints[1::2] *= original_h / RESNET50_PRE_TRAINED_MODEL_IMAGE_SIZE
        
        # Convert to list of (x, y) tuples
        self.predicted_keypoints = [(int(keypoints[i]), int(keypoints[i + 1])) 
                                    for i in range(0, len(keypoints), 2)]

    def refine_predicted_keypoints(self, image: np.ndarray) -> None:
        """
        Refine predicted keypoints using traditional CV techniques.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of refined keypoints, with None for points that couldn't be refined
        """
        if self.predicted_keypoints is None:
            logger.warning("No predicted keypoints available. Run predict_keypoints first.")
            return
            
        self.refined_predicted_keypoints = self._refine_keypoints(image, self.predicted_keypoints)
        
    def _refine_keypoints(self, image: np.ndarray, keypoints: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        refined_keypoints = []
        # Skip specific indices since they are most likely behind the net
        skip_indices = {8, 9, 12}
        
        for idx, keypoint in enumerate(keypoints):
            if idx not in skip_indices:
                refined = self.refine_keypoint(image, keypoint)
                refined_keypoints.append(refined)
            else:
                refined_keypoints.append(None)

        return refined_keypoints
        
    def refine_keypoint(self, image: np.ndarray, keypoint: Tuple[int, int], 
                         crop_size: int = 40) -> Optional[Tuple[int, int]]:
        """
        Refine a single keypoint by finding line intersections in its vicinity.
        
        Args:
            image: Input image
            keypoint: (x, y) coordinates of the keypoint to refine
            crop_size: Size of the crop around the keypoint
            
        Returns:
            Refined keypoint or None if refinement failed
        """
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
        lines = self.detect_lines(img_crop)
        
        # Refine using line intersections
        if len(lines) < 2:
            return None
        
        lines = self.merge_lines(lines)
        if len(lines) != 2:
            return None

        intersection = self.line_intersection(lines[0], lines[1])
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

    def detect_lines(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect lines in an image using Hough transform.
        
        Args:
            image: Input image
            
        Returns:
            List of detected lines as [x1, y1, x2, y2]
        """
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

    def line_intersection(self, line1: np.ndarray, line2: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Find the intersection point of two lines.
        
        Args:
            line1: First line as [x1, y1, x2, y2]
            line2: Second line as [x1, y1, x2, y2]
            
        Returns:
            Intersection point as (x, y) or None if lines don't intersect
        """
        try:
            # Create SymPy Line objects
            l1 = Line((line1[0], line1[1]), (line1[2], line1[3]))
            l2 = Line((line2[0], line2[1]), (line2[2], line2[3]))
            
            # Find intersection
            intersection = l1.intersection(l2) # type: ignore
            
            # Process result
            if not intersection or len(intersection) == 0:
                return None
            
            if isinstance(intersection[0], sympy.geometry.point.Point2D): # type: ignore
                x, y = intersection[0].coordinates
                return (int(x), int(y))
            else:
                return None
        except Exception as e:
            logger.warning(f"Error calculating line intersection: {e}")

    def merge_lines(self, lines: List[np.ndarray]) -> List[np.ndarray]:
        """
        Merge similar lines together.
        
        Args:
            lines: List of lines as [x1, y1, x2, y2]
            
        Returns:
            List of merged lines
        """
        if lines is None or len(lines) == 0:
            return []
            
        # Sort lines based on x-coordinate
        lines = sorted(lines, key=lambda line: line[0])
        
        # Track which lines have been merged
        mask = [True] * len(lines)
        merged_lines = []
        
        # Merge similar lines
        for i, line in enumerate(lines):
            if not mask[i]:
                continue

            curr_line = line.copy()
            
            # Check subsequent lines for possible merges
            for j in range(i + 1, len(lines)):
                if not mask[j]:
                    continue

                # Extract line coordinates
                x1, y1, x2, y2 = curr_line
                x3, y3, x4, y4 = lines[j]
                
                # Calculate distances between endpoints
                dist1 = distance.euclidean((x1, y1), (x3, y3))
                dist2 = distance.euclidean((x2, y2), (x4, y4))
                
                # If endpoints are close, merge the lines
                if dist1 < 20 and dist2 < 20:
                    curr_line = np.array([
                        int((x1 + x3) / 2), 
                        int((y1 + y3) / 2),
                        int((x2 + x4) / 2), 
                        int((y2 + y4) / 2)
                    ], dtype=np.int32)
                    
                    # Mark the merged line
                    mask[j] = False
            
            merged_lines.append(curr_line)
                
        return merged_lines

    def refine_homographied_keypoints(self, image: np.ndarray, homographied_keypoints: List[Tuple[int, int]]) -> None:
        """
        Set homographied keypoints for visualization.
        
        Args:
            homographied_keypoints: List of transformed keypoints
        """
        self.homographied_keypoints = homographied_keypoints
        self.refined_homographied_keypoints = self._refine_keypoints(image, homographied_keypoints)

    def set_court_keypoints(self, court_keypoints: List[Tuple[int, int]]) -> None:
        self.court_keypoints = court_keypoints
        df = pd.DataFrame({
            'predicted_keypoints': self.predicted_keypoints,
            'refined_predicted_keypoints': self.refined_predicted_keypoints,
            'homographied_keypoints': self.homographied_keypoints,
            'refined_homographied_keypoints': self.refined_homographied_keypoints,
            'court_keypoints': self.court_keypoints
        })
        df.to_csv(self.court_keypoints_path, index=False)

    def load_court_keypoints(self) -> List[Tuple[int, int]]:
        """
        Load court keypoints from CSV file.
        
        Returns:
            List of court keypoints
        """
        df = pd.read_csv(self.court_keypoints_path)
        court_keypoints = df['court_keypoints'].tolist()
        court_keypoints = [eval(x) for x in court_keypoints]
        return court_keypoints
    
    def draw_keypoints(self, image: np.ndarray, 
                       color: Tuple[int, int, int] = (0, 255, 255),
                       radius: int = 5) -> np.ndarray:
        """
        Draw keypoints on the image.
        
        Args:
            image: Input image
            color: BGR color for keypoints
            radius: Radius of keypoint circles
            
        Returns:
            Image with drawn keypoints
        """
        if self.court_keypoints is None:
            logger.warning("No homographied keypoints available")
            return image
            
        # Create a copy to avoid modifying the original
        result = image.copy()
        
        # Draw each keypoint
        for keypoint in self.court_keypoints:
            if keypoint is not None:
                cv2.circle(result, keypoint, radius, color, -1)
                
        return result
    
    def draw(self, video_frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Draw keypoints on video frames.
        
        Args:
            video_frames: List of video frames
            
        Returns:
            List of video frames with drawn keypoints
        """
        if not video_frames:
            logger.warning("No video frames provided")
            return []

        print(f'Predicted  keypoints: {self.predicted_keypoints}')    
        print(f'Refined    keypoints: {self.refined_predicted_keypoints}')    
        print(f'Homography keypoints: {self.homographied_keypoints}')    
        print(f'Re-refined keypoints: {self.refined_homographied_keypoints}')    
        print(f'Court      keypoints: {self.court_keypoints}')    

        output_frames = []
        
        # Process each frame
        for frame in video_frames:
            frame_with_keypoints = self.draw_keypoints(frame)
            output_frames.append(frame_with_keypoints)
            
        return output_frames
    
    def visualize_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Create a visualization of the detection process.
        
        Args:
            image: Input image
            
        Returns:
            Visualization image
        """
        # Create a copy of the image
        viz_image = image.copy()
        
        # Draw predicted keypoints in red
        # if self.predicted_keypoints:
        #     for point in self.predicted_keypoints:
        #         if point:
        #             cv2.circle(viz_image, point, 5, (0, 0, 255), -1)
        
        # Draw refined keypoints in green
        if self.refined_predicted_keypoints:
            for point in self.refined_predicted_keypoints:
                if point:
                    cv2.circle(viz_image, point, 3, (0, 255, 0), -1)
        
        # Draw homographied keypoints in yellow
        # if self.homographied_keypoints:
        #     for point in self.homographied_keypoints:
        #         if point:
        #             cv2.circle(viz_image, point, 7, (0, 255, 255), 2)
        
        return viz_image