import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import cv2
import numpy as np
from sympy import Line
import sympy
from scipy.spatial import distance

class CourtLineDetector:
    def __init__(self, model_path):
        # Load the pre-trained model of ResNet50
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # Modify the last layer to output 14 keypoints (x, y) for the court lines
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2)
        # Load the trained weights
        self.device = torch.device('mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available else 'cpu')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        # Transform to preprocess the input image
        # Resize to 224x224, convert to tensor, and normalize
        # using ImageNet statistics
        # Note: The input image should be in RGB format
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_keypoints(self, image):
        # Convert the image to RGB format and apply the transformations
        # Note: The input image should be in BGR format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize to 224x224, convert to tensor, and normalize
        # After the transform, the image tensor will be of shape (3, 224, 224)
        # unsequeeze(0) adds a batch dimension, so the shape becomes (1, 3, 224, 224)
        # and the model expects input of shape (batch_size, 3, 224, 224)
        # The model will output a tensor of shape (1, 28) for 14 keypoints (x, y)
        # The output will be reshaped to (14, 2) for the keypoints
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        # Disable gradient calculation in PyTorch
        # to save memory and computation
        # since we are only doing inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
        # The output tensor will be of shape (1, 28)
        # and we need to reshape it to (14, 2)
        # Move the output tensor to CPU and reshape it
        # to (14, 2) for the keypoints
        # squeeze() removes the batch dimension
        # and the shape becomes (28,) from (1, 28)
        # numpy() converts the tensor to a numpy array, 
        # can only be done on CPU
        keypoints = outputs.squeeze().cpu().numpy()
        # image_rgb is of shape (h, w, 3)
        original_h, original_w = image_rgb.shape[:2]
        # The keypoints are in the range [0, 1]
        # and we need to scale them to the original image size
        # The original image size is (original_h, original_w)
        # and we need to scale the keypoints to that size
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0
        self.predicted_keypoints = [(int(keypoints[i]), int(keypoints[i + 1])) for i in range(0, len(keypoints), 2)]

    def refine_keypoints(self, image):
        self.refined_keypoints = []
        index = 0
        for keypoint in self.predicted_keypoints:
            if index not in [8, 12, 9]:
                self.refined_keypoints.append(self.refine_keypoint(image, keypoint))
            else:
                self.refined_keypoints.append(None)
            index += 1
        
    def refine_keypoint(self, image, keypoint, crop_size = 40):
        refined_x, refined_y = keypoint[0], keypoint[1]
        height, width = image.shape[:2]
        x_min = max(refined_x - crop_size, 0)
        x_max = min(width, refined_x + crop_size)
        y_min = max(refined_y - crop_size, 0)
        y_max = min(height, refined_y + crop_size)

        # Image coordinates are in (y, x) format
        img_crop = image[y_min:y_max, x_min:x_max]
        # Line coordinates are in (x1, y1, x2, y2) format
        lines = self.detect_lines(img_crop)
        
        if len(lines) > 1:
            lines = self.merge_lines(lines)
            if len(lines) == 2:
                inters = self.line_intersection(lines[0], lines[1])
                if inters:
                    new_x = int(inters[0])
                    new_y = int(inters[1])
                    if new_x > 0 and new_x < img_crop.shape[0] and new_y > 0 and new_y < img_crop.shape[1]:
                        refined_x = x_min + new_x
                        refined_y = y_min + new_y
                        return (refined_x, refined_y)
        return None

    def detect_lines(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)[1]
        lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 30, minLineLength=10, maxLineGap=30)
        lines = np.squeeze(lines) 
        if len(lines.shape) > 0:
            if len(lines) == 4 and not isinstance(lines[0], np.ndarray):
                lines = [lines]
        else:
            lines = []
        return lines

    def line_intersection(self, line1, line2):
        l1 = Line((line1[0], line1[1]), (line1[2], line1[3]))
        l2 = Line((line2[0], line2[1]), (line2[2], line2[3]))

        intersection = l1.intersection(l2)
        point = None
        if len(intersection) > 0:
            if isinstance(intersection[0], sympy.geometry.point.Point2D):
                point = intersection[0].coordinates
        return point

    def merge_lines(self, lines):
        lines = sorted(lines, key=lambda item: item[0])
        mask = [True] * len(lines)
        new_lines = []

        for i, line in enumerate(lines):
            if mask[i]:
                for j, s_line in enumerate(lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        dist1 = distance.euclidean((x1, y1), (x3, y3))
                        dist2 = distance.euclidean((x2, y2), (x4, y4))
                        if dist1 < 20 and dist2 < 20:
                            line = np.array([int((x1+x3)/2), int((y1+y3)/2), int((x2+x4)/2), int((y2+y4)/2)],
                                            dtype=np.int32)
                            mask[i + j + 1] = False
                new_lines.append(line)  
        return new_lines       

    def set_homographied_keypoints(self, homographied_keypoints):
        self.homographied_keypoints = homographied_keypoints

    # Draw keypoints on the image    
    def draw_keypoints(self, image):
        for keypoint in self.homographied_keypoints:
            cv2.circle(image, keypoint, 5, (0, 255, 255), -1)
        return image
    
    # Draw keypoints on the video frames
    # Note: The input video frames should be in BGR format
    def draw(self, video_frames):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame)
            output_video_frames.append(frame)
        return output_video_frames
    
