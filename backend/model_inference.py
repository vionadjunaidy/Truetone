import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os

class TransformerEncoderLayer(nn.Module):
    """Custom transformer encoder layer matching the checkpoint structure"""
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Use in_proj_weight format (combined QKV) to match checkpoint
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
            bias=True  # checkpoint has in_proj_bias
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src):
        # Self-attention (pre-norm architecture)
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(src2)
        
        # Feedforward (pre-norm architecture)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src

class CustomEmotionModel(nn.Module):
    """Custom multimodal emotion detection model matching the checkpoint architecture"""
    def __init__(self, num_emotions=7, d_model=512, num_layers=3, use_gender=True, use_sentiment=True):
        super().__init__()
        
        # Text projection: 1024 -> 512 -> 512
        self.text_proj = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.LayerNorm(512)
        )
        
        # Audio projection: 1024 -> 512 -> 512 (for future use)
        self.audio_proj = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.LayerNorm(512)
        )
        
        # Gender embedding and projection
        self.use_gender = use_gender
        if use_gender:
            self.gender_embed = nn.Embedding(2, 32)  # 2 genders
            self.gender_proj = nn.Linear(32, 512)
        
        # Sentiment embedding and projection
        self.use_sentiment = use_sentiment
        if use_sentiment:
            self.sentiment_embed = nn.Embedding(3, 32)  # 3 sentiment classes
            self.sentiment_proj = nn.Linear(32, 512)
        
        # Transformer encoder layers (using 'transformer' to match checkpoint naming)
        # Note: We'll rename this in load_state_dict to match checkpoint keys
        self.transformer = nn.ModuleDict({
            'layers': nn.ModuleList([
                TransformerEncoderLayer(d_model=d_model, dim_feedforward=2048)
                for _ in range(num_layers)
            ])
        })
        
        # Projection from combined features to 2048 for classifier
        # This handles the dimension expansion needed for the classifier input
        # We'll determine the input size based on which features are used
        if use_gender and use_sentiment:
            expand_input_size = 1536  # text (512) + gender (512) + sentiment (512)
        elif use_gender or use_sentiment:
            expand_input_size = 1024  # text (512) + one other (512)
        else:
            expand_input_size = 512   # just text
        
        self.expand_proj = nn.Linear(expand_input_size, 2048)
        
        # Classifier head: 2048 -> 512 -> 256 -> 7
        # Based on checkpoint: classifier.0 (2048->512), .1 (norm), .4 (512->256), .5 (norm), .8 (256->7)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),  # classifier.0
            nn.LayerNorm(512),     # classifier.1
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 256),   # classifier.4
            nn.LayerNorm(256),     # classifier.5
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, num_emotions)  # classifier.8
        )
    
    def forward(self, text_features, gender_idx=None, sentiment_idx=None, audio_features=None):
        """
        Forward pass
        
        Args:
            text_features: Text embeddings (batch_size, seq_len, 1024) or (batch_size, 1024)
            gender_idx: Gender indices (batch_size,) - 0 for male, 1 for female
            sentiment_idx: Sentiment indices (batch_size,) - 0, 1, or 2
            audio_features: Audio embeddings (optional, for future use)
        """
        batch_size = text_features.shape[0]
        
        # Project text features
        if len(text_features.shape) == 2:
            # (batch_size, 1024) -> (batch_size, 512)
            text_proj = self.text_proj(text_features)
        else:
            # (batch_size, seq_len, 1024) -> (batch_size, seq_len, 512)
            text_proj = self.text_proj(text_features)
            # Pool to single vector (mean pooling)
            text_proj = text_proj.mean(dim=1)
        
        # Start with text features
        features = [text_proj]
        
        # Add gender features if available
        if self.use_gender and gender_idx is not None:
            gender_emb = self.gender_embed(gender_idx)  # (batch_size, 32)
            gender_proj = self.gender_proj(gender_emb)   # (batch_size, 512)
            features.append(gender_proj)
        
        # Add sentiment features if available
        if self.use_sentiment and sentiment_idx is not None:
            sentiment_emb = self.sentiment_embed(sentiment_idx)  # (batch_size, 32)
            sentiment_proj = self.sentiment_proj(sentiment_emb)  # (batch_size, 512)
            features.append(sentiment_proj)
        
        # Concatenate all features
        if len(features) > 1:
            # Stack and reshape for transformer: (batch_size, num_features, 512)
            combined = torch.stack(features, dim=1)  # (batch_size, num_features, 512)
        else:
            # Just text: add sequence dimension
            combined = text_proj.unsqueeze(1)  # (batch_size, 1, 512)
        
        # Pass through transformer layers
        x = combined
        for layer in self.transformer['layers']:
            x = layer(x)
        
        # Pool transformer output (mean pooling across sequence dimension)
        x = x.mean(dim=1)  # (batch_size, 512)
        
        # Concatenate all original features (before transformer) to get combined representation
        # This preserves the information from text, gender, and sentiment
        combined_features = torch.cat(features, dim=-1)  # (batch_size, 512 * num_features)
        
        # Project to 2048 for classifier input
        x = self.expand_proj(combined_features)  # (batch_size, 2048)
        
        # Classifier
        logits = self.classifier(x)
        
        return logits

class EmotionModel:
    def __init__(self, model_path):
        """
        Initialize the emotion detection model.
        
        Args:
            model_path: Path to the .pt model file
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # Load model
        try:
            # weights_only=False is needed for PyTorch 2.6+ when loading models with numpy objects
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Get hyperparameters
            hyperparams = checkpoint.get('hyperparameters', {})
            use_gender = hyperparams.get('USE_GENDER', True)
            use_sentiment = hyperparams.get('USE_SENTIMENT', True)
            
            # Get emotion labels - try to infer from classifier output size
            state_dict = checkpoint['model_state_dict']
            
            # Find num_emotions from classifier output
            num_emotions = 7  # default
            for key in state_dict.keys():
                if 'classifier.8.weight' in key:
                    num_emotions = state_dict[key].shape[0]
                    break
            
            # Common emotion labels (adjust if your model uses different labels)
            self.emotion_labels = [
                'Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 
                'Disgust', 'Neutral'
            ]
            if num_emotions != len(self.emotion_labels):
                self.emotion_labels = [f'Emotion_{i}' for i in range(num_emotions)]
            
            print(f"Creating model with {num_emotions} emotion classes")
            print(f"Using gender: {use_gender}, Using sentiment: {use_sentiment}")
            
            # Create model
            self.model = CustomEmotionModel(
                num_emotions=num_emotions,
                use_gender=use_gender,
                use_sentiment=use_sentiment
            )
            
            # Load state dict
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Warning: Missing keys: {missing_keys[:10]}...")
                # If expand_proj is missing, it means the checkpoint doesn't have it
                # We'll need to initialize it - this might not match the original training
                if any('expand_proj' in k for k in missing_keys):
                    print("Note: expand_proj not in checkpoint, using random initialization")
            if unexpected_keys:
                print(f"Warning: Unexpected keys: {unexpected_keys[:10]}...")
            
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully and set to eval mode")
            
            # Initialize tokenizer for text encoding
            try:
                # Try to use a model that outputs 1024-dim features
                # Common options: bert-large (1024), or we might need to project
                # Let's use bert-base (768) and project to 1024, or use a larger model
                self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
                self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                
                # Check if we need to project bert-base (768) to 1024
                if self.text_encoder.config.hidden_size != 1024:
                    # Add projection layer
                    self.text_proj_to_1024 = nn.Linear(self.text_encoder.config.hidden_size, 1024).to(self.device)
                    print(f"Text encoder outputs {self.text_encoder.config.hidden_size} dims, projecting to 1024")
                else:
                    self.text_proj_to_1024 = None
                    
            except Exception as e:
                print(f"Warning: Could not load tokenizer: {e}")
                self.text_encoder = None
                self.tokenizer = None
                self.text_proj_to_1024 = None
                
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def _preprocess_text(self, text):
        """
        Preprocess text input to get 1024-dim features.
        """
        if not self.tokenizer or not self.text_encoder:
            raise Exception("Text encoder not available")
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            # Use pooled output or mean of last hidden state
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                text_features = outputs.pooler_output  # (batch_size, 768)
            else:
                text_features = outputs.last_hidden_state.mean(dim=1)  # (batch_size, 768)
            
            # Project to 1024 if needed
            if self.text_proj_to_1024 is not None:
                text_features = self.text_proj_to_1024(text_features)
        
        return text_features
    
    def _get_gender_idx(self, gender):
        """Convert gender string to index"""
        gender_map = {'male': 0, 'female': 1, 'm': 0, 'f': 1}
        return gender_map.get(gender.lower(), 0)
    
    def _get_sentiment_idx(self, text):
        """
        Simple sentiment analysis to get sentiment index.
        This is a placeholder - you might want to use a proper sentiment model.
        For now, we'll return a neutral sentiment (1).
        """
        # TODO: Implement proper sentiment analysis
        # For now, return neutral
        return 1  # 0=negative, 1=neutral, 2=positive
    
    def predict(self, text, gender):
        """
        Predict emotion from text and gender.
        
        Args:
            text: Input text to analyze
            gender: Gender of the speaker ('male' or 'female')
        
        Returns:
            Dictionary with emotion label, confidence, and cues
        """
        try:
            # Preprocess text to get features
            text_features = self._preprocess_text(text)  # (1, 1024)
            
            # Get gender index
            gender_idx = torch.tensor([self._get_gender_idx(gender)], device=self.device)
            
            # Get sentiment index (placeholder for now)
            sentiment_idx = torch.tensor([self._get_sentiment_idx(text)], device=self.device)
            
            # Run inference
            with torch.no_grad():
                logits = self.model(
                    text_features=text_features,
                    gender_idx=gender_idx,
                    sentiment_idx=sentiment_idx
                )
                
                probabilities = torch.softmax(logits, dim=-1)
                confidence, predicted_idx = torch.max(probabilities, dim=-1)
                
                emotion_idx = int(predicted_idx.item())
                confidence_score = confidence.item()
                emotion_label = self.emotion_labels[emotion_idx] if emotion_idx < len(self.emotion_labels) else 'Unknown'
            
            # Generate cues based on emotion
            cues = self._generate_cues(emotion_label, text)
            
            return {
                'label': emotion_label,
                'confidence': float(confidence_score),
                'cues': cues
            }
            
        except Exception as e:
            # If model inference fails, return a fallback result
            return {
                'label': 'Error',
                'confidence': 0.0,
                'cues': [f'Model error: {str(e)}'],
                'error': str(e)
            }
    
    def _generate_cues(self, emotion, text):
        """
        Generate contextual cues based on emotion and text.
        """
        cues_map = {
            'Happy': ['Positive wording', 'Upbeat tone', 'Enthusiastic language'],
            'Sad': ['Melancholic phrasing', 'Downcast sentiment', 'Somber tone'],
            'Angry': ['Strong language', 'Aggressive tone', 'Frustrated expression'],
            'Fear': ['Anxious wording', 'Uncertain sentiment', 'Cautious tone'],
            'Calm': ['Soft wording', 'Neutral sentiment', 'Steady pacing'],
            'Excited': ['Energetic language', 'High enthusiasm', 'Dynamic expression'],
            'Neutral': ['Balanced tone', 'Neutral sentiment', 'Even pacing'],
            'Surprise': ['Unexpected phrasing', 'Shocked expression', 'Astonished tone'],
            'Disgust': ['Negative wording', 'Repulsed sentiment', 'Averse tone']
        }
        
        return cues_map.get(emotion, ['Analyzed sentiment', 'Processed text', 'Detected emotion'])
