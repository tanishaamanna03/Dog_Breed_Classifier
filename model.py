import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import json
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import tarfile
from torch.cuda.amp import autocast, GradScaler

class DogBreedClassifier:
    def __init__(self, num_classes=120):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model with pretrained weights
        self.model = models.resnet50(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        self.model = self.model.to(self.device)
        
        # Enable cuDNN benchmarking for faster training
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        # Load breed information
        self.breed_info = self._load_breed_info()
        
        # Balanced transforms for training
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),  # Reduced rotation
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Reduced jitter
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def _load_breed_info(self):
        breed_info = {
            "n02085620": "Chihuahua - Small, alert, and loyal companion",
            "n02085782": "Japanese Spaniel - Elegant, intelligent, and affectionate",
            "n02085936": "Maltese - Gentle, playful, and fearless",
            "n02086079": "Pekinese - Independent, dignified, and loyal",
            "n02086240": "Shih-Tzu - Affectionate, playful, and outgoing",
            "n02086646": "Blenheim Spaniel - Friendly, gentle, and good with children",
            "n02086910": "Papillon - Happy, alert, and friendly",
            "n02087046": "Toy Terrier - Alert, active, and loyal",
            "n02087394": "Rhodesian Ridgeback - Dignified, strong-willed, and athletic",
            "n02088094": "Afghan Hound - Independent, dignified, and aloof",
            "n02088238": "Basset Hound - Patient, low-key, and charming",
            "n02088364": "Beagle - Merry, friendly, and curious",
            "n02088466": "Bloodhound - Patient, independent, and gentle",
            "n02088632": "Bluetick Coonhound - Confident, alert, and friendly",
            "n02089078": "Black and Tan Coonhound - Easygoing, bright, and active",
            "n02089867": "Walker Hound - Confident, alert, and friendly",
            "n02089973": "English Foxhound - Sociable, gentle, and independent",
            "n02090379": "Redbone Coonhound - Even-tempered, alert, and confident",
            "n02090622": "Borzoi - Independent, athletic, and quiet",
            "n02090721": "Irish Wolfhound - Patient, dignified, and courageous",
            "n02091032": "Italian Greyhound - Sensitive, alert, and athletic",
            "n02091134": "Whippet - Calm, affectionate, and athletic",
            "n02091244": "Ibizan Hound - Independent, active, and intelligent",
            "n02091467": "Norwegian Elkhound - Confident, friendly, and bold",
            "n02091635": "Otterhound - Amiable, boisterous, and even-tempered",
            "n02091831": "Saluki - Independent, gentle, and dignified",
            "n02092002": "Scottish Deerhound - Dignified, gentle, and polite",
            "n02092339": "Weimaraner - Fearless, alert, and friendly",
            "n02093256": "Staffordshire Bullterrier - Brave, tenacious, and affectionate",
            "n02093428": "American Staffordshire Terrier - Confident, smart, and good-natured",
            "n02093647": "Bedlington Terrier - Loyal, cheerful, and courageous",
            "n02093754": "Border Terrier - Alert, good-natured, and adaptable",
            "n02093859": "Kerry Blue Terrier - Alert, confident, and adaptable",
            "n02093991": "Irish Terrier - Bold, dashing, and tenderhearted",
            "n02094114": "Norfolk Terrier - Alert, fearless, and fun-loving",
            "n02094258": "Norwich Terrier - Alert, curious, and affectionate",
            "n02094433": "Yorkshire Terrier - Feisty, brave, and determined",
            "n02095314": "Wire-haired Fox Terrier - Alert, quick, and confident",
            "n02095570": "Lakeland Terrier - Friendly, bold, and independent",
            "n02095889": "Sealyham Terrier - Alert, outgoing, and fearless",
            "n02096051": "Airedale - Confident, smart, and courageous",
            "n02096177": "Cairn - Alert, independent, and hardy",
            "n02096294": "Australian Terrier - Alert, confident, and spirited",
            "n02096437": "Dandie Dinmont - Independent, proud, and smart",
            "n02096585": "Boston Bull - Friendly, bright, and amusing",
            "n02097047": "Miniature Schnauzer - Alert, friendly, and obedient",
            "n02097130": "Giant Schnauzer - Alert, dependable, and well-mannered",
            "n02097209": "Standard Schnauzer - Alert, fearless, and trainable",
            "n02097298": "Scotch Terrier - Independent, confident, and alert",
            "n02097474": "Tibetan Terrier - Confident, smart, and courageous",
            "n02097658": "Silky Terrier - Alert, friendly, and quick",
            "n02098105": "Soft-coated Wheaten Terrier - Happy, steady, and confident",
            "n02098286": "West Highland White Terrier - Alert, independent, and self-confident",
            "n02098413": "Lhasa - Confident, smart, and independent",
            "n02099267": "Flat-coated Retriever - Optimistic, outgoing, and active",
            "n02099429": "Curly-coated Retriever - Confident, proud, and trainable",
            "n02099601": "Golden Retriever - Intelligent, friendly, and devoted",
            "n02099712": "Labrador Retriever - Friendly, active, and outgoing",
            "n02099849": "Chesapeake Bay Retriever - Affectionate, bright, and sensitive",
            "n02100236": "German Short-haired Pointer - Friendly, smart, and willing to please",
            "n02100583": "Vizsla - Gentle, energetic, and loyal",
            "n02100735": "English Setter - Mellow, gentle, and family-oriented",
            "n02100877": "Irish Setter - Outgoing, sweet-natured, and active",
            "n02101006": "Gordon Setter - Alert, confident, and courageous",
            "n02101388": "Brittany Spaniel - Bright, upbeat, and fun-loving",
            "n02101556": "Clumber - Dignified, loyal, and gentle",
            "n02102040": "English Springer - Friendly, playful, and obedient",
            "n02102177": "Welsh Springer Spaniel - Friendly, active, and loyal",
            "n02102318": "Cocker Spaniel - Gentle, smart, and happy",
            "n02102480": "Sussex Spaniel - Cheerful, friendly, and devoted",
            "n02102973": "Irish Water Spaniel - Alert, quick to learn, and determined",
            "n02104029": "Kuvasz - Loyal, courageous, and patient",
            "n02104365": "Schipperke - Alert, independent, and confident",
            "n02105056": "Groenendael - Smart, alert, and hardworking",
            "n02105162": "Malinois - Smart, confident, and hardworking",
            "n02105251": "Briard - Smart, faithful, and protective",
            "n02105412": "Kelpie - Smart, work-oriented, and eager",
            "n02105505": "Komondor - Dignified, brave, and calm",
            "n02105641": "Old English Sheepdog - Adaptable, smart, and gentle",
            "n02105855": "Shetland Sheepdog - Bright, gentle, and strong",
            "n02106030": "Collie - Devoted, graceful, and proud",
            "n02106166": "Border Collie - Smart, energetic, and workaholic",
            "n02106382": "Bouvier des Flandres - Rational, protective, and strong-willed",
            "n02106550": "Rottweiler - Loyal, loving, and confident guardian",
            "n02106662": "German Shepherd - Loyal, confident, and courageous",
            "n02107142": "Doberman - Alert, fearless, and loyal",
            "n02107312": "Miniature Pinscher - Fearless, energetic, and alert",
            "n02107574": "Greater Swiss Mountain Dog - Good-natured, calm, and strong",
            "n02107683": "Bernese Mountain Dog - Good-natured, calm, and strong",
            "n02107908": "Appenzeller - Self-confident, reliable, and fearless",
            "n02108000": "EntleBucher - Self-confident, reliable, and fearless",
            "n02108089": "Boxer - Playful, bright, and active",
            "n02108422": "Bull Mastiff - Protective, loyal, and calm",
            "n02108551": "Tibetan Mastiff - Independent, reserved, and intelligent",
            "n02108915": "French Bulldog - Playful, smart, and adaptable",
            "n02109047": "Great Dane - Friendly, patient, and dependable",
            "n02109525": "Saint Bernard - Patient, gentle, and watchful",
            "n02109961": "Eskimo Dog - Friendly, alert, and loyal",
            "n02110063": "Malamute - Affectionate, loyal, and playful",
            "n02110185": "Siberian Husky - Independent, athletic, and mischievous",
            "n02110627": "Affenpinscher - Confident, fearless, and amusing",
            "n02110806": "Basenji - Independent, smart, and poised",
            "n02110958": "Pug - Charming, mischievous, and loving",
            "n02111129": "Leonberg - Playful, friendly, and devoted",
            "n02111277": "Newfoundland - Sweet-tempered, patient, and devoted",
            "n02111500": "Great Pyrenees - Patient, smart, and strong-willed",
            "n02111889": "Samoyed - Alert, friendly, and adaptable",
            "n02112018": "Pomeranian - Inquisitive, bold, and lively",
            "n02112137": "Chow - Dignified, bright, and serious-minded",
            "n02112350": "Keeshond - Alert, friendly, and intelligent",
            "n02112706": "Brabancon Griffon - Alert, loyal, and self-possessed",
            "n02113023": "Pembroke - Smart, alert, and affectionate",
            "n02113186": "Cardigan - Alert, active, and loyal",
            "n02113624": "Toy Poodle - Smart, alert, and trainable",
            "n02113712": "Miniature Poodle - Smart, alert, and trainable",
            "n02113799": "Standard Poodle - Smart, alert, and trainable",
            "n02113978": "Mexican Hairless - Alert, intelligent, and loyal",
            "n02115641": "Dingo - Independent, alert, and intelligent",
            "n02115913": "Dhole - Alert, intelligent, and social",
            "n02116738": "African Hunting Dog - Cooperative, intelligent, and loyal"
        }
        return breed_info
    
    def train(self, train_data_path, num_epochs=10):
        # Initialize loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Simple learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )
        
        # Load training data with balanced settings
        train_dataset = ImageFolder(train_data_path, transform=self.train_transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=64,  # Balanced batch size
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.model.train()
        best_accuracy = 0.0
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Use tqdm for progress bar
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
            
            # Calculate epoch statistics
            epoch_loss = running_loss / len(train_loader)
            accuracy = 100 * correct / total
            
            # Update learning rate based on accuracy
            scheduler.step(accuracy)
            
            print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save_model('best_model.pth')
                print(f'New best model saved with accuracy: {best_accuracy:.2f}%')
    
    def predict(self, image_path):
        self.model.eval()
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.val_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        # Get breed information
        breed_idx = predicted.item()
        breed_keys = list(self.breed_info.keys())
        if breed_idx < len(breed_keys):
            breed_key = breed_keys[breed_idx]
            breed_description = self.breed_info[breed_key]
            # Extract just the breed name (before the dash)
            breed_name = breed_description.split(' - ')[0]
        else:
            breed_name = "Unknown Breed"
            breed_description = "No description available"
        
        return {
            'breed': breed_name,
            'confidence': confidence.item(),
            'description': breed_description
        }
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval() 