import os
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO


class ModelOptimizer:
    def __init__(self, config_path="config.yaml"):
        """Initialize the model optimizer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.paths = self.config['paths']
        self.task2_config = self.config['task2']
        
        # Create necessary directories
        Path(self.paths['models_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.paths['validation_images']).mkdir(parents=True, exist_ok=True)
        Path(f"{self.paths['results_dir']}/task2").mkdir(parents=True, exist_ok=True)
        
        # Load base model
        print(f"Loading base {self.model_config['name']} model...")
        self.base_model_path = f"{self.paths['models_dir']}/{self.model_config['name']}.pt"
        
        if not os.path.exists(self.base_model_path):
            print("Model not found locally. Downloading...")
            model = YOLO(self.model_config['name'])
            model.save(self.base_model_path)
    
    def export_quantized_models(self):
        """Export models in different quantization formats."""
        print("\n" + "="*60)
        print("EXPORTING QUANTIZED MODELS")
        print("="*60)
        
        model = YOLO(self.base_model_path)
        exported_models = {}
        
        for format_type in self.task2_config['quantization']['formats']:
            print(f"\nExporting {format_type} model...")
            
            try:
                if format_type == "fp32":
                    exported_models[format_type] = self.base_model_path
                    print(f"Using original model: {self.base_model_path}")
                
                # FP16 Quantization
                elif format_type == "fp16":
                    export_path = f"{self.paths['models_dir']}/{self.model_config['name']}_fp16.torchscript"
                    
                    # Half precision export
                    model.export(format='torchscript', half=True)
                    
                    default_export = self.base_model_path.replace('.pt', '.torchscript')
                    if os.path.exists(default_export):
                        os.rename(default_export, export_path)
                    exported_models[format_type] = export_path
                    print(f"Exported FP16 model: {export_path}")
                
                # INT8 Quantization
                elif format_type == "int8":
                    export_path = f"{self.paths['models_dir']}/{self.model_config['name']}_int8.torchscript"
                    
                    model.export(format='torchscript')
                    ts_path = self.base_model_path.replace('.pt', '.torchscript')
                    
                    # Load and quantize the TorchScript model
                    if os.path.exists(ts_path):
                        quantized_model = torch.jit.load(ts_path)
                        quantized_model = torch.quantization.quantize_dynamic(
                            quantized_model, {torch.nn.Linear}, dtype=torch.qint8
                        )
                        torch.jit.save(quantized_model, export_path)
                        os.remove(ts_path)  # Clean up intermediate file
                        exported_models[format_type] = export_path
                        print(f"Exported INT8 quantized model: {export_path}")
                    else:
                        print(f"Warning: TorchScript export not found at {ts_path}")
                
            except Exception as e:
                print(f"Error exporting {format_type}: {e}")
        
        return exported_models


if __name__ == "__main__":
    optimizer = ModelOptimizer()
    
    # Export quantized models
    exported_models = optimizer.export_quantized_models()
    
    print("\nModel optimization complete!")
    print(f"Exported models: {list(exported_models.keys())}")
    print("\nRun task2_comparison.py to benchmark and compare the models.")
