import os
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO


class ModelOptimizer:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.paths = self.config['paths']
        self.task2_config = self.config['task2']
        
        # Create necessary directories
        Path(self.paths['models_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.paths['validation_images']).mkdir(parents=True, exist_ok=True)
        
        self.base_model_path = f"{self.paths['models_dir']}/{self.model_config['name']}.pt"
        
        if not os.path.exists(self.base_model_path):
            print("Model not found locally. Downloading...")
            model = YOLO(self.model_config['name'])
            model.save(self.base_model_path)
    
    def export_quantized_models(self):
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
                
                # FP16 Quantization - Export to ONNX with FP16
                elif format_type == "fp16":
                    export_path = f"{self.paths['models_dir']}/{self.model_config['name']}_fp16.onnx"
                    
                    # Export to ONNX with half precision and opset 17 for compatibility
                    model.export(format='onnx', half=True, simplify=True, opset=17)
                    
                    # Rename to target path
                    default_export = self.base_model_path.replace('.pt', '.onnx')
                    if os.path.exists(default_export):
                        os.rename(default_export, export_path)
                    exported_models[format_type] = export_path
                    print(f"Exported FP16 ONNX model: {export_path}")
                    print(f"  Size: {self.get_model_size(export_path):.2f} MB")
                
                # INT8 Quantization - Export to ONNX with INT8
                elif format_type == "int8":
                    export_path = f"{self.paths['models_dir']}/{self.model_config['name']}_int8.onnx"
                    
                    # Export to ONNX with dynamic quantization and opset 17 for compatibility
                    model.export(format='onnx', dynamic=True, simplify=True, opset=17)
                    
                    default_export = self.base_model_path.replace('.pt', '.onnx')
                    
                    if os.path.exists(default_export):
                        # Apply INT8 quantization using onnxruntime
                        try:
                            from onnxruntime.quantization import quantize_dynamic, QuantType
                            
                            quantize_dynamic(
                                model_input=default_export,
                                model_output=export_path,
                                weight_type=QuantType.QUInt8
                            )
                            os.remove(default_export)  # Clean up intermediate file
                            print(f"Exported INT8 ONNX model: {export_path}")
                            print(f"  Size: {self.get_model_size(export_path):.2f} MB")
                        except ImportError:
                            print("Warning: onnxruntime not available for INT8 quantization")
                            print("  Installing onnxruntime...")
                            os.system("pip install onnxruntime>=1.16.0")
                            print("  Please run the script again after installation.")
                            if os.path.exists(default_export):
                                os.remove(default_export)
                            continue
                        
                        exported_models[format_type] = export_path
                    else:
                        print(f"Warning: ONNX export not found at {default_export}")
                
            except Exception as e:
                print(f"Error exporting {format_type}: {e}")
                import traceback
                traceback.print_exc()
        
        return exported_models
    
    def get_model_size(self, model_path):
        """Get model file size in MB"""
        if os.path.exists(model_path):
            size_bytes = os.path.getsize(model_path)
            size_mb = size_bytes / (1024 * 1024)
            return size_mb
        return 0


if __name__ == "__main__":
    optimizer = ModelOptimizer()
    
    # Export quantized models
    exported_models = optimizer.export_quantized_models()
    print("\nEXPORT COMPLETED")
    print(f"Exported models: {list(exported_models.keys())}")
