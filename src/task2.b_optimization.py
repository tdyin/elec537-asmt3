import os
from ultralytics import YOLO
from common import load_config, ensure_dir, get_model_size
from common import download_model_if_needed


class ModelOptimizer:
    def __init__(self, config_path="src/config.yaml"):
        self.config = load_config(config_path)
        self.model_config = self.config['model']
        self.paths = self.config['paths']
        self.task2_config = self.config['task2']
        
        # Create necessary directories
        ensure_dir(self.paths['models_dir'])
        ensure_dir(self.paths['validation_images'])
        
        self.base_model_path = download_model_if_needed(
            self.model_config['name'], self.paths['models_dir']
        )
    
    def export_quantized_models(self):
        print("\nExporting quantized models...")
        model = YOLO(self.base_model_path)
        exported_models = {}
        
        for format_type in self.task2_config['quantization']['formats']:
            print(f"\nExporting {format_type} model...")
            
            try:
                if format_type == "fp32":
                    export_path = f"{self.paths['models_dir']}/{self.model_config['name']}_fp32.onnx"
                    model.export(format='onnx', half=False, simplify=True, opset=17)
                    default_export = self.base_model_path.replace('.pt', '.onnx')
                    if os.path.exists(default_export):
                        os.rename(default_export, export_path)
                    exported_models[format_type] = export_path
                    print(f"Exported {format_type.upper()}: {get_model_size(export_path):.2f} MB")
                
                elif format_type == "fp16":
                    export_path = f"{self.paths['models_dir']}/{self.model_config['name']}_fp16.onnx"
                    model.export(format='onnx', half=False, simplify=True, opset=12)  # Changed: simplify=False, opset=12
                    default_export = self.base_model_path.replace('.pt', '.onnx')
                    if os.path.exists(default_export):
                        try:
                            import onnx
                            from onnxmltools.utils.float16_converter import convert_float_to_float16
                            
                            onnx_model = onnx.load(default_export)
                            onnx_model_fp16 = convert_float_to_float16(onnx_model)
                            onnx.save(onnx_model_fp16, export_path)
                            os.remove(default_export)
                            exported_models[format_type] = export_path
                            print(f"Exported {format_type.upper()}: {get_model_size(export_path):.2f} MB")
                        except ImportError:
                            print("Installing onnxmltools...")
                            import subprocess
                            subprocess.run(["pip", "install", "onnxmltools"])

                elif format_type == "int8":
                    export_path = f"{self.paths['models_dir']}/{self.model_config['name']}_int8.onnx"
                    model.export(format='onnx', dynamic=True, simplify=True, opset=17)
                    default_export = self.base_model_path.replace('.pt', '.onnx')
                    
                    if os.path.exists(default_export):
                        try:
                            from onnxruntime.quantization import quantize_dynamic, QuantType
                            quantize_dynamic(model_input=default_export, model_output=export_path, weight_type=QuantType.QUInt8)
                            os.remove(default_export)
                            exported_models[format_type] = export_path
                            print(f"Exported {format_type.upper()}: {get_model_size(export_path):.2f} MB")
                        except ImportError:
                            print("Warning: onnxruntime not available. Run: pip install onnxruntime>=1.16.0")
                            if os.path.exists(default_export):
                                os.remove(default_export)
                            continue
                    else:
                        print(f"Warning: ONNX export not found at {default_export}")
                
            except Exception as e:
                print(f"Error exporting {format_type}: {e}")
        
        return exported_models


if __name__ == "__main__":
    optimizer = ModelOptimizer()
    exported_models = optimizer.export_quantized_models()
    print(f"\nExport completed: {list(exported_models.keys())}")
