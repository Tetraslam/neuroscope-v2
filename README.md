# Neuroscope

An interactive neural network visualization engine for both artificial and biological neural networks, designed to make neural architectures and their dynamics accessible and interpretable.

## 🚀 Quick Start

1. Start the services:
```bash
# Terminal 1: Start frontend
npm install
npm run dev

# Terminal 2: Start Python service (requires Python 3.11)
cd python_service
pip install -r requirements.txt
python3.11 -m uvicorn main:app --reload
```

2. Open http://localhost:3000/demo

3. Visualize your model by:
   - Dragging and dropping your PyTorch model (`.pt` or `.pth`)
   - Using example models in `python_service/models/`
   - Loading training data from `python_service/neuroscope_training.json`

## Training Visualization
```bash
# Terminal 1: Start frontend
npm run dev
```

```bash
# Terminal 2: Start Python service (requires Python 3.11)
```bash
cd python_service
python3.11 -m uvicorn main:app --reload
```

Open http://localhost:3000/training

Drop your neuroscope_training.json file into the input box.

## 🎯 Features

- Interactive visualization of neural networks using NextJS, TypeScript, and D3
- Support for PyTorch models, Spiking Neural Networks, and biological connectomes
- Real-time visualization of network dynamics and training processes
- Comprehensive analysis tools for network architecture and behavior

## 📚 Documentation

See `python_service/examples/README.md` for detailed instructions and examples.

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for more information.

---
*Neuroscope is currently under active development. Feature requests and contributions are welcome!*
